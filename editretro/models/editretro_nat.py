# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
from typing import NamedTuple, Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from editretro.models.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (Embedding, TransformerDecoderLayer)

from fairseq.models.nat import (FairseqNATModel, FairseqNATDecoder, FairseqNATEncoder,
                                ensemble_decoder)

from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq import utils
from editretro.models.levenshtein_utils import (
    _skip,
    _skip_encoder_out,
    _fill,
    _get_advanced_ins_targets,
    _get_advanced_reposition_targets,
    _apply_ins_masks,
    _apply_ins_words,
    _apply_reposition_words,
)

from fairseq.models import CompositeEncoder

# import pandas as pd
import numpy as np

from fairseq.modules import MultiheadAttention, LayerNorm

SEED = 2025


# MSK_INS = 16

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


setup_seed(SEED)


def same_size(t1, t2, pad_idx):
    if t1.size(1) > t2.size(1):
        pads = t2.new_full((t2.size(0), t1.size(1) - t2.size(1)), pad_idx)
        t2 = torch.cat([t2, pads], 1)
    elif t1.size(1) < t2.size(1):
        pads = t1.new_full((t1.size(0), t2.size(1) - t1.size(1)), pad_idx)
        t1 = torch.cat([t1, pads], 1)
    else:  # 如果长度原本就相等，上面两个 if 没走的话，需要 clone 一下，避免后续修改影响到原始张量
        t1 = t1.clone()
        t2 = t2.clone()
    return (t1, t2)


# === 1. 定义新的输出类 (扩充版 EncoderOut) ===
# 它拥有原生 EncoderOut 的所有字段，加上你的新字段
EditRetroEncoderOut = NamedTuple(
    "EditRetroEncoderOut",
    [
        ("encoder_out", Tensor),  # T x B x C
        ("encoder_padding_mask", Tensor),  # B x T
        ("encoder_embedding", Tensor),  # B x T x C
        ("encoder_states", Optional[List[Tensor]]),
        ("src_tokens", Optional[Tensor]),  # B x T
        ("src_lengths", Optional[Tensor]),  # B x 1

        # === 新增字段 (默认为 None) ===
        ("ref_out", Optional[Dict]),  # 存为字典，因为 Decoder 内部需要用 ['key'] 访问
        ("frag_out", Optional[Dict]),  # 存为字典
    ],
)


# ===========================================================================
# 1. 自定义 Encoder: 支持 Shared Encoding 和 Introspector (内省器)
# ===========================================================================
class EditRetroEncoder(FairseqNATEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        # 初始化 Introspector 输入维度是 embed_dim, 输出是 1 (scalar score)
        self.introspector = nn.Sequential(
            nn.Linear(args.encoder_embed_dim, 1),
            nn.Sigmoid()
        )

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        在 Beam Search/Top-K 时，必须手动把 ref_out 和 frag_out 也复制 K 份。
        """
        # 1. 先让父类处理标准数据 (src_tokens 等)
        temp_out = super().reorder_encoder_out(encoder_out, new_order)

        # 2. 手动复制 Ref 数据
        new_ref_out = None
        if encoder_out.ref_out is not None:
            ref_dict = encoder_out.ref_out
            new_ref_out = {}

            # 复制 encoder_out (Seq x Batch x Dim) -> index_select(1, new_order)
            if 'encoder_out' in ref_dict and ref_dict['encoder_out'] is not None:
                new_ref_out['encoder_out'] = ref_dict['encoder_out'].index_select(1, new_order)

            # 复制 padding_mask (Batch x Seq) -> index_select(0, new_order)
            if 'encoder_padding_mask' in ref_dict and ref_dict['encoder_padding_mask'] is not None:
                new_ref_out['encoder_padding_mask'] = ref_dict['encoder_padding_mask'].index_select(0, new_order)

            # 复制 sim_scores (Batch)
            sim = ref_dict['sim_scores']
            if isinstance(sim, list):  # 如果是list先转tensor
                sim = torch.tensor(sim, device=new_order.device)
            if isinstance(sim, torch.Tensor) and sim is not None:
                new_ref_out['sim_scores'] = sim.index_select(0, new_order)

        # 3. 手动复制 Frag 数据
        new_frag_out = None
        if encoder_out.frag_out is not None:
            frag_dict = encoder_out.frag_out
            new_frag_out = {}

            if 'encoder_out' in frag_dict and frag_dict['encoder_out'] is not None:
                new_frag_out['encoder_out'] = frag_dict['encoder_out'].index_select(1, new_order)

            if 'encoder_padding_mask' in frag_dict and frag_dict['encoder_padding_mask'] is not None:
                new_frag_out['encoder_padding_mask'] = frag_dict['encoder_padding_mask'].index_select(0, new_order)

            # 复制 frag_sim (Batch x 1)
            new_frag_out['frag_sim'] = frag_dict['frag_sim'].index_select(0, new_order)

        # 4. 重新打包返回
        return EditRetroEncoderOut(
            encoder_out=temp_out.encoder_out,
            encoder_padding_mask=temp_out.encoder_padding_mask,
            encoder_embedding=temp_out.encoder_embedding,
            encoder_states=temp_out.encoder_states,
            src_tokens=temp_out.src_tokens,
            src_lengths=temp_out.src_lengths,
            # 我们要的新字段
            ref_out=new_ref_out,
            frag_out=new_frag_out,
        )

    def forward(self, src_tokens, src_lengths,
                ref_tokens=None, ref_lengths=None,
                frag_tokens=None, frag_lengths=None,
                sim_scores=None, **kwargs):

        # 1. 编码主 Source
        encoder_out = super().forward(src_tokens, src_lengths, **kwargs)

        # 2. 编码 Reference (共享权重)
        ref_out_dict = None
        if ref_tokens is not None and ref_lengths is not None:
            # 复用父类的 forward
            ref_out = super().forward(ref_tokens, ref_lengths, **kwargs)
            ref_out_dict = {
                'encoder_out': ref_out.encoder_out,
                'encoder_padding_mask': ref_out.encoder_padding_mask,
                'sim_scores': sim_scores  # <--- 存这里，Decoder 才能拿到
            }

        # 3. 编码 Fragment (共享权重) 并计算 Introspector Score
        frag_out_dict = None
        if frag_tokens is not None and frag_lengths is not None:
            frag_out = super().forward(frag_tokens, frag_lengths, **kwargs)
            # 计算 Frag Sim (内省器)
            # frag_out['encoder_out'] shape: [seq_len, batch, embed_dim]
            # 我们需要对其进行 Average Pooling (注意处理 padding)
            x = frag_out.encoder_out.transpose(0, 1)  # [batch, seq, dim]
            x_avg = x.mean(dim=1)

            # 通过 MLP 得到分数 [batch, 1]
            frag_sim = self.introspector(x_avg)
            frag_out_dict = {
                'encoder_out': frag_out.encoder_out,
                'encoder_padding_mask': frag_out.encoder_padding_mask,
                'frag_sim': frag_sim  # <--- 存这里，Decoder 才能拿到
            }

        return EditRetroEncoderOut(
            encoder_out=encoder_out.encoder_out,
            encoder_padding_mask=encoder_out.encoder_padding_mask,
            encoder_embedding=encoder_out.encoder_embedding,
            encoder_states=encoder_out.encoder_states,
            src_tokens=encoder_out.src_tokens,
            src_lengths=encoder_out.src_lengths,
            # 新字段
            ref_out=ref_out_dict,
            frag_out=frag_out_dict,
        )


# ===========================================================================
# 2. 自定义 Decoder Layer: 实现 RetroDKR 的三阶段注意力 (Src, Ref, Frag)
# ===========================================================================
class RetroNATDecoderLayer(TransformerDecoderLayer):
    def __init__(self, args, no_encoder_attn=False):
        super().__init__(args, no_encoder_attn)

        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = nn.Dropout(args.dropout)  # 获取 dropout 模块
        self.activation_dropout_module = nn.Dropout(args.activation_dropout)  # <--- 加上这一行

        # === 新增：Reference Attention ===
        self.ref_attn = MultiheadAttention(
            self.embed_dim,
            args.decoder_attention_heads,
            kdim=args.encoder_embed_dim,
            vdim=args.encoder_embed_dim,
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
        )
        self.ref_layer_norm = LayerNorm(self.embed_dim)

        # === 新增：Fragment Attention ===
        self.frag_attn = MultiheadAttention(
            self.embed_dim,
            args.decoder_attention_heads,
            kdim=args.encoder_embed_dim,
            vdim=args.encoder_embed_dim,
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
        )
        self.frag_layer_norm = LayerNorm(self.embed_dim)

        if getattr(args, "zero_init_fusion", False):
            nn.init.zeros_(self.ref_attn.out_proj.weight)
            nn.init.zeros_(self.ref_attn.out_proj.bias)
            nn.init.zeros_(self.frag_attn.out_proj.weight)
            nn.init.zeros_(self.frag_attn.out_proj.bias)

    def forward(
            self,
            x,
            encoder_out=None,
            encoder_padding_mask=None,
            self_attn_mask=None,
            self_attn_padding_mask=None,
            need_attn=False,
            need_head_weights=False,
            # 新增参数
            ref_out=None,
            frag_out=None,
    ):
        # 1. Self Attention (标准流程)
        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            attn_mask=self_attn_mask,
            need_weights=False,
        )
        x = self.dropout_module(x)
        x = residual + x

        # 2. Reference Attention (如果有 Ref)
        if ref_out is not None:
            residual = x
            x = self.ref_layer_norm(x)

            # ref_out['encoder_out']: [seq, batch, dim]
            # ref_out['encoder_padding_mask']: [batch, seq]
            # ref_out['sim_scores']: [batch] (float list or tensor)

            ref_k = ref_out['encoder_out']
            ref_v = ref_out['encoder_out']
            ref_mask = ref_out['encoder_padding_mask']

            x, attn = self.ref_attn(
                query=x,
                key=ref_k,
                value=ref_v,
                key_padding_mask=ref_mask,
                static_kv=True,
                need_weights=need_attn,
            )

            # === Gating with SIM ===
            sim = ref_out.get('sim_scores', None)
            if sim is not None:
                if isinstance(sim, list):
                    sim = torch.tensor(sim, device=x.device, dtype=x.dtype)
                else:
                    sim = sim.to(dtype=x.dtype)
                # sim: [batch] -> [1, batch, 1] 广播到 [seq, batch, dim]
                sim = sim.view(1, -1, 1)
                x = x * sim

            x = self.dropout_module(x)
            x = residual + x

        # 3. Fragment Attention (如果有 Frag)
        if frag_out is not None:
            residual = x
            x = self.frag_layer_norm(x)

            frag_k = frag_out['encoder_out']
            frag_v = frag_out['encoder_out']
            frag_mask = frag_out['encoder_padding_mask']

            x, attn = self.frag_attn(
                query=x,
                key=frag_k,
                value=frag_v,
                key_padding_mask=frag_mask,
                static_kv=True,
                need_weights=need_attn,
            )

            # === Gating with Frag SIM (Learned) ===
            frag_sim = frag_out.get('frag_sim', None)  # [batch, 1]
            if frag_sim is not None:
                # frag_sim: [batch, 1] -> [1, batch, 1]
                frag_sim = frag_sim.unsqueeze(0)
                x = x * frag_sim

            x = self.dropout_module(x)
            x = residual + x

        # 4. Context Attention (Src Attention) (标准流程)
        if self.encoder_attn is not None:
            residual = x
            x = self.encoder_attn_layer_norm(x)
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                static_kv=True,
                need_weights=need_attn,
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = residual + x

        # 5. Feed Forward (标准流程)
        residual = x
        x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x

        return x, attn, None


@register_model("editretro_nat")
class EditRetroModel(FairseqNATModel):

    @property
    def allow_length_beam(self):
        return False

    @staticmethod
    def add_args(parser):
        FairseqNATModel.add_args(parser)
        parser.add_argument(
            "--early-exit",
            default="6,6,6",  # TODO:
            type=str,
            help="number of decoder layers before word_repos, mask_ins, word_ins",
        )
        parser.add_argument(
            "--no-share-discriminator",
            action="store_true",
            help="separate parameters for discriminator",
        )
        parser.add_argument(
            "--no-share-maskpredictor",
            action="store_true",
            help="separate parameters for mask-predictor",
        )
        parser.add_argument(
            "--share-discriminator-maskpredictor",
            action="store_true",
            help=
            "share the parameters for both mask-predictor and discriminator",
        )
        parser.add_argument(
            "--sampling-for-deletion",
            action='store_true',
            help='instead of argmax, use sampling to predict the tokens')

        parser.add_argument("--dae-ratio",
                            type=float,
                            help='ratio of noisy target as y_ins')

        parser.add_argument("--alpha-ratio",
                            type=float,
                            help='ratio of inserted string as y_reps')

        parser.add_argument(
            "--pretrained-ckpt",
            type=str,
            default=None,
            help="path to load pretrained model weights (for fine-tuning)"
        )
        parser.add_argument(
            "--zero-init-fusion",
            action="store_true",
            default=False,
            help="Zero-initialize the output projection of ref/frag cross-attention layers",
        )

    # === 重写 build_encoder ===
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = EditRetroEncoder(args, src_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = EditRetroDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        decoder.dae_ratio = getattr(args, "dae_ratio", 0.5)
        decoder.alpha_ratio = getattr(args, "alpha_ratio", 0.5)
        return decoder

    # === 重写 build_model 以支持部分参数加载 ===
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # 1. Build the model architecture normally (random initialization)
        model = super().build_model(args, task)

        # 2. Check if we need to load a pretrained checkpoint
        pretrained_path = getattr(args, 'pretrained_ckpt', None)
        if pretrained_path and os.path.exists(pretrained_path):
            from fairseq import checkpoint_utils
            import logging
            logger = logging.getLogger(__name__)

            logger.info(f"Loading pretrained weights from {pretrained_path}...")

            # 加载 Checkpoint 到 CPU
            state = checkpoint_utils.load_checkpoint_to_cpu(pretrained_path)

            # 兼容处理：有的 ckpt 保存的是 {'model': ...}，有的是直接的 state_dict
            if "model" in state:
                pretrained_state_dict = state["model"]
            else:
                pretrained_state_dict = state

            # 获取当前模型的 state_dict
            model_state_dict = model.state_dict()

            # 过滤掉形状不匹配的参数（兼容 ref/frag attention 层）
            filtered_state_dict = {}
            for k, v in pretrained_state_dict.items():
                if k in model_state_dict:
                    if v.shape == model_state_dict[k].shape:
                        filtered_state_dict[k] = v
                    else:
                        logger.warning(f"Skipping {k} due to shape mismatch: {v.shape} vs {model_state_dict[k].shape}")

            # 加载参数，Strict=False 允许新加的层保持随机初始化
            model.load_state_dict(filtered_state_dict, strict=False)

            # 计算未加载的键（新层 + 形状不匹配被跳过的层 + 名字不匹配的层）
            # loaded_keys = set(filtered_state_dict.keys())
            # model_keys = set(model_state_dict.keys())
            # missing_keys = model_keys - loaded_keys
            # if len(missing_keys) > 0:
            #     logger.info(f"Total parameters: {len(model_keys)}, Loaded: {len(loaded_keys)}")
            #     logger.info(f"The following parameters were NOT loaded (randomly initialized):")
            #     for k in sorted(list(missing_keys)):
            #         # 过滤掉一些无关紧要的统计量，只看权重
            #         if "num_batches_tracked" not in k:
            #             logger.info(f"  - {k}")
            #
            #             # 特别检查：如果是 Embeddings 没加载，必须报个大警告
            #             if "embed_tokens" in k or "embed_positions" in k:
            #                 logger.warning(
            #                     f"!!! CRITICAL: Embedding layer {k} was NOT loaded! Check dictionary size match.")
            # else:
            #     logger.info("All parameters loaded successfully.")

        return model

    def forward(self, src_tokens, src_lengths, lp_src_tokens,
                prev_output_tokens, tgt_tokens,
                ref_tokens=None, ref_lengths=None,
                frag_tokens=None, frag_lengths=None,
                sim_scores=None,  # 新增参数
                **kwargs):

        assert tgt_tokens is not None, "forward function only supports training."

        objs = {}

        encoder_out = self.encoder(src_tokens,
                                   src_lengths=src_lengths,
                                   ref_tokens=ref_tokens,  # 新增
                                   ref_lengths=ref_lengths,  # 新增
                                   frag_tokens=frag_tokens,  # 新增
                                   frag_lengths=frag_lengths,  # 新增
                                   sim_scores=sim_scores,  # 新增
                                   **kwargs)

        word_reposition = _get_advanced_reposition_targets(
            lp_src_tokens, tgt_tokens, self.pad)

        y_ins, _, _, _ = _apply_reposition_words(
            lp_src_tokens,
            None,
            None,
            None,
            word_reposition,
            self.pad,
            self.bos,
            self.eos,
        )

        if self.decoder.dae_ratio > 0:
            corrupted = (torch.rand(size=(lp_src_tokens.size(0),),
                                    device=lp_src_tokens.device)
                         < self.decoder.dae_ratio)
            y_ins, prev_output_tokens = same_size(y_ins, prev_output_tokens,
                                                  self.pad)
            y_ins[corrupted] = prev_output_tokens[corrupted]

        y_ins, tgt_tokens = same_size(y_ins, tgt_tokens, self.pad)

        masked_tgt_masks, masked_tgt_tokens, mask_ins_targets = _get_advanced_ins_targets(
            y_ins, tgt_tokens, self.pad, self.unk)
        mask_ins_targets = mask_ins_targets.clamp(min=0, max=255)  # for safe prediction # TODO
        mask_ins_masks = y_ins[:, 1:].ne(self.pad)

        mask_ins_out, _ = self.decoder.forward_mask_ins(
            normalize=False, prev_output_tokens=y_ins, encoder_out=encoder_out)
        word_ins_out, _ = self.decoder.forward_word_ins(
            normalize=False,
            prev_output_tokens=masked_tgt_tokens,
            encoder_out=encoder_out)

        objs['mask_ins'] = {
            "out": mask_ins_out,
            "tgt": mask_ins_targets,
            "mask": mask_ins_masks,
            "ls": 0.01,  # 0.2,
            "ls-type": "uniform",  # "binomial"
            "factor": 1.0,
        }

        objs['word_ins'] = {
            "out": word_ins_out,
            "tgt": tgt_tokens,
            "mask": masked_tgt_masks,
            "ls": self.args.label_smoothing,
            "nll_loss": True,
            "factor": 1.0,  # TODO;
        }

        if self.decoder.sampling_for_deletion:
            word_predictions = torch.multinomial(
                F.softmax(word_ins_out, -1).view(-1, word_ins_out.size(-1)),
                1).view(word_ins_out.size(0), -1)
        else:
            word_predictions = F.log_softmax(word_ins_out, dim=-1).max(-1)[1]

        word_predictions.masked_scatter_(~masked_tgt_masks,
                                         tgt_tokens[~masked_tgt_masks])

        ####### string ready to learn reposition from #######
        y_reps = lp_src_tokens
        if self.decoder.alpha_ratio > 0:
            y_reps, word_predictions = same_size(y_reps, word_predictions,
                                                 self.pad)
            corrupted = (torch.rand(size=(lp_src_tokens.size(0),),
                                    device=lp_src_tokens.device)
                         < self.decoder.alpha_ratio)
            y_reps[corrupted] = word_predictions[corrupted]

        y_reps, tgt_tokens = same_size(y_reps, tgt_tokens, self.pad)

        word_reposition_targets = _get_advanced_reposition_targets(
            y_reps, tgt_tokens, self.pad)
        word_reposition_out, _ = self.decoder.forward_word_reposition(
            normalize=False, prev_output_tokens=y_reps, encoder_out=encoder_out)
        word_reposition_masks = y_reps.ne(self.pad)

        objs['word_reposition'] = {
            "out": word_reposition_out,
            "tgt": word_reposition_targets,
            "mask": word_reposition_masks,
            "ls": 0,
            "factor": 1.0,  # TODO;
        }

        return objs

    def forward_decoder(self,
                        decoder_out,
                        encoder_out,
                        src_tokens,
                        tgt_tokens,
                        hard_constrained_decoding=False,
                        eos_penalty=0.0,
                        del_reward=0.0,
                        max_ratio=None,
                        oracle_repos=False,
                        oracle_mask=False,
                        oracle_token=False,
                        **kwargs):
        output_tokens = decoder_out.output_tokens
        output_marks = decoder_out.output_marks
        output_scores = decoder_out.output_scores
        attn = decoder_out.attn
        total_reposition_ops, total_deletion_ops, total_insertion_ops = decoder_out.num_ops
        history = decoder_out.history
        bsz = output_tokens.size(0)
        if max_ratio is None:
            max_lens = torch.zeros_like(output_tokens).fill_(255)
        else:
            if encoder_out.encoder_padding_mask is None:
                max_src_len = encoder_out.encoder_out.size(0)
                src_lens = encoder_out.encoder_out.new(bsz).fill_(max_src_len)
            else:
                src_lens = (~encoder_out.encoder_padding_mask).sum(1)
            max_lens = (src_lens * max_ratio).clamp(min=10).long()

        # reposition words
        # do not apply if it is <s> </s>
        can_reposition_word = output_tokens.ne(self.pad).sum(1) > 2
        if can_reposition_word.sum() != 0:

            if oracle_repos:
                oracle_word_reposition_pred = _get_advanced_reposition_targets(
                    output_tokens, tgt_tokens, self.pad)
                _tokens, _marks, _scores, _attn = _apply_reposition_words(
                    output_tokens[can_reposition_word],
                    output_marks[can_reposition_word],
                    output_scores[can_reposition_word],
                    None,
                    oracle_word_reposition_pred,
                    self.pad,
                    self.bos,
                    self.eos,
                )

            else:
                word_reposition_score, word_reposition_attn = self.decoder.forward_word_reposition(
                    normalize=True,
                    prev_output_tokens=_skip(output_tokens, can_reposition_word),
                    encoder_out=_skip_encoder_out(self.encoder, encoder_out,
                                                  can_reposition_word))

                if hard_constrained_decoding:
                    no_del_mask = output_marks[can_reposition_word].ne(0)
                    word_del_score = word_reposition_score[:, :, 0]
                    word_del_score.masked_fill_(no_del_mask, -float('Inf'))
                    word_reposition_score = torch.cat([
                        word_del_score.unsqueeze(2), word_reposition_score[:, :,
                                                     1:]
                    ], 2)

                word_reposition_score[:, :, 0] = word_reposition_score[:, :, 0] + del_reward
                word_reposition_pred = word_reposition_score.max(-1)[1]

                num_deletion = word_reposition_pred.eq(
                    0).sum().item() - word_reposition_pred.size(0)
                total_deletion_ops += num_deletion
                total_reposition_ops += word_reposition_pred.ne(
                    torch.arange(word_reposition_pred.size(1),
                                 device=word_reposition_pred.device).unsqueeze(
                        0)).sum().item() - num_deletion

                _tokens, _marks, _scores, _attn = _apply_reposition_words(
                    output_tokens[can_reposition_word],
                    output_marks[can_reposition_word],
                    output_scores[can_reposition_word],
                    word_reposition_attn,
                    word_reposition_pred,
                    self.pad,
                    self.bos,
                    self.eos,
                )

            output_tokens = _fill(output_tokens, can_reposition_word, _tokens,
                                  self.pad)
            output_marks = _fill(output_marks, can_reposition_word, _marks, 0)
            output_scores = _fill(output_scores, can_reposition_word, _scores,
                                  0)
            attn = _fill(attn, can_reposition_word, _attn, 0.)
            if history is not None:
                history.append(output_tokens.clone())

        # insert placeholders
        can_ins_mask = output_tokens.ne(self.pad).sum(1) < max_lens
        if can_ins_mask.sum() != 0:

            if oracle_mask:
                _, _, mask_ins_targets = _get_advanced_ins_targets(
                    output_tokens, tgt_tokens, self.pad, self.unk)
                mask_ins_targets = mask_ins_targets.clamp(min=0, max=255)  # for safe prediction
                _tokens, _marks, _scores = _apply_ins_masks(
                    output_tokens[can_ins_mask],
                    output_marks[can_ins_mask],
                    output_scores[can_ins_mask],
                    mask_ins_targets,
                    self.pad,
                    self.unk,
                    self.eos,
                )

            else:
                mask_ins_score, _ = self.decoder.forward_mask_ins(
                    normalize=True,
                    prev_output_tokens=_skip(output_tokens, can_ins_mask),
                    encoder_out=_skip_encoder_out(self.encoder, encoder_out,
                                                  can_ins_mask))
                if eos_penalty > 0:
                    mask_ins_score[:, :, 0] = mask_ins_score[:, :, 0] - eos_penalty
                mask_ins_pred = mask_ins_score.max(-1)[1]
                mask_ins_pred = torch.min(
                    mask_ins_pred, max_lens[can_ins_mask,
                    None].expand_as(mask_ins_pred))

                if hard_constrained_decoding:
                    no_ins_mask = output_marks[can_ins_mask][:, :-1].eq(1)
                    mask_ins_pred.masked_fill_(no_ins_mask, 0)

                total_insertion_ops += mask_ins_pred.sum().item()

                _tokens, _marks, _scores = _apply_ins_masks(
                    output_tokens[can_ins_mask],
                    output_marks[can_ins_mask],
                    output_scores[can_ins_mask],
                    mask_ins_pred,
                    self.pad,
                    self.unk,
                    self.eos,
                )

            output_tokens = _fill(output_tokens, can_ins_mask, _tokens,
                                  self.pad)
            output_marks = _fill(output_marks, can_ins_mask, _marks, 0)
            output_scores = _fill(output_scores, can_ins_mask, _scores, 0)
            if history is not None:
                history.append(output_tokens.clone())

        # insert words
        can_ins_word = output_tokens.eq(self.unk).sum(1) > 0
        if can_ins_word.sum() != 0:

            if oracle_token:
                _, masked_tgt_tokens, _ = _get_advanced_ins_targets(output_tokens, tgt_tokens, self.pad, self.unk)
                _tokens, _scores = _apply_ins_words(
                    output_tokens[can_ins_word],
                    output_scores[can_ins_word],
                    masked_tgt_tokens,
                    None,
                    self.unk,
                )

            else:
                word_ins_score, word_ins_attn = self.decoder.forward_word_ins(
                    normalize=True,  # TODO
                    prev_output_tokens=_skip(output_tokens, can_ins_word),
                    encoder_out=_skip_encoder_out(self.encoder, encoder_out,
                                                  can_ins_word))

                # TODO: run argmax greedy decoding 
                word_ins_score, word_ins_pred = word_ins_score.max(-1)

                _tokens, _scores = _apply_ins_words(
                    output_tokens[can_ins_word],
                    output_scores[can_ins_word],
                    word_ins_pred,
                    word_ins_score,
                    self.unk,
                )

            output_tokens = _fill(output_tokens, can_ins_word, _tokens,
                                  self.pad)
            output_scores = _fill(output_scores, can_ins_word, _scores, 0)
            attn = _fill(attn, can_ins_word, word_ins_attn, 0.)

            if history is not None:
                history.append(output_tokens.clone())

        # delete some unnecessary paddings
        cut_off = output_tokens.ne(self.pad).sum(1).max()
        output_tokens = output_tokens[:, :cut_off]
        output_marks = output_marks[:, :cut_off]
        output_scores = output_scores[:, :cut_off]
        attn = None if attn is None else attn[:, :cut_off, :]

        return decoder_out._replace(output_tokens=output_tokens,
                                    output_marks=output_marks,
                                    output_scores=output_scores,
                                    attn=attn,
                                    num_ops=(total_reposition_ops,
                                             total_deletion_ops,
                                             total_insertion_ops),
                                    history=history)

    def forward_decoder_reposition(self,
                                   decoder_out,
                                   encoder_out,
                                   tgt_tokens,
                                   del_reward=0.0,
                                   oracle_repos=False,
                                   repos_beam=5,
                                   insert_beam=1,
                                   **kwargs):

        output_tokens = decoder_out.output_tokens
        output_marks = decoder_out.output_marks
        output_scores = decoder_out.output_scores
        attn = decoder_out.attn
        total_reposition_ops, total_deletion_ops, total_insertion_ops = decoder_out.num_ops
        history = decoder_out.history
        bsz = output_tokens.size(0)

        can_reposition_word = output_tokens.ne(self.pad).sum(1) > 2
        if can_reposition_word.sum() != 0:
            word_reposition_score, word_reposition_attn = self.decoder.forward_word_reposition(
                normalize=True,  ### TODO
                prev_output_tokens=_skip(output_tokens, can_reposition_word),
                encoder_out=_skip_encoder_out(self.encoder, encoder_out,
                                              can_reposition_word))

            word_reposition_score[:, :, 0] = word_reposition_score[:, :, 0] + del_reward

            if repos_beam > 1:
                bz, SEQLEN = word_reposition_score.size(0), word_reposition_score.size(1)

                try:
                    word_reposition_pred = word_reposition_score.topk(
                        repos_beam, dim=-1)[1].permute(0, 2, 1)
                    N = repos_beam
                except:
                    word_reposition_pred = word_reposition_score.topk(
                        word_reposition_score.size(-1),
                        dim=-1)[1].permute(0, 2, 1)
                    N = word_reposition_score.size(-1)

                rank1 = word_reposition_pred[:, 0, :].repeat_interleave(repos_beam, dim=0)
                rank2 = word_reposition_pred[:, 1, :].repeat_interleave(repos_beam, dim=0)

                ranks = []
                for i in range(word_reposition_pred.size(1)):
                    a = word_reposition_pred[:, i, :]
                    ranks.append(a)

                ranki = torch.zeros((bz * repos_beam, SEQLEN))
                for k in range(repos_beam):
                    for i in range(bz):
                        if k == repos_beam - 1:
                            ranki[i * repos_beam + k, :] = ranks[1][i, :]
                        else:
                            ranki[i * repos_beam + k, :] = ranks[min(max(1, int(repos_beam * k / N)),
                                                                     len(ranks) - 1)][i, :]

                c = (torch.rand(size=(bz * repos_beam, SEQLEN),
                                device=word_reposition_pred.device) < 0.2)
                rank2[c] = ranki.long().cuda(rank1.device)[c]

                corrupted = torch.rand(size=(bz, repos_beam, SEQLEN),
                                       device=word_reposition_pred.device)
                MIX = torch.arange(0, repos_beam,
                                   device=word_reposition_pred.device) / (repos_beam - 1)
                corrupted = corrupted < MIX[..., None]
                corrupted = corrupted.reshape(bz * repos_beam, SEQLEN)
                rank1[corrupted] = rank2[corrupted]
                word_reposition_pred = rank1

                output_tokens = output_tokens.repeat_interleave(repos_beam, dim=0)
                output_marks = output_marks.repeat_interleave(repos_beam, dim=0)
                output_scores = output_scores.repeat_interleave(repos_beam, dim=0)
                can_reposition_word = output_tokens.ne(self.pad).sum(1) > 2

            else:
                word_reposition_pred = word_reposition_score.max(-1)[1]

            num_deletion = word_reposition_pred.eq(0).sum().item() - word_reposition_pred.size(0)
            total_deletion_ops += num_deletion
            total_reposition_ops += word_reposition_pred.ne(
                torch.arange(word_reposition_pred.size(1),
                             device=word_reposition_pred.device).unsqueeze(
                    0)).sum().item() - num_deletion

            if oracle_repos:
                oracle_word_reposition_pred = _get_advanced_reposition_targets(
                    output_tokens, tgt_tokens, self.pad)
                _tokens, _marks, _scores, _attn = _apply_reposition_words(
                    output_tokens[can_reposition_word],
                    output_marks[can_reposition_word],
                    output_scores[can_reposition_word],
                    None,
                    oracle_word_reposition_pred,
                    self.pad,
                    self.bos,
                    self.eos,
                )
            else:
                _tokens, _marks, _scores, _attn = _apply_reposition_words(
                    output_tokens[can_reposition_word],
                    output_marks[can_reposition_word],
                    output_scores[can_reposition_word],
                    word_reposition_attn,
                    word_reposition_pred,
                    self.pad,
                    self.bos,
                    self.eos,
                )
            output_tokens = _fill(output_tokens, can_reposition_word, _tokens, self.pad)
            output_marks = _fill(output_marks, can_reposition_word, _marks, 0)
            output_scores = _fill(output_scores, can_reposition_word, _scores, 0)
            attn = _fill(attn, can_reposition_word, _attn, 0.)

            if history is not None:
                if insert_beam > 1:
                    history.append(output_tokens.clone().repeat_interleave(insert_beam, dim=0))
                else:
                    history.append(output_tokens.clone())

        # delete some unnecessary paddings
        cut_off = output_tokens.ne(self.pad).sum(1).max()
        output_tokens = output_tokens[:, :cut_off]
        output_marks = output_marks[:, :cut_off]
        output_scores = output_scores[:, :cut_off]
        attn = None if attn is None else attn[:, :cut_off, :]

        return decoder_out._replace(output_tokens=output_tokens,
                                    output_marks=output_marks,
                                    output_scores=output_scores,
                                    attn=attn,
                                    num_ops=(total_reposition_ops,
                                             total_deletion_ops,
                                             total_insertion_ops),
                                    history=history)

    def forward_decoder_mask(self,
                             decoder_out,
                             encoder_out,
                             tgt_tokens,
                             eos_penalty=0.0,
                             max_ratio=None,
                             oracle_mask=False,
                             mask_beam=1,
                             token_beam=1,
                             mask_mode='topk',  # 'mixed', 'topk'
                             **kwargs):

        output_tokens = decoder_out.output_tokens
        output_marks = decoder_out.output_marks
        output_scores = decoder_out.output_scores
        attn = decoder_out.attn
        total_reposition_ops, total_deletion_ops, total_insertion_ops = decoder_out.num_ops
        history = decoder_out.history
        bsz = output_tokens.size(0)
        if max_ratio is None:
            max_lens = torch.zeros_like(output_tokens).fill_(255)
        else:
            if encoder_out.encoder_padding_mask is None:
                max_src_len = encoder_out.encoder_out.size(0)
                src_lens = encoder_out.encoder_out.new(bsz).fill_(max_src_len)
            else:
                src_lens = (~encoder_out.encoder_padding_mask).sum(1)
            max_lens = (src_lens * max_ratio).clamp(min=10).long()

        # insert mask placeholder
        can_ins_mask = output_tokens.ne(self.pad).sum(1) < max_lens
        if can_ins_mask.sum() != 0:
            mask_ins_score, _ = self.decoder.forward_mask_ins(
                normalize=True,
                prev_output_tokens=_skip(output_tokens, can_ins_mask),
                encoder_out=_skip_encoder_out(self.encoder, encoder_out, can_ins_mask))

            if eos_penalty > 0:
                mask_ins_score[:, :, 0] = mask_ins_score[:, :, 0] - eos_penalty

            if mask_beam == 1:
                mask_ins_pred = mask_ins_score.max(-1)[1]
                mask_ins_pred = torch.min(mask_ins_pred, max_lens[can_ins_mask, None].expand_as(mask_ins_pred))

            # TODO: length beam
            elif mask_mode == 'topk':
                SEQLEN = mask_ins_score.size(1)
                mask_ins_pred = mask_ins_score.topk(mask_beam, dim=-1)[1]
                mask_ins_pred = mask_ins_pred.permute(0, 2, 1).reshape(-1, SEQLEN)
                can_ins_mask = can_ins_mask.repeat_interleave(mask_beam, dim=0)
                max_lens = max_lens.repeat_interleave(mask_beam, dim=0)
                mask_ins_pred = torch.min(mask_ins_pred, max_lens[can_ins_mask, None].expand_as(mask_ins_pred))

                output_tokens = output_tokens.repeat_interleave(mask_beam, dim=0)
                output_marks = output_marks.repeat_interleave(mask_beam, dim=0)
                output_scores = output_scores.repeat_interleave(mask_beam, dim=0)
                if attn != None:
                    attn = attn.repeat_interleave(mask_beam, dim=0)

            elif mask_mode == 'mixed':

                bz, SEQLEN = mask_ins_score.size(0), mask_ins_score.size(1)
                mask_ins_pred = mask_ins_score.topk(mask_beam, dim=-1)[1].permute(0, 2, 1)

                rank1 = mask_ins_pred[:, 0, :].repeat_interleave(mask_beam, dim=0)
                rank2 = mask_ins_pred[:, 1, :].repeat_interleave(mask_beam, dim=0)

                c = (torch.rand(size=(bz * mask_beam, SEQLEN), device=mask_ins_pred.device) < 0.2)
                mask_ins_pred_reshape = mask_ins_pred.reshape(-1, SEQLEN)
                rank2[c] = mask_ins_pred_reshape.long().cuda(rank1.device)[c]

                corrupted = torch.rand(size=(bz, mask_beam, SEQLEN), device=mask_ins_pred.device)
                MIX = torch.arange(0, mask_beam, device=mask_ins_pred.device) / (mask_beam - 1)
                corrupted = corrupted < MIX[..., None]
                corrupted = corrupted.reshape(bz * mask_beam, SEQLEN)
                rank1[corrupted] = rank2[corrupted]
                mask_ins_pred = rank1

                can_ins_mask = can_ins_mask.repeat_interleave(mask_beam, dim=0)
                max_lens = max_lens.repeat_interleave(mask_beam, dim=0)
                mask_ins_pred = torch.min(mask_ins_pred, max_lens[can_ins_mask, None].expand_as(mask_ins_pred))

                output_tokens = output_tokens.repeat_interleave(mask_beam, dim=0)
                output_marks = output_marks.repeat_interleave(mask_beam, dim=0)
                output_scores = output_scores.repeat_interleave(mask_beam, dim=0)
                if attn != None:
                    attn = attn.repeat_interleave(mask_beam, dim=0)

            else:
                raise NotImplementedError

            total_insertion_ops += mask_ins_pred.sum().item()

            _tokens, _marks, _scores = _apply_ins_masks(
                output_tokens[can_ins_mask],
                output_marks[can_ins_mask],
                output_scores[can_ins_mask],
                mask_ins_pred,
                self.pad,
                self.unk,
                self.eos,
            )

            output_tokens = _fill(output_tokens, can_ins_mask, _tokens, self.pad)
            output_marks = _fill(output_marks, can_ins_mask, _marks, 0)
            output_scores = _fill(output_scores, can_ins_mask, _scores, 0)

            if history is not None:
                if token_beam > 1:
                    history.append(output_tokens.clone().repeat_interleave(token_beam, dim=0))
                else:
                    history.append(output_tokens.clone())

        # delete some unnecessary paddings
        cut_off = output_tokens.ne(self.pad).sum(1).max()
        output_tokens = output_tokens[:, :cut_off]
        output_marks = output_marks[:, :cut_off]
        output_scores = output_scores[:, :cut_off]
        attn = None if attn is None else attn[:, :cut_off, :]

        return decoder_out._replace(output_tokens=output_tokens,
                                    output_marks=output_marks,
                                    output_scores=output_scores,
                                    attn=attn,
                                    num_ops=(total_reposition_ops,
                                             total_deletion_ops,
                                             total_insertion_ops),
                                    history=history)

    def forward_decoder_token(self,
                              decoder_out,
                              encoder_out,
                              tgt_tokens,
                              eos_penalty=0.0,
                              max_ratio=None,
                              oracle_token=False,
                              token_beam=1,
                              token_mode='topk',  # 'mixed', 'topk' 
                              **kwargs):

        output_tokens = decoder_out.output_tokens
        output_marks = decoder_out.output_marks
        output_scores = decoder_out.output_scores
        attn = decoder_out.attn
        total_reposition_ops, total_deletion_ops, total_insertion_ops = decoder_out.num_ops
        history = decoder_out.history

        # insert words
        can_ins_word = output_tokens.eq(self.unk).sum(1) > 0
        if can_ins_word.sum() != 0:
            word_ins_score, word_ins_attn = self.decoder.forward_word_ins(
                normalize=True,
                prev_output_tokens=_skip(output_tokens, can_ins_word),
                encoder_out=_skip_encoder_out(self.encoder, encoder_out,
                                              can_ins_word))

            if token_beam == 1:
                # TODO: argmax greedy decoding
                word_ins_score, word_ins_pred = word_ins_score.max(-1)

            elif token_mode == 'topk':
                # TODO: top-k decoding
                SEQLEN = word_ins_score.size(1)
                word_ins_score, word_ins_pred = word_ins_score.topk(token_beam, dim=-1)

                word_ins_score = word_ins_score.permute(0, 2, 1).reshape(-1, SEQLEN)
                word_ins_pred = word_ins_pred.permute(0, 2, 1).reshape(-1, SEQLEN)

                output_tokens = output_tokens.repeat_interleave(token_beam, dim=0)
                output_marks = output_marks.repeat_interleave(token_beam, dim=0)
                output_scores = output_scores.repeat_interleave(token_beam, dim=0)
                if attn != None:
                    attn = attn.repeat_interleave(token_beam, dim=0)

                can_ins_word = output_tokens.eq(self.unk).sum(1) > 0

            elif token_mode == 'mixed':

                bz, SEQLEN = word_ins_score.size(0), word_ins_score.size(1)
                word_ins_score, word_ins_pred = word_ins_score.topk(token_beam, dim=-1)

                word_ins_score = word_ins_score.permute(0, 2, 1)
                word_ins_pred = word_ins_pred.permute(0, 2, 1)

                rank1 = word_ins_pred[:, 0, :].repeat_interleave(token_beam, dim=0)
                rank2 = word_ins_pred[:, 1, :].repeat_interleave(token_beam, dim=0)

                score1 = word_ins_score[:, 0, :].repeat_interleave(token_beam, dim=0)
                score2 = word_ins_score[:, 1, :].repeat_interleave(token_beam, dim=0)

                word_ins_score_reshape = word_ins_score.reshape(-1, SEQLEN)
                word_ins_pred_reshape = word_ins_pred.reshape(-1, SEQLEN)
                c = (torch.rand(size=(bz * token_beam, SEQLEN), device=word_ins_pred.device) < 0.2)
                rank2[c] = word_ins_pred_reshape.long().cuda(rank1.device)[c]
                score2[c] = word_ins_score_reshape.float().cuda(score1.device)[c]

                corrupted = torch.rand(size=(bz, token_beam, SEQLEN), device=word_ins_pred.device)
                MIX = torch.arange(0, token_beam, device=word_ins_pred.device) / (token_beam - 1)
                corrupted = corrupted < MIX[..., None]
                corrupted = corrupted.reshape(bz * token_beam, SEQLEN)
                rank1[corrupted] = rank2[corrupted]
                word_ins_pred = rank1
                score1[corrupted] = score2[corrupted]
                word_ins_score = score1

                output_tokens = output_tokens.repeat_interleave(token_beam, dim=0)
                output_marks = output_marks.repeat_interleave(token_beam, dim=0)
                output_scores = output_scores.repeat_interleave(token_beam, dim=0)
                if attn != None:
                    attn = attn.repeat_interleave(token_beam, dim=0)

                can_ins_word = output_tokens.eq(self.unk).sum(1) > 0

            else:
                raise NotImplementedError

            _tokens, _scores = _apply_ins_words(
                output_tokens[can_ins_word],
                output_scores[can_ins_word],
                word_ins_pred,
                word_ins_score,
                self.unk,
            )

            output_tokens = _fill(output_tokens, can_ins_word, _tokens, self.pad)
            output_scores = _fill(output_scores, can_ins_word, _scores, 0)
            attn = _fill(attn, can_ins_word, word_ins_attn, 0.)

            if history is not None:
                history.append(output_tokens.clone())

        # delete some unnecessary paddings
        cut_off = output_tokens.ne(self.pad).sum(1).max()
        output_tokens = output_tokens[:, :cut_off]
        output_marks = output_marks[:, :cut_off]
        output_scores = output_scores[:, :cut_off]
        attn = None if attn is None else attn[:, :cut_off, :]

        return decoder_out._replace(output_tokens=output_tokens,
                                    output_marks=output_marks,
                                    output_scores=output_scores,
                                    attn=attn,
                                    num_ops=(total_reposition_ops,
                                             total_deletion_ops,
                                             total_insertion_ops),
                                    history=history)

    def initialize_output_tokens(self,
                                 encoder_out,
                                 src_tokens,
                                 init_tokens=None):
        if init_tokens is not None:
            initial_output_tokens = init_tokens
        else:
            initial_output_tokens = src_tokens.new_zeros(src_tokens.size(0), 2)
            initial_output_tokens[:, 0] = self.bos
            initial_output_tokens[:, 1] = self.eos

        initial_output_marks = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()).type_as(encoder_out.encoder_out)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()).type_as(encoder_out.encoder_out)

        return DecoderOut(output_tokens=initial_output_tokens,
                          output_marks=initial_output_marks,
                          output_scores=initial_output_scores,
                          attn=None,
                          step=0,
                          max_step=0,
                          num_ops=(0, 0, 0),
                          history=None)


class EditRetroDecoder(FairseqNATDecoder):

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args,
                         dictionary,
                         embed_tokens,
                         no_encoder_attn=no_encoder_attn)
        # 1. 替换标准 Layer 为自定义 RetroNATDecoderLayer

        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()
        self.sampling_for_deletion = getattr(args, "sampling_for_deletion", False)
        self.embed_mask_ins = Embedding(256, self.output_embed_dim * 2, None)  # TODO

        # del_word, ins_mask, ins_word
        self.early_exit = [int(i) for i in args.early_exit.split(',')]
        assert len(self.early_exit) == 3

        # copy layers for mask-predict/deletion
        self.layers_msk = None
        if getattr(args, "no_share_maskpredictor", False):
            self.layers_msk = nn.ModuleList(
                [RetroNATDecoderLayer(args, no_encoder_attn) for _ in range(self.early_exit[1])])

        self.layers_reposition = None
        if getattr(args, "no_share_discriminator", False):
            self.layers_reposition = nn.ModuleList(
                [RetroNATDecoderLayer(args, no_encoder_attn) for _ in range(self.early_exit[0])])

        if getattr(args, "share_discriminator_maskpredictor", False):
            assert getattr(args, "no_share_discriminator", False), "must set saperate discriminator"
            self.layers_msk = self.layers_reposition

        # === 替换 layers ===
        self.layers = nn.ModuleList([
            RetroNATDecoderLayer(args, no_encoder_attn)
            for _ in range(args.decoder_layers)
        ])

    def extract_features(self,
                         prev_output_tokens,
                         encoder_out=None,
                         early_exit=None,
                         layers=None,
                         **unused):
        """
        Similar to *forward* but only return features.
        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the EDITORTransformer decoder has full-attention to all generated tokens
        """

        # embed positions
        positions = (self.embed_positions(prev_output_tokens)
                     if self.embed_positions is not None else None)

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder layers
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        layers = self.layers if layers is None else layers
        early_exit = len(layers) if early_exit is None else early_exit

        # === 从 encoder_out 提取 ref/frag 信息 ===
        ref_out = getattr(encoder_out, "ref_out", None) if encoder_out else None
        frag_out = getattr(encoder_out, "frag_out", None) if encoder_out else None

        for _, layer in enumerate(layers[:early_exit]):
            x, attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
                ref_out=ref_out,
                frag_out=frag_out
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": attn, "inner_states": inner_states}

    @ensemble_decoder
    def forward_mask_ins(self, normalize, encoder_out, prev_output_tokens,
                         **unused):
        features, extra = self.extract_features(prev_output_tokens,
                                                encoder_out=encoder_out,
                                                early_exit=self.early_exit[1],
                                                layers=self.layers_msk,
                                                **unused)
        features_cat = torch.cat([features[:, :-1, :], features[:, 1:, :]], 2)
        decoder_out = F.linear(features_cat, self.embed_mask_ins.weight)
        if normalize:
            return F.log_softmax(decoder_out, -1), extra['attn']
        return decoder_out, extra['attn']

    @ensemble_decoder
    def forward_word_ins(self, normalize, encoder_out, prev_output_tokens,
                         **unused):
        features, extra = self.extract_features(prev_output_tokens,
                                                encoder_out=encoder_out,
                                                early_exit=self.early_exit[2],
                                                layers=self.layers,
                                                **unused)
        decoder_out = self.output_layer(features)
        if normalize:
            return F.log_softmax(decoder_out, -1), extra['attn']
        return decoder_out, extra['attn']

    @ensemble_decoder
    def forward_word_reposition(self, normalize, encoder_out,
                                prev_output_tokens, **unused):
        features, extra = self.extract_features(prev_output_tokens,
                                                encoder_out=encoder_out,
                                                early_exit=self.early_exit[0],
                                                layers=self.layers_reposition,
                                                **unused)
        prev_output_embed = self.embed_tokens(prev_output_tokens)

        # B x T x T
        decoder_out = torch.bmm(features, prev_output_embed.transpose(1, 2))

        if normalize:
            return F.log_softmax(decoder_out, -1), extra['attn']
        return decoder_out, extra['attn']


@register_model_architecture("editretro_nat", "editretro_nat")  # TODO: architecture name
def EditRetro_base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before",
                                            False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim",
                                     args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim",
                                         args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False)
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False)
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(args, "decoder_output_dim",
                                      args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, "decoder_input_dim",
                                     args.decoder_embed_dim)
    args.early_exit = getattr(args, "early_exit", "6,6,6")
    args.no_share_discriminator = getattr(args, "no_share_discriminator", False)
    args.no_share_maskpredictor = getattr(args, "no_share_maskpredictor", False)
    args.share_discriminator_maskpredictor = getattr(
        args, "share_discriminator_maskpredictor", False)
    args.no_share_last_layer = getattr(args, "no_share_last_layer", False)
    args.zero_init_fusion = getattr(args, "zero_init_fusion", False)


@register_model_architecture("editretro_nat", "editretro")  # TODO: architecture name
def EditRetro_full_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim",
                                     args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim",
                                         args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False)
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False)
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(args, "decoder_output_dim",
                                      args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, "decoder_input_dim",
                                     args.decoder_embed_dim)
    args.early_exit = getattr(args, "early_exit", "6,6,6")
    args.no_share_discriminator = getattr(args, "no_share_discriminator", False)
    args.no_share_maskpredictor = getattr(args, "no_share_maskpredictor", False)
    args.share_discriminator_maskpredictor = getattr(
        args, "share_discriminator_maskpredictor", False)
    args.no_share_last_layer = getattr(args, "no_share_last_layer", False)
    args.zero_init_fusion = getattr(args, "zero_init_fusion", False)

@register_model_architecture("editretro_nat", "editretro_nat_50k")  # TODO: architecture name
def EditRetro_small_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False)
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False)
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)
    args.decoder_output_dim = getattr(args, "decoder_output_dim", args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)

    args.early_exit = getattr(args, "early_exit", "6,6,6")
    args.no_share_discriminator = getattr(args, "no_share_discriminator", False)
    args.no_share_maskpredictor = getattr(args, "no_share_maskpredictor", False)
    args.share_discriminator_maskpredictor = getattr(
        args, "share_discriminator_maskpredictor", False)
    args.no_share_last_layer = getattr(args, "no_share_last_layer", False)
    args.zero_init_fusion = getattr(args, "zero_init_fusion", False)