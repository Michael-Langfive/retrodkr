""" Onmt NMT Model base class definition """
import logging
import os
from datetime import datetime

import torch
import torch.nn as nn
from onmt.utils.misc import sequence_mask

logger = logging.getLogger()


class BaseModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder / decoder or decoder only model.
    """

    def __init__(self, encoder, decoder):
        super(BaseModel, self).__init__()

    def forward(self, src, tgt, lengths, bptt=False, with_align=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(tgt_len, batch, features)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        raise NotImplementedError

    def update_dropout(self, dropout):
        raise NotImplementedError

    def count_parameters(self, log=print):
        raise NotImplementedError


class NMTModel(BaseModel):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.
    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, bptt=False, with_align=False):
        dec_in = tgt[:-1]  # exclude last target from inputs

        enc_state, memory_bank, lengths = self.encoder(src, lengths)

        if not bptt:
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(dec_in, memory_bank,
                                      memory_lengths=lengths,
                                      with_align=with_align)
        return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)

    def count_parameters(self, log=print):
        """Count number of parameters in model (& print with `log` callback).

        Returns:
            (int, int):
            * encoder side parameter count
            * decoder side parameter count
        """

        enc, dec = 0, 0
        for name, param in self.named_parameters():
            if 'encoder' in name:
                enc += param.nelement()
            else:
                dec += param.nelement()
        if callable(log):
            log('encoder: {}'.format(enc))
            log('decoder: {}'.format(dec))
            log('* number of parameters: {}'.format(enc + dec))
        return enc, dec


class LanguageModel(BaseModel):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic decoder only model.
    Currently TransformerLMDecoder is the only LM decoder implemented
    Args:
      decoder (onmt.decoders.TransformerLMDecoder): a transformer decoder
    """

    def __init__(self, encoder=None, decoder=None):
        super(LanguageModel, self).__init__(encoder, decoder)
        if encoder is not None:
            raise ValueError("LanguageModel should not be used"
                             "with an encoder")
        self.decoder = decoder

    def forward(self, src, tgt, lengths, bptt=False, with_align=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.
        Args:
            src (Tensor): A source sequence passed to decoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on decoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(tgt_len, batch, features)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.
        Returns:
            (FloatTensor, dict[str, FloatTensor]):
            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """

        if not bptt:
            self.decoder.init_state()
        dec_out, attns = self.decoder(
            src, memory_bank=None, memory_lengths=lengths,
            with_align=with_align
        )
        return dec_out, attns

    def update_dropout(self, dropout):
        self.decoder.update_dropout(dropout)

    def count_parameters(self, log=print):
        """Count number of parameters in model (& print with `log` callback).
        Returns:
            (int, int):
            * encoder side parameter count
            * decoder side parameter count
        """

        enc, dec = 0, 0
        for name, param in self.named_parameters():
            if "decoder" in name:
                dec += param.nelement()

        if callable(log):
            # No encoder in LM, seq2seq count formatting kept
            log("encoder: {}".format(enc))
            log("decoder: {}".format(dec))
            log("* number of parameters: {}".format(enc + dec))
        return enc, dec


class RetroDKR(NMTModel):
    """
    扩展NMTModel以支持参考序列的处理和相似度引导的融合
    Args:
        encoder (onmt.encoders.EncoderBase): 编码器
        decoder (onmt.decoders.RetroDKRDecoder): RetroDKR解码器
    """

    def __init__(self, encoder, decoder):
        super(RetroDKR, self).__init__(encoder, decoder)
        # 获取编码器输出的维度
        hidden_size = encoder.embeddings.embedding_size
        # 初始化内省器
        self.introspector = nn.Sequential(  # [batch, hidden]
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        self.setup_logger()
        self.batch_count = 0  # 改为记录batch数而不是步数

    def setup_logger(self):
        """设置日志记录器"""
        os.makedirs('introspector_logs', exist_ok=True)
        self.log_file = f'introspector_logs/frag_scores_{self.decoder.lambda2}_{self.decoder.lambda3}.txt'
        # 训练开始时清空旧文件
        with open(self.log_file, "w") as f:
            f.write("")  # 或 f.truncate(0)

    def log_fragment_scores(self, scores, batch_size):
        """记录每个样本的分数
        Args:
            scores: [batch_size, 1] 张量
            batch_size: 当前batch的大小
        """
        # append 模式 "a"
        with open(self.log_file, "a") as f:
            for i in range(batch_size):
                self.batch_count += 1
                f.write(f"{self.batch_count}\t{scores[i].item():.6f}\n")

    def forward(self, src, tgt, lengths, ref=None, ref_lengths=None, sim=None,
                frag=None, frag_lengths=None, bptt=False, with_align=False):
        dec_in = tgt[:-1]
        enc_state, memory_bank, lengths = self.encoder(src, lengths)
        if self.decoder.lambda2 > 0 and ref is not None:
            _, ref_memory, ref_lengths = self.encoder(ref, ref_lengths)
        else:
            ref_memory, ref_lengths, sim = None, None, None
        if self.decoder.lambda3 > 0 and frag is not None:
            frag_emb, frag_memory, frag_lengths = self.encoder(frag, frag_lengths)
            if hasattr(self.encoder, 'transformer'):
                frag_avg = frag_emb.transpose(0, 1).mean(dim=1).contiguous()  # [batch, hidden]
                frag_sim = self.introspector(frag_avg).unsqueeze(1)
            else:
                frag_avg = frag_memory.transpose(0, 1).mean(dim=1).contiguous()
                frag_sim = self.introspector(frag_avg).unsqueeze(1)
        else:
            frag_memory, frag_lengths, frag_sim = None, None, None
        if not bptt:
            self.decoder.init_state(src, memory_bank, enc_state)

        dec_out, attns = self.decoder(
            dec_in,  # [BOS, w1, w2, w3]
            memory_bank,
            memory_lengths=lengths,
            ref_memory_bank=ref_memory,
            ref_memory_lengths=ref_lengths,
            sim=sim,
            frag_memory_bank=frag_memory,
            frag_memory_lengths=frag_lengths,
            frag_sim=frag_sim,
            with_align=with_align
        )
        return dec_out, attns
