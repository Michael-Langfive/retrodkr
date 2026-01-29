# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import itertools
import torch

from fairseq.data import (
    MultiSourceTranslationDataset,
    AppendTokenDataset,
    noising,
    ConcatDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    indexed_dataset
)
from fairseq.utils import new_arange
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask  # , load_langpair_dataset
from fairseq import utils


from editretro.data.language_pair_dataset import LanguagePairDataset, logger
from editretro.data.masked_pair_dataset import RawFloatDataset

import random


@register_task('translation_retro')
class EditRetroTask(TranslationTask):
    """
    Translation (Sequence Generation) task for Levenshtein Transformer
    See `"Levenshtein Transformer" <https://arxiv.org/abs/1905.11006>`_.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)

        parser.add_argument(
            '--noise',
            default='random_delete',
            choices=['random_delete', 'random_delete_shuffle', 'random_mask', 'no_noise', 'full_mask'])

        parser.add_argument('--random-seed', default=1, type=int)

        parser.add_argument(
            "--init-src", action="store_true",
            help="initialize with src during inference",
        )

        parser.add_argument(
            "--oracle-repos", action="store_true",
            help="use oracle reposition during inference",
        )
        parser.add_argument(
            "--oracle-mask", action="store_true",
            help="use oracle mask during inference",
        )
        parser.add_argument(
            "--oracle-token", action="store_true",
            help="use oracle token during inference",
        )
        parser.add_argument('--TOPK', default=10, type=int)
        parser.add_argument('--repos-beam', default=5, type=int)
        parser.add_argument('--token-beam', default=2, type=int)
        parser.add_argument('--mask-beam', default=1, type=int)
        parser.add_argument('--inference-with-augmentation', default=False, action="store_true")
        parser.add_argument('--aug', default=20, type=int)
        parser.add_argument('-r', '--ref-lang', default=None, metavar='REF',
                            help='reference language ')
        parser.add_argument('-f', '--frag-lang', default=None, metavar='FRAG',
                            help='fragment language')
        parser.add_argument('--sim-lang', default=None, metavar='SIM', help='similarity score')
        parser.add_argument('--lambda1', default=1.0, type=float, help='Weight for Null Context (Src only)')
        parser.add_argument('--lambda2', default=0.0, type=float, help='Weight for Reference Context (Src + Ref)')
        parser.add_argument('--lambda3', default=0.0, type=float, help='Weight for Fragment Context (Src + Frag)')

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang
        ref, frag, sim = self.args.ref_lang, self.args.frag_lang, self.args.sim_lang
        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            ref, frag, sim,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            prepend_bos=True,
        )

    def inject_noise(self, target_tokens):
        def _random_delete(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()

            max_len = target_tokens.size(1)
            target_mask = target_tokens.eq(pad)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(
                target_tokens.eq(bos) | target_tokens.eq(eos), 0.0)
            target_score.masked_fill_(target_mask, 1)
            target_score, target_rank = target_score.sort(1)
            target_length = target_mask.size(1) - target_mask.float().sum(
                1, keepdim=True)

            # do not delete <bos> and <eos> (we assign 0 score for them)
            target_cutoff = 2 + ((target_length - 2) * target_score.new_zeros(
                target_score.size(0), 1).uniform_()).long()
            target_cutoff = target_score.sort(1)[1] >= target_cutoff

            prev_target_tokens = target_tokens.gather(
                1, target_rank).masked_fill_(target_cutoff, pad).gather(
                1,
                target_rank.masked_fill_(target_cutoff,
                                         max_len).sort(1)[1])
            prev_target_tokens = prev_target_tokens[:, :prev_target_tokens.ne(pad).sum(1).max()]

            return prev_target_tokens

        def _random_shuffle(target_tokens, p, max_shuffle_distance):
            word_shuffle = noising.WordShuffle(self.tgt_dict)
            target_mask = target_tokens.eq(self.tgt_dict.pad())
            target_length = target_mask.size(1) - target_mask.long().sum(1)
            prev_target_tokens, _ = word_shuffle.noising(
                target_tokens.t().cpu(), target_length.cpu(), max_shuffle_distance)
            prev_target_tokens = prev_target_tokens.to(target_tokens.device).t()
            masks = (target_tokens.clone().sum(dim=1, keepdim=True).float()
                     .uniform_(0, 1) < p)
            prev_target_tokens = masks * prev_target_tokens + (~masks) * target_tokens
            return prev_target_tokens

        def _random_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_masks = target_tokens.ne(pad) & \
                           target_tokens.ne(bos) & \
                           target_tokens.ne(eos)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()
            target_length = target_length * target_length.clone().uniform_()
            target_length = target_length + 1  # make sure to mask at least one token.

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(target_rank) < target_length[:, None].long()
            prev_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), unk)
            return prev_target_tokens

        def _full_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_mask = target_tokens.eq(bos) | target_tokens.eq(
                eos) | target_tokens.eq(pad)
            return target_tokens.masked_fill(~target_mask, unk)

        if self.args.noise == 'random_delete_shuffle':
            return _random_shuffle(_random_delete(target_tokens), 0.5, 3)
        elif self.args.noise == 'random_delete':
            return _random_delete(target_tokens)
        elif self.args.noise == 'random_mask':
            return _random_mask(target_tokens)
        elif self.args.noise == 'full_mask':
            return _full_mask(target_tokens)
        elif self.args.noise == 'no_noise':
            return target_tokens
        else:
            raise NotImplementedError

    @property
    def topk(self):
        return self.args.TOPK

    def build_generator(self, args):
        from editretro.models.iterative_refinement_generator import IterativeRefinementGenerator
        return IterativeRefinementGenerator(
            self.target_dictionary,
            eos_penalty=getattr(args, 'iter_decode_eos_penalty', 0.0),
            del_reward=getattr(args, 'iter_decode_deletion_reward', 0.0),
            max_iter=getattr(args, 'iter_decode_max_iter', 10),
            beam_size=getattr(args, 'iter_decode_with_beam', 1),
            reranking=getattr(args, 'iter_decode_with_external_reranker', False),
            decoding_format=getattr(args, 'decoding_format', None),
            adaptive=not getattr(args, 'iter_decode_force_max_iter', False),
            retain_history=getattr(args, 'retain_iter_history', False),
            constrained_decoding=getattr(args, 'constrained_decoding', False),
            hard_constrained_decoding=getattr(args, 'hard_constrained_decoding', False),
            random_seed=getattr(args, 'random_seed', 1),
            init_src=getattr(args, 'init_src', True),
            oracle_repos=getattr(args, 'oracle_repos', False),
            oracle_mask=getattr(args, 'oracle_mask', False),
            oracle_token=getattr(args, 'oracle_token', False),
            TOPK=getattr(args, 'TOPK', 10),
            repos_beam=getattr(args, 'repos_beam', 5),
            token_beam=getattr(args, 'token_beam', 2),
            mask_beam=getattr(args, 'mask_beam', 1),
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, tgt_tokens=None, tgt_lengths=None,
                                    num_source_inputs=1):
        if num_source_inputs == 1:
            return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary, tgt=tgt_tokens,
                                       tgt_sizes=tgt_lengths, append_bos=True)
        else:
            return MultiSourceTranslationDataset(src_tokens, src_lengths, self.source_dictionary, tgt=tgt_tokens,
                                                 tgt_sizes=tgt_lengths, append_bos=True)

    def train_step(self,
                   sample,
                   model,
                   criterion,
                   optimizer,
                   update_num,
                   ignore_grad=False):
        model.train()

        sample['prev_target'] = self.inject_noise(sample['target'])
        w1 = getattr(self.args, 'lambda1', 1.0)
        w2 = getattr(self.args, 'lambda2', 0.0)
        w3 = getattr(self.args, 'lambda3', 0.0)
        full_net_input = sample['net_input'].copy()

        # 定义一个辅助函数来处理日志累加，避免代码重复
        def accumulate_logging_output(agg_log, current_log, weight):
            if current_log is None:
                return
            # 累加 Loss (加权)
            agg_log['loss'] += current_log.get('loss', 0) * weight
            agg_log['nll_loss'] += current_log.get('nll_loss', 0) * weight

        # =======================================================
        # Branch 1: Null Context (Src Only)
        # =======================================================
        sample['net_input']['ref_tokens'] = None
        sample['net_input']['ref_lengths'] = None
        sample['net_input']['frag_tokens'] = None
        sample['net_input']['frag_lengths'] = None
        sample['net_input']['sim_scores'] = None

        loss_null, sample_size, logging_output = criterion(model, sample)
        total_loss = w1 * loss_null
        agg_logging_output = logging_output
        agg_logging_output['loss'] *= w1
        agg_logging_output['nll_loss'] *= w1

        # =======================================================
        # Branch 2: Ref Context (Src + Ref)
        # =======================================================
        if w2 > 0 and full_net_input.get('ref_tokens') is not None:
            # 恢复 Ref, 保持 Frag 为 None
            sample['net_input']['ref_tokens'] = full_net_input['ref_tokens']
            sample['net_input']['ref_lengths'] = full_net_input['ref_lengths']
            sample['net_input']['sim_scores'] = full_net_input.get('sim_scores')
            sample['net_input']['frag_tokens'] = None
            sample['net_input']['frag_lengths'] = None

            loss_ref, _, logging_output = criterion(model, sample)
            total_loss += w2 * loss_ref
            accumulate_logging_output(agg_logging_output, logging_output, w2)
        # =======================================================
        # Branch 3: Frag Context (Src + Frag)
        # =======================================================
        if w3 > 0 and full_net_input.get('frag_tokens') is not None:
            # 屏蔽 Ref
            sample['net_input']['ref_tokens'] = None
            sample['net_input']['ref_lengths'] = None
            sample['net_input']['sim_scores'] = None
            # 恢复 Frag
            sample['net_input']['frag_tokens'] = full_net_input['frag_tokens']
            sample['net_input']['frag_lengths'] = full_net_input['frag_lengths']

            loss_frag, _, logging_output = criterion(model, sample)
            total_loss += w3 * loss_frag
            accumulate_logging_output(agg_logging_output, logging_output, w3)

        sample['net_input'] = full_net_input
        if ignore_grad:
            total_loss *= 0
        optimizer.backward(total_loss)
        return total_loss, sample_size, agg_logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            sample['prev_target'] = self.inject_noise(sample['target'])
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output


def load_langpair_dataset(
        data_path,
        split,
        src, src_dict,
        tgt, tgt_dict,
        ref, frag, sim,
        combine, dataset_impl, upsample_primary,
        left_pad_source, left_pad_target, max_source_positions,
        max_target_positions, prepend_bos=False, load_alignments=False,
        truncate_source=False, append_source_id=False
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(
            data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []
    ref_datasets = []
    frag_datasets = []
    sim_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(
                data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(
                data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        def only_source_load(ext, dictionary):
            if ext is None: return None
            path_only_source = os.path.join(data_path, "{}.{}-None.{}".format(split_k, ext, ext))
            if indexed_dataset.dataset_exists(path_only_source, impl=dataset_impl):
                return data_utils.load_indexed_dataset(path_only_source, dictionary, dataset_impl)
            return None

        if ref is not None:
            ref_dataset = only_source_load(ref, src_dict)
            if ref_dataset is not None:
                ref_datasets.append(ref_dataset)

        if frag is not None:
            frag_dataset = only_source_load(frag, src_dict)
            if frag_dataset is not None:
                frag_datasets.append(frag_dataset)

        if sim is not None:
            path_simple = os.path.join(data_path, "{}.{}".format(split_k, sim))
            if os.path.exists(path_simple):
                sim_datasets.append(RawFloatDataset(path_simple))

        log_info = '{} {} {}-{} {} examples'.format(
            data_path, split_k, src, tgt, len(src_datasets[-1])
        )

        if len(ref_datasets) > 0:
            log_info += ' | ref: {} examples'.format(len(ref_datasets[-1]))


        if len(frag_datasets) > 0:
            log_info += ' | frag: {} examples'.format(len(frag_datasets[-1]))

        if len(sim_datasets) > 0:
            log_info += ' | sim: loaded'

        logger.info(log_info)
        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    def make_concat(datasets):
        if len(datasets) == 0: return None
        if len(datasets) == 1: return datasets[0]
        return ConcatDataset(datasets, sample_ratios=[1] * len(datasets))

    src_dataset = make_concat(src_datasets)
    tgt_dataset = make_concat(tgt_datasets)
    ref_dataset = make_concat(ref_datasets)
    frag_dataset = make_concat(frag_datasets)
    sim_dataset = make_concat(sim_datasets)

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(
            tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
        if ref_dataset is not None:
            ref_dataset = PrependTokenDataset(ref_dataset, src_dict.bos())
        if frag_dataset is not None:
            frag_dataset = PrependTokenDataset(frag_dataset, src_dict.bos())

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(
            data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    ref_dataset_sizes = ref_dataset.sizes if ref_dataset is not None else None
    frag_dataset_sizes = frag_dataset.sizes if frag_dataset is not None else None

    return LanguagePairDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset_sizes, tgt_dict,
        ref=ref_dataset,
        ref_sizes=ref_dataset_sizes,
        frag=frag_dataset,
        frag_sizes=frag_dataset_sizes,
        sim=sim_dataset,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
        align_dataset=align_dataset,
        eos=eos,
        input_feeding=True
    )
