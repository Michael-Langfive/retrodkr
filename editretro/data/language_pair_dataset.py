# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import numpy as np
import torch
from fairseq.data import FairseqDataset, data_utils

logger = logging.getLogger(__name__)


def collate(
        samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
        input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])

    # 1. Source
    src_tokens = merge('source', left_pad=left_pad_source)
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)
    decoder_input = merge('source', left_pad=left_pad_target)
    decoder_input = decoder_input.index_select(0, sort_order)

    # 2. Target
    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    # 3. Ref (自定义)
    ref_tokens = None
    ref_lengths = None
    if samples[0].get('ref', None) is not None:
        ref_tokens = merge('ref', left_pad=left_pad_source)
        ref_tokens = ref_tokens.index_select(0, sort_order)
        ref_lengths = torch.LongTensor([s['ref'].numel() for s in samples]).index_select(0, sort_order)

    # 4. Frag (自定义)
    frag_tokens = None
    frag_lengths = None
    if samples[0].get('frag', None) is not None:
        frag_tokens = merge('frag', left_pad=left_pad_source)
        frag_tokens = frag_tokens.index_select(0, sort_order)
        frag_lengths = torch.LongTensor([s['frag'].numel() for s in samples]).index_select(0, sort_order)

    # 5. Sim (自定义)
    sim_scores = None
    if samples[0].get('sim', None) is not None:
        sim_scores = torch.tensor([s['sim'] for s in samples], dtype=torch.float)
        sim_scores = sim_scores.index_select(0, sort_order)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
        'decoder_input': decoder_input,
    }

    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens

    # 将自定义数据注入 net_input
    if ref_tokens is not None:
        batch['net_input']['ref_tokens'] = ref_tokens
        batch['net_input']['ref_lengths'] = ref_lengths

    if frag_tokens is not None:
        batch['net_input']['frag_tokens'] = frag_tokens
        batch['net_input']['frag_lengths'] = frag_lengths

    if sim_scores is not None:
        batch['net_input']['sim_scores'] = sim_scores

    return batch


class LanguagePairDataset(FairseqDataset):
    """
    一个干净的 Dataset 类，专门用于 EditRetro 任务。
    去除了 Masked Source 逻辑，原生支持 Ref/Frag/Sim。
    """

    def __init__(
            self, src, src_sizes, src_dict,
            tgt=None, tgt_sizes=None, tgt_dict=None,
            # === Retro Args (No Masked Src) ===
            ref=None, ref_sizes=None,
            frag=None, frag_sizes=None,
            sim=None,
            # === Standard Args ===
            left_pad_source=True, left_pad_target=False,
            max_source_positions=1024, max_target_positions=1024,
            shuffle=True, input_feeding=True,
            remove_eos_from_source=False, append_eos_to_target=False,
            align_dataset=None,
            append_bos=False, eos=None
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()

        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

        # === Custom Data ===
        self.ref = ref
        self.frag = frag
        self.sim = sim
        self.ref_sizes = np.array(ref_sizes) if ref_sizes is not None else None
        self.frag_sizes = np.array(frag_sizes) if frag_sizes is not None else None

        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        self.append_bos = append_bos
        self.eos = (eos if eos is not None else src_dict.eos())

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]

        # === Custom Items ===
        ref_item = self.ref[index] if self.ref is not None else None
        frag_item = self.frag[index] if self.frag is not None else None
        sim_item = self.sim[index] if self.sim is not None else None

        # === BOS / EOS Processing ===
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.src_dict.bos()
            if src_item[0] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), src_item])

            # 同时给 Ref/Frag 加 BOS
            if ref_item is not None and ref_item[0] != bos:
                ref_item = torch.cat([torch.LongTensor([bos]), ref_item])
            if frag_item is not None and frag_item[0] != bos:
                frag_item = torch.cat([torch.LongTensor([bos]), frag_item])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if src_item[-1] == eos:
                src_item = src_item[:-1]
            if ref_item is not None and ref_item[-1] == eos:
                ref_item = ref_item[:-1]
            if frag_item is not None and frag_item[-1] == eos:
                frag_item = frag_item[:-1]

        example = {
            'id': index,
            'source': src_item,
            'target': tgt_item,
            'ref': ref_item,
            'frag': frag_item,
            'sim': sim_item
        }

        if self.align_dataset is not None:
            example['alignment'] = self.align_dataset[index]

        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.eos,
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

    def num_tokens(self, index):
        return max(
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
            self.ref_sizes[index] if self.ref_sizes is not None else 0,
            self.frag_sizes[index] if self.frag_sizes is not None else 0,
        )

    def size(self, index):
        return max(
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
            self.ref_sizes[index] if self.ref_sizes is not None else 0,
            self.frag_sizes[index] if self.frag_sizes is not None else 0,
        )

    def ordered_indices(self):
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return getattr(self.src, 'supports_prefetch', False)

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.ref is not None:
            self.ref.prefetch(indices)
        if self.frag is not None:
            self.frag.prefetch(indices)