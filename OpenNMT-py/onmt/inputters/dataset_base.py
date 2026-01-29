# coding: utf-8

from itertools import chain, starmap
from collections import Counter

import torch
from torchtext.data import Dataset as TorchtextDataset
from torchtext.data import Example
from torchtext.vocab import Vocab


def _join_dicts(*args):
    return dict(chain(*[d.items() for d in args]))


def _dynamic_dict(example, src_field, tgt_field):
    """处理动态词典，增加对参考序列的支持"""
    src = src_field.tokenize(example["src"]["src"])
    # make a small vocab containing just the tokens in the source sequence
    unk = src_field.unk_token
    pad = src_field.pad_token

    # add init_token and eos_token according to src construction
    if src_field.init_token:
        src = [src_field.init_token] + src
    if src_field.eos_token:
        src.append(src_field.eos_token)

    src_ex_vocab = Vocab(Counter(src), specials=[unk, pad])
    unk_idx = src_ex_vocab.stoi[unk]
    # Map source tokens to indices in the dynamic dict.
    src_map = torch.LongTensor([src_ex_vocab.stoi[w] for w in src])
    example["src_map"] = src_map
    example["src_ex_vocab"] = src_ex_vocab

    if "tgt" in example:
        tgt = tgt_field.tokenize(example["tgt"]["tgt"])
        mask = torch.LongTensor(
            [unk_idx] + [src_ex_vocab.stoi[w] for w in tgt] + [unk_idx])
        example["alignment"] = mask

    # 参考序列映射
    if "ref" in example:
        ref = src_field.tokenize(example["ref"]["ref"])
        if src_field.init_token:
            ref = [src_field.init_token] + ref
        if src_field.eos_token:
            ref.append(src_field.eos_token)
        ref_map = torch.LongTensor([src_ex_vocab.stoi.get(w, unk_idx) for w in ref])
        example["ref_map"] = ref_map
    # 片段映射
    if "frag" in example:
        frag = src_field.tokenize(example["frag"]["frag"])
        if src_field.init_token:
            frag = [src_field.init_token] + frag
        if src_field.eos_token:
            frag.append(src_field.eos_token)
        frag_map = torch.LongTensor([src_ex_vocab.stoi.get(w, unk_idx) for w in frag])
        example["frag_map"] = frag_map
    # 相似度值不需要词表映射，直接保留
    if "sim" in example:
        example["sim"] = {"sim": float(example["sim"]["sim"])}

    return example


class Dataset(TorchtextDataset):
    """支持参考序列的数据集类"""
    def __init__(self, fields, readers, data, sort_key, filter_pred=None):
        self.sort_key = sort_key
        can_copy = 'src_map' in fields and 'alignment' in fields

        read_iters = [r.read(dat, name, feats) for r, (name, dat, feats) in zip(readers, data)]

        # self.src_vocabs is used in collapse_copy_scores and Translator.py
        self.src_vocabs = []
        examples = []
        for ex_dict in starmap(_join_dicts, zip(*read_iters)):
            if can_copy:
                src_field = fields['src']
                tgt_field = fields['tgt']
                # this assumes src_field and tgt_field are both text
                ex_dict = _dynamic_dict(
                    ex_dict, src_field.base_field, tgt_field.base_field)
                self.src_vocabs.append(ex_dict["src_ex_vocab"])
            ex_fields = {k: [(k, v)] for k, v in fields.items() if
                         k in ex_dict}
            ex = Example.fromdict(ex_dict, ex_fields)
            examples.append(ex)

        # fields needs to have only keys that examples have as attrs
        fields = []
        for _, nf_list in ex_fields.items():
            assert len(nf_list) == 1
            fields.append(nf_list[0])

        super(Dataset, self).__init__(examples, fields, filter_pred)

    def __getattr__(self, attr):
        # avoid infinite recursion when fields isn't defined
        if 'fields' not in vars(self):
            raise AttributeError
        if attr in self.fields:
            return (getattr(x, attr) for x in self.examples)
        else:
            raise AttributeError

    def save(self, path, remove_fields=True):
        if remove_fields:
            self.fields = []
        torch.save(self, path)

    @staticmethod
    def config(fields):
        """配置数据集

        Args:
            fields: 包含源序列、目标序列和参考序列的字段配置
        """
        readers, data = [], []
        for name, field in fields:
            if field["data"] is not None:
                readers.append(field["reader"])
                data.append((name, field["data"], field["features"]))
        return readers, data
