# -*- coding: utf-8 -*-
from functools import partial

import torch
from torchtext.data import Field, RawField

from onmt.constants import DefaultTokens
from onmt.inputters.datareader_base import DataReaderBase


class TextDataReader(DataReaderBase):
    def read(self, sequences, side, features={}):
        if isinstance(sequences, str):
            sequences = DataReaderBase._read_file(sequences)

        features_names = []
        features_values = []
        for feat_name, v in features.items():
            features_names.append(feat_name)
            if isinstance(v, str):
                features_values.append(DataReaderBase._read_file(features))
            else:
                features_values.append(v)  
        for i, (seq, *feats) in enumerate(zip(sequences, *features_values)):
            ex_dict = {}
            if isinstance(seq, bytes):
                seq = seq.decode("utf-8")
            ex_dict[side] = seq
            for i, f in enumerate(feats):
                if isinstance(f, bytes):
                    f = f.decode("utf-8")
                ex_dict[features_names[i]] = f
            yield {side: ex_dict, "indices": i}


def text_sort_key(ex):
    """Sort using the number of tokens in the sequence."""
    if hasattr(ex, "tgt"):
        return len(ex.src[0]), len(ex.tgt[0])
    return len(ex.src[0])


# Legacy function. Currently it only truncates input if truncate is set.
# mix this with partial
def _feature_tokenize(
        string, layer=0, tok_delim=None, feat_delim=None, truncate=None):

    tokens = string.split(tok_delim)
    if truncate is not None:
        tokens = tokens[:truncate]
    if feat_delim is not None:
        tokens = [t.split(feat_delim)[layer] for t in tokens]
    return tokens


class TextMultiField(RawField):

    def __init__(self, base_name, base_field, feats_fields):
        super(TextMultiField, self).__init__()
        self.fields = [(base_name, base_field)]
        for name, ff in sorted(feats_fields, key=lambda kv: kv[0]):
            self.fields.append((name, ff))

    @property
    def base_field(self):
        return self.fields[0][1]

    def process(self, batch, device=None):
        # batch (list(list(list))): batch_size x len(self.fields) x seq_len
        batch_by_feat = list(zip(*batch))
        base_data = self.base_field.process(batch_by_feat[0], device=device)
        if self.base_field.include_lengths:
            # lengths: batch_size
            base_data, lengths = base_data

        feats = [ff.process(batch_by_feat[i], device=device)
                 for i, (_, ff) in enumerate(self.fields[1:], 1)]
        levels = [base_data] + feats
        # data: seq_len x batch_size x len(self.fields)
        data = torch.stack(levels, 2)
        if self.base_field.include_lengths:
            return data, lengths
        else:
            return data

    def preprocess(self, x):
        return [f.preprocess(x[fn]) for fn, f in self.fields]

    def __getitem__(self, item):
        return self.fields[item]


def text_fields(**kwargs):

    feats = kwargs["feats"]
    include_lengths = kwargs["include_lengths"]
    base_name = kwargs["base_name"]
    pad = kwargs.get("pad", DefaultTokens.PAD)
    bos = kwargs.get("bos", DefaultTokens.BOS)
    eos = kwargs.get("eos", DefaultTokens.EOS)
    truncate = kwargs.get("truncate", None)
    fields_ = []

    feat_delim = None #u"￨" if n_feats > 0 else None

    # Base field
    tokenize = partial(
        _feature_tokenize,
        layer=None,
        truncate=truncate,
        feat_delim=feat_delim)
    feat = Field(
        init_token=bos, eos_token=eos,
        pad_token=pad, tokenize=tokenize,
        include_lengths=include_lengths)
    fields_.append((base_name, feat))

    # Feats fields
    if feats:
        for feat_name in feats.keys():
            # Legacy function, it is not really necessary
            tokenize = partial(
                _feature_tokenize,
                layer=None,
                truncate=truncate,
                feat_delim=feat_delim)
            feat = Field(
                init_token=bos, eos_token=eos,
                pad_token=pad, tokenize=tokenize,
                include_lengths=False)
            fields_.append((feat_name, feat))

    assert fields_[0][0] == base_name  # sanity check
    field = TextMultiField(fields_[0][0], fields_[0][1], fields_[1:])
    return field
