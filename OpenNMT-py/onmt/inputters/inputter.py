# -*- coding: utf-8 -*-
import os
import codecs
import math
import warnings

from collections import Counter, defaultdict, OrderedDict

import torch
from torchtext.data import Field, RawField, LabelField
from torchtext.vocab import Vocab

from onmt.constants import DefaultTokens, ModelTask
from onmt.inputters.text_dataset import text_fields
from onmt.utils.logging import logger
# backwards compatibility
from onmt.inputters.text_dataset import _feature_tokenize  # noqa: F401

import gc


# monkey-patch to make torchtext Vocab's pickleable
def _getstate(self):
    return dict(self.__dict__, stoi=dict(self.stoi))


def _setstate(self, state):
    self.__dict__.update(state)
    self.stoi = defaultdict(lambda: 0, self.stoi)


Vocab.__getstate__ = _getstate
Vocab.__setstate__ = _setstate

warnings.filterwarnings("ignore", message="To copy construct from a tensor")


def make_src(data, vocab):
    src_size = max([t.size(0) for t in data])
    src_vocab_size = max([t.max() for t in data]) + 1
    alignment = torch.zeros(src_size, len(data), src_vocab_size)
    for i, sent in enumerate(data):
        for j, t in enumerate(sent):
            alignment[j, i, t] = 1
    return alignment


def make_tgt(data, vocab):
    tgt_size = max([t.size(0) for t in data])
    alignment = torch.zeros(tgt_size, len(data)).long()
    for i, sent in enumerate(data):
        alignment[:sent.size(0), i] = sent
    return alignment


class AlignField(LabelField):
    """
    Parse ['<src>-<tgt>', ...] into ['<src>','<tgt>', ...]
    """

    def __init__(self, **kwargs):
        kwargs['use_vocab'] = False
        kwargs['preprocessing'] = parse_align_idx
        super(AlignField, self).__init__(**kwargs)

    def process(self, batch, device=None):
        """ Turn a batch of align-idx to a sparse align idx Tensor"""
        sparse_idx = []
        for i, example in enumerate(batch):
            for src, tgt in example:
                # +1 for tgt side to keep coherent after "bos" padding,
                # register ['N°_in_batch', 'tgt_id+1', 'src_id']
                sparse_idx.append([i, tgt + 1, src])
        _build_field_vocab
        align_idx = torch.tensor(sparse_idx, dtype=self.dtype, device=device)

        return align_idx


def parse_align_idx(align_pharaoh):
    """
    Parse Pharaoh alignment into [[<src>, <tgt>], ...]
    """
    align_list = align_pharaoh.strip().split(' ')
    flatten_align_idx = []
    for align in align_list:
        try:
            src_idx, tgt_idx = align.split('-')
        except ValueError:
            logger.warning("{} in `{}`".format(align, align_pharaoh))
            logger.warning("Bad alignement line exists. Please check file!")
            raise
        flatten_align_idx.append([int(src_idx), int(tgt_idx)])
    return flatten_align_idx


def get_task_spec_tokens(data_task, pad, bos, eos):
    """
    Retrieve pad/bos/eos tokens for each data task
    """
    if data_task == ModelTask.SEQ2SEQ:
        return {
            "src": {"pad": pad, "bos": None, "eos": None},
            "tgt": {"pad": pad, "bos": bos, "eos": eos},
        }
    elif data_task == ModelTask.LANGUAGE_MODEL:
        return {
            "src": {"pad": pad, "bos": bos, "eos": None},
            "tgt": {"pad": pad, "bos": None, "eos": eos},
        }
    elif data_task == ModelTask.RetroDKR:  # 添加 RetroDKR 支持
        return {
            "src": {"pad": pad, "bos": None, "eos": None},
            "tgt": {"pad": pad, "bos": bos, "eos": eos},
            "ref": {"pad": pad, "bos": None, "eos": None},
            "frag": {"pad": pad, "bos": None, "eos": None},
            "sim": None
        }
    else:
        raise ValueError(f"No task specific tokens defined for {data_task}")


# 定义一个可序列化的函数
def process_sim_values(x, _):
    """处理相似度值的函数
    Args:
        x: 包含sim值的字典列表
        _: 未使用的参数
    Returns:
        torch.Tensor: 处理后的相似度张量
    """
    try:
        sim_values = [float(item["sim"].strip()) if isinstance(item["sim"], str)
                      else float(item["sim"]) for item in x]
        tensor = torch.tensor(sim_values, dtype=torch.float)
        return tensor.clone().detach()  # 使用clone().detach()创建新的张量
    except (ValueError, KeyError) as e:
        logger.error(f"Error processing similarity values: {e}")
        return torch.zeros(len(x), dtype=torch.float).clone().detach()


def get_fields(
        src_data_type,
        src_feats,
        tgt_feats,
        pad=DefaultTokens.PAD,
        bos=DefaultTokens.BOS,
        eos=DefaultTokens.EOS,
        dynamic_dict=False,
        with_align=False,
        src_truncate=None,
        tgt_truncate=None,
        ref_truncate=None,
        frag_truncate=None,
        data_task=ModelTask.SEQ2SEQ
):
    assert src_data_type in ['text'], \
        "Data type not implemented"
    assert not dynamic_dict or src_data_type == 'text', \
        'it is not possible to use dynamic_dict with non-text input'
    fields = {}

    fields_getters = {"text": text_fields}
    task_spec_tokens = get_task_spec_tokens(data_task, pad, bos, eos)

    src_field_kwargs = {
        "feats": src_feats,
        "include_lengths": True,
        "pad": task_spec_tokens["src"]["pad"],
        "bos": task_spec_tokens["src"]["bos"],
        "eos": task_spec_tokens["src"]["eos"],
        "truncate": src_truncate,
        "base_name": "src",
    }
    fields["src"] = fields_getters[src_data_type](**src_field_kwargs)

    tgt_field_kwargs = {
        "feats": tgt_feats,
        "include_lengths": False,  # 没传长度
        "pad": task_spec_tokens["tgt"]["pad"],
        "bos": task_spec_tokens["tgt"]["bos"],
        "eos": task_spec_tokens["tgt"]["eos"],
        "truncate": tgt_truncate,
        "base_name": "tgt",
    }
    fields["tgt"] = fields_getters["text"](**tgt_field_kwargs)
    if data_task == ModelTask.RetroDKR:
        ref_field_kwargs = {
            "feats": None,
            "include_lengths": True,
            "pad": task_spec_tokens["ref"]["pad"],
            "bos": task_spec_tokens["ref"]["bos"],
            "eos": task_spec_tokens["ref"]["eos"],
            "truncate": ref_truncate,
            "base_name": "ref",
        }
        fields["ref"] = text_fields(**ref_field_kwargs)
        # Add sim field
        fields["sim"] = Field(
            use_vocab=False,
            dtype=torch.float,
            sequential=False,
            postprocessing=process_sim_values,  # 替换 lambda 函数
            batch_first=True,
        )
        frag_field_kwargs = {
            "feats": None,
            "include_lengths": True,
            "pad": task_spec_tokens["frag"]["pad"],
            "bos": task_spec_tokens["frag"]["bos"],
            "eos": task_spec_tokens["frag"]["eos"],
            "truncate": frag_truncate,
            "base_name": "frag",
        }
        fields["frag"] = text_fields(**frag_field_kwargs)
    indices = Field(use_vocab=False, dtype=torch.long, sequential=False)
    fields["indices"] = indices

    if dynamic_dict:
        src_map = Field(
            use_vocab=False, dtype=torch.float,
            postprocessing=make_src, sequential=False)
        fields["src_map"] = src_map

        src_ex_vocab = RawField()
        fields["src_ex_vocab"] = src_ex_vocab

        align = Field(
            use_vocab=False, dtype=torch.long,
            postprocessing=make_tgt, sequential=False)
        fields["alignment"] = align

    if with_align:
        word_align = AlignField()
        fields["align"] = word_align

    return fields


class IterOnDevice(object):
    """Sent items from `iterable` on `device_id` and yield."""

    def __init__(self, iterable, device_id):
        self.iterable = iterable
        self.device_id = device_id

    @staticmethod
    def batch_to_device(batch, device_id):
        """Move `batch` to `device_id`, cpu if `device_id` < 0."""
        curr_device = batch.indices.device
        device = torch.device(device_id) if device_id >= 0 \
            else torch.device('cpu')
        if curr_device != device:
            # 主要字段的处理
            for field in ['src', 'tgt', 'ref', 'frag']:
                if hasattr(batch, field):
                    field_value = getattr(batch, field)
                    if isinstance(field_value, tuple):
                        setattr(batch, field,
                                tuple([_.to(device) for _ in field_value]))
                    else:
                        setattr(batch, field, field_value.to(device))
            # 相似度值处理
            if hasattr(batch, 'sim'):
                batch.sim = batch.sim.to(device)

            batch.indices = batch.indices.to(device)
            batch.alignment = batch.alignment.to(device) \
                if hasattr(batch, 'alignment') else None
            batch.src_map = batch.src_map.to(device) \
                if hasattr(batch, 'src_map') else None
            batch.align = batch.align.to(device) \
                if hasattr(batch, 'align') else None

    def __iter__(self):
        for batch in self.iterable:
            self.batch_to_device(batch, self.device_id)
            yield batch


def filter_example(ex, use_src_len=True, use_tgt_len=True,
                   use_ref_len=True, use_frag_len=True,
                   min_src_len=1, max_src_len=float('inf'),
                   min_tgt_len=1, max_tgt_len=float('inf'),
                   min_ref_len=1, max_ref_len=float('inf'),
                   min_frag_len=1, max_frag_len=float('inf')):  # 新增参数:
    src_len = len(ex.src[0])
    tgt_len = len(ex.tgt[0])
    ref_len = len(ex.ref[0]) if hasattr(ex, 'ref') else 0
    frag_len = len(ex.frag[0]) if hasattr(ex, 'frag') else 0
    return (not use_src_len or min_src_len <= src_len <= max_src_len) and \
        (not use_tgt_len or min_tgt_len <= tgt_len <= max_tgt_len) and \
        (not use_ref_len or min_ref_len <= ref_len <= max_ref_len) and \
        (not use_frag_len or min_frag_len <= frag_len <= max_frag_len)


def _pad_vocab_to_multiple(vocab, multiple):
    vocab_size = len(vocab)
    if vocab_size % multiple == 0:
        return
    target_size = int(math.ceil(vocab_size / multiple)) * multiple
    padding_tokens = ["{}{}".format(DefaultTokens.VOCAB_PAD, i)
                      for i in range(target_size - vocab_size)]
    vocab.extend(Vocab(Counter(), specials=padding_tokens))
    return vocab


def _build_field_vocab(field, counter, size_multiple=1, **kwargs):
    # this is basically copy-pasted from torchtext.
    all_special = [
        field.unk_token, field.pad_token, field.init_token, field.eos_token
    ]
    specials = kwargs.pop('specials', [])
    all_special.extend(list(specials or []))
    specials = list(OrderedDict.fromkeys(
        tok for tok in all_special if tok is not None))
    field.vocab = field.vocab_cls(counter, specials=specials, **kwargs)
    if size_multiple > 1:
        _pad_vocab_to_multiple(field.vocab, size_multiple)


def _load_vocab(vocab_path, name, counters, min_freq=0):
    # counters changes in place
    vocab, has_count = _read_vocab_file(vocab_path, name)
    vocab_size = len(vocab)
    logger.info('Loaded %s vocab has %d tokens.' % (name, vocab_size))
    if not has_count:
        for i, token in enumerate(vocab):
            # keep the order of tokens specified in the vocab file by
            # adding them to the counter with decreasing counting values
            counters[name][token] = vocab_size - i + min_freq
    else:
        for token_and_count in vocab:
            if len(token_and_count) != 2:
                logger.info(f'Filtered invalid vocab token {token_and_count}')
                continue
            token, count = token_and_count
            counters[name][token] = int(count)
    return vocab, vocab_size


def _build_fv_from_multifield(multifield, counters, build_fv_kwargs,
                              size_multiple=1):
    for name, field in multifield:
        _build_field_vocab(
            field,
            counters[name],
            size_multiple=size_multiple,
            **build_fv_kwargs[name])
        logger.info(" * %s vocab size: %d." % (name, len(field.vocab)))


def _build_fields_vocab(fields, counters, data_type, share_vocab,
                        vocab_size_multiple,
                        src_vocab_size, src_words_min_frequency,
                        tgt_vocab_size, tgt_words_min_frequency,
                        ref_vocab_size, ref_words_min_frequency,
                        frag_vocab_size, frag_words_min_frequency,
                        src_specials=None, tgt_specials=None, ref_specials=None, frag_specials=None):
    has_ref = "ref" in fields
    has_frag = "frag" in fields
    src_specials = list(src_specials) if src_specials is not None else []
    tgt_specials = list(tgt_specials) if tgt_specials is not None else []
    if has_ref:
        ref_specials = list(ref_specials) if ref_specials is not None else []
    if has_frag:
        frag_specials = list(frag_specials) if frag_specials is not None else []
    build_fv_kwargs = defaultdict(dict)
    build_fv_kwargs["src"] = dict(
        max_size=src_vocab_size, min_freq=src_words_min_frequency,
        specials=src_specials)
    build_fv_kwargs["tgt"] = dict(
        max_size=tgt_vocab_size, min_freq=tgt_words_min_frequency,
        specials=tgt_specials)
    if has_ref:
        build_fv_kwargs["ref"] = dict(
            max_size=ref_vocab_size, min_freq=ref_words_min_frequency,
            specials=ref_specials)
        ref_multifield = fields["ref"]
        _build_fv_from_multifield(
            ref_multifield,
            counters,
            build_fv_kwargs,
            size_multiple=vocab_size_multiple if not share_vocab else 1)
    if has_frag:
        build_fv_kwargs["frag"] = dict(
            max_size=frag_vocab_size, min_freq=frag_words_min_frequency,
            specials=frag_specials)
        frag_multifield = fields["frag"]
        _build_fv_from_multifield(
            frag_multifield,
            counters,
            build_fv_kwargs,
            size_multiple=vocab_size_multiple if not share_vocab else 1)
    tgt_multifield = fields["tgt"]
    _build_fv_from_multifield(
        tgt_multifield,
        counters,
        build_fv_kwargs,
        size_multiple=vocab_size_multiple if not share_vocab else 1)

    if data_type == 'text':
        src_multifield = fields["src"]
        _build_fv_from_multifield(
            src_multifield,
            counters,
            build_fv_kwargs,
            size_multiple=vocab_size_multiple if not share_vocab else 1)

        if share_vocab:
            logger.info(" * merging src and tgt vocab...")
            src_field = src_multifield.base_field
            tgt_field = tgt_multifield.base_field
            if has_ref:
                logger.info(" * including ref and frag vocab in merge...")
                ref_field = ref_multifield.base_field
                frag_field = frag_multifield.base_field
                _all_specials = [item for item in src_specials + tgt_specials + ref_specials + frag_specials]
                _merge_field_vocabs(
                    src_field, tgt_field, ref_field, frag_field,
                    vocab_size=src_vocab_size,
                    min_freq=src_words_min_frequency,
                    vocab_size_multiple=vocab_size_multiple,
                    specials=_all_specials)
            else:
                # 如果没有ref，只合并src和tgt
                _all_specials = [item for item in src_specials + tgt_specials]
                _merge_field_vocabs_binary(
                    src_field, tgt_field,
                    vocab_size=src_vocab_size,
                    min_freq=src_words_min_frequency,
                    vocab_size_multiple=vocab_size_multiple,
                    specials=_all_specials)

            logger.info(" * merged vocab size: %d." % len(src_field.vocab))
    return fields


def build_vocab(train_dataset_files, fields, data_type, share_vocab,
                src_vocab_path, src_vocab_size, src_words_min_frequency,
                tgt_vocab_path, tgt_vocab_size, tgt_words_min_frequency,
                ref_vocab_path, ref_vocab_size, ref_words_min_frequency,
                frag_vocab_path, frag_vocab_size, frag_words_min_frequency,
                vocab_size_multiple=1):
    counters = defaultdict(Counter)

    if src_vocab_path:
        try:
            logger.info("Using existing vocabulary...")
            vocab = torch.load(src_vocab_path)
            # return vocab to dump with standard name
            return vocab
        except torch.serialization.pickle.UnpicklingError:
            logger.info("Building vocab from text file...")
            train_dataset_files = []

    # 处理词表路径
    vocab_paths = {
        'src': (src_vocab_path, src_words_min_frequency),
        'tgt': (tgt_vocab_path, tgt_words_min_frequency)
    }
    if ref_vocab_path:
        vocab_paths['ref'] = (ref_vocab_path, ref_words_min_frequency)
    if frag_vocab_path:
        vocab_paths['frag'] = (frag_vocab_path, frag_words_min_frequency)
    # 加载各个词表
    vocabs = {}
    for name, (path, min_freq) in vocab_paths.items():
        if path:
            vocabs[name], _ = _load_vocab(path, name, counters, min_freq)
        else:
            vocabs[name] = None

    for i, path in enumerate(train_dataset_files):
        dataset = torch.load(path)
        logger.info(" * reloading %s." % path)
        for ex in dataset.examples:
            for name, field in fields.items():
                # 跳过不需要构建词表的字段
                if name == "sim":
                    continue
                try:
                    f_iter = iter(field)
                except TypeError:
                    f_iter = [(name, field)]
                    all_data = [getattr(ex, name, None)]
                else:
                    all_data = getattr(ex, name)
                for (sub_n, sub_f), fd in zip(f_iter, all_data):
                    has_vocab = (sub_n == 'src' and vocabs['src']) or \
                                (sub_n == 'tgt' and vocabs['tgt']) or \
                                (sub_n == 'ref' and vocabs['ref']) or \
                                (sub_n == 'frag' and vocabs['frag'])
                    if sub_f.sequential and not has_vocab:
                        val = fd
                        counters[sub_n].update(val)

        # Drop the none-using from memory but keep the last
        if i < len(train_dataset_files) - 1:
            dataset.examples = None
            gc.collect()
            del dataset.examples
            gc.collect()
            del dataset
            gc.collect()

    fields = _build_fields_vocab(
        fields, counters, data_type,
        share_vocab, vocab_size_multiple,
        src_vocab_size, src_words_min_frequency,
        tgt_vocab_size, tgt_words_min_frequency,
        ref_vocab_size, ref_words_min_frequency,
        frag_vocab_size, frag_words_min_frequency,
    )

    return fields  # is the return necessary?


def _merge_field_vocabs(src_field, tgt_field, ref_field, frag_field,
                        vocab_size, min_freq,
                        vocab_size_multiple, specials):
    # 使用tgt_field的特殊token作为标准
    init_specials = [tgt_field.unk_token, tgt_field.pad_token,
                     tgt_field.init_token, tgt_field.eos_token]

    # 保持特殊token的唯一性和顺序
    all_specials = list(OrderedDict.fromkeys(
        tok for tok in init_specials + specials
        if tok is not None))

    # 合并四个字段的词频
    merged = sum(
        [src_field.vocab.freqs, tgt_field.vocab.freqs, ref_field.vocab.freqs, frag_field.vocab.freqs],
        Counter()
    )

    # 创建合并的词表
    merged_vocab = Vocab(
        merged, specials=all_specials,
        max_size=vocab_size, min_freq=min_freq
    )

    if vocab_size_multiple > 1:
        _pad_vocab_to_multiple(merged_vocab, vocab_size_multiple)

    # 将合并的词表应用到所有字段
    src_field.vocab = merged_vocab
    tgt_field.vocab = merged_vocab
    ref_field.vocab = merged_vocab
    frag_field.vocab = merged_vocab

    # 确保所有字段的词表大小相同
    assert len(src_field.vocab) == len(tgt_field.vocab) == len(ref_field.vocab) == len(frag_field.vocab)


def _merge_field_vocabs_binary(src_field, tgt_field,
                               vocab_size, min_freq,
                               vocab_size_multiple, specials):
    # 使用tgt_field的特殊token作为标准
    init_specials = [tgt_field.unk_token, tgt_field.pad_token,
                     tgt_field.init_token, tgt_field.eos_token]

    # 保持特殊token的唯一性和顺序
    all_specials = list(OrderedDict.fromkeys(
        tok for tok in init_specials + specials
        if tok is not None))

    # 只合并src和tgt的词频
    merged = sum(
        [src_field.vocab.freqs, tgt_field.vocab.freqs],
        Counter()
    )

    # 创建合并的词表
    merged_vocab = Vocab(
        merged, specials=all_specials,
        max_size=vocab_size, min_freq=min_freq
    )

    if vocab_size_multiple > 1:
        _pad_vocab_to_multiple(merged_vocab, vocab_size_multiple)

    # 将合并的词表应用到两个字段
    src_field.vocab = merged_vocab
    tgt_field.vocab = merged_vocab

    # 确保两个字段的词表大小相同
    assert len(src_field.vocab) == len(tgt_field.vocab)


def _read_vocab_file(vocab_path, tag):
    logger.info("Loading {} vocabulary from {}".format(tag, vocab_path))

    if not os.path.exists(vocab_path):
        raise RuntimeError(
            "{} vocabulary not found at {}".format(tag, vocab_path))
    else:
        with codecs.open(vocab_path, 'r', 'utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            first_line = lines[0].split(None, 1)
            has_count = (len(first_line) == 2 and first_line[-1].isdigit())
            if has_count:
                vocab = [line.split(None, 1) for line in lines]
            else:
                vocab = [line.strip().split()[0] for line in lines]
            return vocab, has_count
