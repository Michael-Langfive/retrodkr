"""Module that contain shard utils for dynamic data."""
import os
from itertools import repeat

from onmt.utils.logging import logger
from onmt.constants import CorpusName, ModelTask
from onmt.transforms import TransformPipe
from onmt.inputters.dataset_base import _dynamic_dict
from torchtext.data import Dataset as TorchtextDataset, \
    Example as TorchtextExample

from collections import Counter, defaultdict
from contextlib import contextmanager

import multiprocessing as mp
from collections import defaultdict


@contextmanager
def exfile_open(filename, *args, **kwargs):
    if filename is None:
        from itertools import repeat
        _file = repeat(None)
    else:
        import codecs
        _file = codecs.open(filename, *args, **kwargs)
    yield _file
    if filename is not None and _file:
        _file.close()


class DatasetAdapter(object):
    valid_field_name = (
        'src', 'tgt', 'ref', 'sim', 'frag', 'indices',
        'src_map', 'ref_map','frag_map',
        'src_ex_vocab', 'alignment', 'align')

    def __init__(self, fields, is_train):
        self.fields_dict = self._valid_fields(fields)
        self.is_train = is_train

    @classmethod
    def _valid_fields(cls, fields):
        """Return valid fields in dict format."""
        return {
            f_k: f_v for f_k, f_v in fields.items()
            if f_k in cls.valid_field_name
        }

    @staticmethod
    def _process(item, is_train):
        """Return valid transformed example from `item`."""
        example, transform, cid = item
        # this is a hack: appears quicker to apply it here
        # than in the ParallelCorpusIterator
        maybe_example = transform.apply(
            example, is_train=is_train, corpus_name=cid)
        if maybe_example is None:
            return None

        maybe_example['src'] = {"src": ' '.join(maybe_example['src'])}
        if 'src_feats' in maybe_example:
            for feat_name, feat_value in maybe_example['src_feats'].items():
                maybe_example['src'][feat_name] = ' '.join(feat_value)
            del maybe_example["src_feats"]
        maybe_example['tgt'] = {"tgt": ' '.join(maybe_example['tgt'])}
        # 兼容没有ref数据的情况
        if 'ref' in maybe_example:
            maybe_example['ref'] = {"ref": ' '.join(maybe_example['ref'])}
        if 'frag' in maybe_example:
            maybe_example['frag'] = {"frag": ' '.join(maybe_example['frag'])}

        # 兼容没有sim数据的情况
        if 'sim' in maybe_example:
            maybe_example['sim'] = {"sim": maybe_example['sim']}

        if 'align' in maybe_example:
            maybe_example['align'] = ' '.join(maybe_example['align'])
        return maybe_example

    def _maybe_add_dynamic_dict(self, example, fields):
        """maybe update `example` with dynamic_dict related fields."""
        if 'src_map' in fields and 'alignment' in fields:
            example = _dynamic_dict(
                example,
                fields['src'].base_field,
                fields['tgt'].base_field,
            )
        return example

    def _to_examples(self, bucket, is_train=False):
        examples = []
        for item in bucket:
            maybe_example = self._process(item, is_train=is_train)
            if maybe_example is not None:
                example = self._maybe_add_dynamic_dict(
                    maybe_example, self.fields_dict)
                ex_fields = {k: [(k, v)] for k, v in self.fields_dict.items()
                             if k in example}
                ex = TorchtextExample.fromdict(example, ex_fields)
                examples.append(ex)
        return examples

    def __call__(self, bucket):
        examples = self._to_examples(bucket, is_train=self.is_train)
        dataset = TorchtextDataset(examples, self.fields_dict)
        return dataset


class ParallelCorpus(object):
    """A parallel corpus file pair that can be loaded to iterate."""

    def __init__(self, name, src, tgt, ref=None, sim=None, frag=None, align=None, src_feats=None):
        """Initialize file path."""
        self.id = name
        self.src = src
        self.tgt = tgt
        self.ref = ref  # 新增 ref 属性
        self.sim = sim  # 新增 sim 属性
        self.frag = frag  # 新增 frag 属性
        self.align = align
        self.src_feats = src_feats

    def load(self, offset=0, stride=1):
        """
        Load file and iterate by lines.
        `offset` and `stride` allow to iterate only on every
        `stride` example, starting from `offset`.
        """
        if self.src_feats:
            features_names = []
            features_files = []
            for feat_name, feat_path in self.src_feats.items():
                features_names.append(feat_name)
                features_files.append(open(feat_path, mode='rb'))
        else:
            features_files = []

        with exfile_open(self.src, mode='rb') as fs, \
                exfile_open(self.tgt, mode='rb') as ft, \
                exfile_open(self.align, mode='rb') as fa, \
                exfile_open(self.ref, mode='rb') as fr, \
                exfile_open(self.sim, mode='rb') as fsim, \
                exfile_open(self.frag, mode='rb') as ff:  # 新增 sim 文件
            for i, (sline, tline, align, rline, simline, fline, *features) in enumerate(
                    zip(fs, ft, fa, fr, fsim, ff, *features_files)):
                if (i % stride) == offset:
                    sline = sline.decode('utf-8') if sline else None
                    tline = tline.decode('utf-8') if tline else None
                    example = {
                        'src': sline,
                        'tgt': tline,
                    }
                    # 只有当有ref数据时才添加ref字段
                    if rline is not None:
                        example['ref'] = rline.decode('utf-8')
                    # 只有当有sim数据时才添加sim字段
                    if simline is not None:
                        example['sim'] = simline.decode('utf-8')
                    # 只有当有frag数据时才添加frag字段
                    if fline is not None:
                        example['frag'] = fline.decode('utf-8')
                    if align is not None:
                        example['align'] = align.decode('utf-8')
                    if features:
                        example["src_feats"] = dict()
                        for j, feat in enumerate(features):
                            example["src_feats"][features_names[j]] = feat.decode("utf-8")
                    yield example
        for f in features_files:
            f.close()

    def __str__(self):
        cls_name = type(self).__name__
        return '{}({}, {}, ref={}, sim={}, align={}, frag={}, src_feats={})'.format(
            cls_name, self.src, self.tgt, self.ref, self.sim,self.frag, self.align, self.src_feats)


def get_corpora(opts, is_train=False):
    corpora_dict = {}
    if is_train:
        for corpus_id, corpus_dict in opts.data.items():
            if corpus_id != CorpusName.VALID:
                corpora_dict[corpus_id] = ParallelCorpus(
                    corpus_id,
                    corpus_dict["path_src"],
                    corpus_dict["path_tgt"],
                    corpus_dict.get("path_ref"),  # 新增 ref 路径
                    corpus_dict.get("path_sim"),  # 新增 sim 路径
                    corpus_dict.get("path_frag"),  # 新增 frag 路径
                    corpus_dict["path_align"],
                    corpus_dict["src_feats"])
    else:
        if CorpusName.VALID in opts.data.keys():
            corpora_dict[CorpusName.VALID] = ParallelCorpus(
                CorpusName.VALID,
                opts.data[CorpusName.VALID]["path_src"],
                opts.data[CorpusName.VALID]["path_tgt"],
                opts.data[CorpusName.VALID].get("path_ref"),  # 新增 ref 路径
                opts.data[CorpusName.VALID].get("path_sim"),  # 新增 sim 路径
                opts.data[CorpusName.VALID].get("path_frag"),  # 新增 frag 路径
                opts.data[CorpusName.VALID]["path_align"],
                opts.data[CorpusName.VALID]["src_feats"])
        else:
            return None
    return corpora_dict


class ParallelCorpusIterator(object):
    def __init__(self, corpus, transform,
                 skip_empty_level='warning', stride=1, offset=0):
        self.cid = corpus.id
        self.corpus = corpus
        self.transform = transform
        if skip_empty_level not in ['silent', 'warning', 'error']:
            raise ValueError(
                f"Invalid argument skip_empty_level={skip_empty_level}")
        self.skip_empty_level = skip_empty_level
        self.stride = stride
        self.offset = offset

    def _tokenize(self, stream):
        for example in stream:
            example['src'] = example['src'].strip('\n').split()
            example['tgt'] = example['tgt'].strip('\n').split()
            # 只处理存在的ref字段
            if 'ref' in example:
                example['ref'] = example['ref'].strip('\n').split()
            # 只处理存在的sim字段
            if 'sim' in example:
                example['sim'] = float(example['sim'].strip('\n'))
            # 只处理存在的frag字段
            if 'frag' in example:
                example['frag'] = example['frag'].strip('\n').split()
            if 'align' in example:
                example['align'] = example['align'].strip('\n').split()
            if 'src_feats' in example:
                for k in example['src_feats'].keys():
                    example['src_feats'][k] = example['src_feats'][k].strip('\n').split()
            yield example

    def _transform(self, stream):
        for example in stream:
            # NOTE: moved to DatasetAdapter._process method in iterator.py
            # item = self.transform.apply(
            # example, is_train=self.infinitely, corpus_name=self.cid)
            item = (example, self.transform, self.cid)
            if item is not None:
                yield item
        report_msg = self.transform.stats()
        if report_msg != '':
            logger.info(
                "* Transform statistics for {}({:.2f}%):\n{}\n".format(
                    self.cid, 100 / self.stride, report_msg
                )
            )

    def _add_index(self, stream):
        for i, item in enumerate(stream):
            example = item[0]
            line_number = i * self.stride + self.offset
            example['indices'] = line_number
            if (len(example['src']) == 0 or
                    len(example['tgt']) == 0 or
                    ('ref' in example and len(example['ref']) == 0) or
                    ('frag' in example and len(example['frag']) == 0) or
                    ('align' in example and example['align'] == 0)):
                # empty example: skip
                empty_msg = f"Empty line exists in {self.cid}#{line_number}."
                if self.skip_empty_level == 'error':
                    raise IOError(empty_msg)
                elif self.skip_empty_level == 'warning':
                    logger.warning(empty_msg)
                continue
            yield item

    def __iter__(self):
        corpus_stream = self.corpus.load(
            stride=self.stride, offset=self.offset
        )
        tokenized_corpus = self._tokenize(corpus_stream)
        transformed_corpus = self._transform(tokenized_corpus)
        indexed_corpus = self._add_index(transformed_corpus)
        yield from indexed_corpus


def build_corpora_iters(corpora, transforms, corpora_info,
                        skip_empty_level='warning', stride=1, offset=0):
    """Return `ParallelCorpusIterator` for all corpora defined in opts."""
    corpora_iters = dict()
    for c_id, corpus in corpora.items():
        transform_names = corpora_info[c_id].get('transforms', [])
        corpus_transform = [
            transforms[name] for name in transform_names if name in transforms
        ]
        transform_pipe = TransformPipe.build_from(corpus_transform)
        logger.info(f"{c_id}'s transforms: {str(transform_pipe)}")
        corpus_iter = ParallelCorpusIterator(
            corpus, transform_pipe,
            skip_empty_level=skip_empty_level, stride=stride, offset=offset)
        corpora_iters[c_id] = corpus_iter
    return corpora_iters


def write_files_from_queues(sample_path, queues):
    """
    Standalone process that reads data from
    queues in order and write to sample files.
    """
    os.makedirs(sample_path, exist_ok=True)
    for c_name in queues.keys():
        dest_base = os.path.join(
            sample_path, "{}.{}".format(c_name, CorpusName.SAMPLE))
        with open(dest_base + ".src", 'w', encoding="utf-8") as f_src, \
                open(dest_base + ".tgt", 'w', encoding="utf-8") as f_tgt, \
                open(dest_base + ".ref", 'w', encoding="utf-8") as f_ref, \
                open(dest_base + ".sim", 'w', encoding="utf-8") as f_sim, \
                open(dest_base + ".frag", 'w', encoding="utf-8") as f_frag:
            while True:
                _next = False
                for q in queues[c_name]:
                    item = q.get()
                    if item == "blank":
                        continue
                    if item == "break":
                        _next = True
                        break
                    _, src_line, tgt_line, ref_line, sim_line, frag_line = item  # 假设队列传递 ref
                    f_src.write(src_line + '\n')
                    f_tgt.write(tgt_line + '\n')
                    f_ref.write(ref_line + '\n')
                    f_sim.write(sim_line + '\n')
                    f_frag.write(frag_line + '\n')
                if _next:
                    break


def build_sub_vocab(corpora, transforms, opts, n_sample, stride, offset):
    """Build vocab on (strided) subpart of the data."""
    sub_counter_src = Counter()
    sub_counter_tgt = Counter()
    if opts.data_task == ModelTask.RetroDKR:
        sub_counter_ref = Counter()    # 新增 ref 计数器
        sub_counter_frag = Counter()  # 新增 frag 计数器
    else:
        sub_counter_ref, sub_counter_frag = None, None
    sub_counter_src_feats = defaultdict(Counter)
    datasets_iterables = build_corpora_iters(
        corpora, transforms, opts.data,
        skip_empty_level=opts.skip_empty_level,
        stride=stride, offset=offset)
    for c_name, c_iter in datasets_iterables.items():
        for i, item in enumerate(c_iter):
            maybe_example = DatasetAdapter._process(item, is_train=True)
            if maybe_example is None:
                if opts.dump_samples:
                    build_sub_vocab.queues[c_name][offset].put("blank")
                continue
            src_line = maybe_example['src']['src']
            tgt_line = maybe_example['tgt']['tgt']
            # 获取ref和sim，如果存在
            ref_line = maybe_example.get('ref', {}).get('ref', '')
            sim_line = maybe_example.get('sim', {}).get('sim', '')
            frag_line = maybe_example.get('frag', {}).get('frag', '')

            for feat_name, feat_line in maybe_example["src"].items():
                if feat_name != "src":
                    sub_counter_src_feats[feat_name].update(feat_line.split(' '))
            sub_counter_src.update(src_line.split(' '))
            sub_counter_tgt.update(tgt_line.split(' '))
            # 只有当ref存在时更新ref计数器
            if ref_line:
                sub_counter_ref.update(ref_line.split(' '))
            if frag_line:
                sub_counter_frag.update(frag_line.split(' '))
            if opts.dump_samples:
                build_sub_vocab.queues[c_name][offset].put(
                    (i, src_line, tgt_line, ref_line, str(sim_line)))
            if n_sample > 0 and ((i + 1) * stride + offset) >= n_sample:
                if opts.dump_samples:
                    build_sub_vocab.queues[c_name][offset].put("break")
                break
        if opts.dump_samples:
            build_sub_vocab.queues[c_name][offset].put("break")
    return sub_counter_src, sub_counter_tgt, sub_counter_ref, sub_counter_frag, sub_counter_src_feats


def init_pool(queues):
    """Add the queues as attribute of the pooled function."""
    build_sub_vocab.queues = queues


def build_vocab(opts, transforms, n_sample=3):
    """Build vocabulary from data."""

    if n_sample == -1:
        logger.info(f"n_sample={n_sample}: Build vocab on full datasets.")
    elif n_sample > 0:
        logger.info(f"Build vocab on {n_sample} transformed examples/corpus.")
    else:
        raise ValueError(f"n_sample should > 0 or == -1, get {n_sample}.")

    if opts.dump_samples:
        logger.info("The samples on which the vocab is built will be "
                    "dumped to disk. It may slow down the process.")
    corpora = get_corpora(opts, is_train=True)
    counter_src = Counter()
    counter_tgt = Counter()
    if opts.data_task == ModelTask.RetroDKR:
        counter_ref, counter_frag = Counter(), Counter()
    else:
        counter_ref, counter_frag = None, None

    counter_src_feats = defaultdict(Counter)
    from functools import partial
    queues = {c_name: [mp.Queue(opts.vocab_sample_queue_size)
                       for i in range(opts.num_threads)]
              for c_name in corpora.keys()}
    sample_path = os.path.join(
        os.path.dirname(opts.save_data), CorpusName.SAMPLE)
    if opts.dump_samples:
        write_process = mp.Process(
            target=write_files_from_queues,
            args=(sample_path, queues),
            daemon=True)
        write_process.start()
    with mp.Pool(opts.num_threads, init_pool, [queues]) as p:
        func = partial(
            build_sub_vocab, corpora, transforms,
            opts, n_sample, opts.num_threads)
        for sub_counter_src, sub_counter_tgt, sub_counter_ref, sub_counter_frag, sub_counter_src_feats in p.imap(
                func, range(0, opts.num_threads)):
            counter_src.update(sub_counter_src)
            counter_tgt.update(sub_counter_tgt)
            if opts.data_task == ModelTask.RetroDKR:
                counter_ref.update(sub_counter_ref)  # 更新 ref 计数器
                counter_frag.update(sub_counter_frag) # 更新 frag 计数器
            counter_src_feats.update(sub_counter_src_feats)
    if opts.dump_samples:
        write_process.join()
    return counter_src, counter_tgt, counter_ref, counter_frag, counter_src_feats


def save_transformed_sample(opts, transforms, n_sample=3):
    """Save transformed data sample as specified in opts."""

    if n_sample == -1:
        logger.info(f"n_sample={n_sample}: Save full transformed corpus.")
    elif n_sample == 0:
        logger.info(f"n_sample={n_sample}: no sample will be saved.")
        return
    elif n_sample > 0:
        logger.info(f"Save {n_sample} transformed example/corpus.")
    else:
        raise ValueError(f"n_sample should >= -1, get {n_sample}.")

    corpora = get_corpora(opts, is_train=True)
    datasets_iterables = build_corpora_iters(
        corpora, transforms, opts.data,
        skip_empty_level=opts.skip_empty_level)
    sample_path = os.path.join(
        os.path.dirname(opts.save_data), CorpusName.SAMPLE)
    os.makedirs(sample_path, exist_ok=True)
    for c_name, c_iter in datasets_iterables.items():
        dest_base = os.path.join(
            sample_path, "{}.{}".format(c_name, CorpusName.SAMPLE))
        with open(dest_base + ".src", 'w', encoding="utf-8") as f_src, \
                open(dest_base + ".tgt", 'w', encoding="utf-8") as f_tgt, \
                open(dest_base + ".ref", 'w', encoding="utf-8") as f_ref, \
                open(dest_base + ".sim", 'w', encoding="utf-8") as f_sim, \
                open(dest_base + ".frag", 'w', encoding="utf-8") as f_frag:  # 新增 sim 文件
            for i, item in enumerate(c_iter):
                maybe_example = DatasetAdapter._process(item, is_train=True)
                if maybe_example is None:
                    continue
                src_line = maybe_example['src']['src']
                tgt_line = maybe_example['tgt']['tgt']
                ref_line = maybe_example.get('ref', {}).get('ref', '')  # 使用安全的get方法
                sim_line = maybe_example.get('sim', {}).get('sim', '')  # 使用安全的get方法
                frag_line = maybe_example.get('frag', {}).get('frag', '')  # 使用安全的get方法
                f_src.write(src_line + '\n')
                f_tgt.write(tgt_line + '\n')
                if 'ref' in maybe_example:  # 只在ref存在时写入
                    f_ref.write(ref_line + '\n')
                if 'sim' in maybe_example:  # 只在sim存在时写入
                    f_sim.write(sim_line + '\n')
                if 'frag' in maybe_example:  # 只在sim存在时写入
                    f_frag.write(frag_line + '\n')
                if n_sample > 0 and i >= n_sample:
                    break
