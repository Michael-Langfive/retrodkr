import numpy as np
import pandas as pd
import argparse
import os
import re
import random
import math
import textdistance
import multiprocessing

from rdkit import Chem
from tqdm import tqdm

from rdkit import RDLogger

import selfies as sf

import codecs
from SmilesPE.tokenizer import *

from preprocessing.generate_data import process_single_product, process_single_cano_product

RDLogger.DisableLog('rdApp.*')

spe_vob = codecs.open('./SPE_ChEMBL.txt')
spe_tokenizer = SPE_Tokenizer(spe_vob, merges=-1)


def smi_tokenizer(smi, spe=False, self=False, dropout=0):  # dropout:  bpe dropout
    if spe:
        return spe_tokenizer.tokenize(smi, dropout=dropout)
    elif self:
        return ' '.join(sf.split_selfies(sf.encoder(smi)))
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)


def clear_map_canonical_smiles(smi, canonical=True, root=-1):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        for atom in mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom.ClearProp('molAtomMapNumber')
        return Chem.MolToSmiles(mol,
                                isomericSmiles=True,
                                rootedAtAtom=root,
                                canonical=canonical)
    else:
        return smi


def get_cano_map_number(smi, root=-1):
    atommap_mol = Chem.MolFromSmiles(smi)
    canonical_mol = Chem.MolFromSmiles(
        clear_map_canonical_smiles(smi, root=root))
    cano2atommapIdx = atommap_mol.GetSubstructMatch(canonical_mol)
    correct_mapped = [
        canonical_mol.GetAtomWithIdx(i).GetSymbol() ==
        atommap_mol.GetAtomWithIdx(index).GetSymbol()
        for i, index in enumerate(cano2atommapIdx)
    ]
    atom_number = len(canonical_mol.GetAtoms())
    if np.sum(correct_mapped) < atom_number or len(
            cano2atommapIdx) < atom_number:
        cano2atommapIdx = [0] * atom_number
        atommap2canoIdx = canonical_mol.GetSubstructMatch(atommap_mol)
        if len(atommap2canoIdx) != atom_number:
            return None
        for i, index in enumerate(atommap2canoIdx):
            cano2atommapIdx[index] = i
    id2atommap = [atom.GetAtomMapNum() for atom in atommap_mol.GetAtoms()]

    return [id2atommap[cano2atommapIdx[i]] for i in range(atom_number)]


def get_root_id(mol, root_map_number):
    root = -1
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomMapNum() == root_map_number:
            root = i
            break
    return root


"""multiprocess"""


def preprocess(save_dir, reactants, products,
               cano_products, ref_cano_reactants, sim_similarities,
               set_name, augmentation=1, root_aligned=True,
               character=False, processes=-1, retriever=False):
    """
    preprocess reaction data to extract graph adjacency matrix and features
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    processes = multiprocessing.cpu_count() if processes < 0 else processes
    print('processors: ', processes)
    if retriever:
        with multiprocessing.Pool(processes=processes) as pool:
            # 将产品列表分配给进程池并获取结果
            iterator = (
                pool.imap(process_single_product, products)
                if root_aligned
                else pool.imap(process_single_cano_product, cano_products)
            )
            # 用 tqdm 包装
            fragments = list(tqdm(
                iterator,
                total=len(products) if root_aligned else len(cano_products),
                desc="Processing fragments",
            ))
        data = [{
            "reactant": i,
            "product": j,
            "ref_cano_reactant": k,
            "similarity": l,
            "fragment": m,
            "augmentation": augmentation,
            "root_aligned": root_aligned,
        } for i, j, k, l, m in zip(reactants, products, ref_cano_reactants, sim_similarities, fragments)]
    else:
        data = [{
            "reactant": i,
            "product": j,
            "augmentation": augmentation,
            "root_aligned": root_aligned,
        } for i, j in zip(reactants, products)]

    src_data = []
    tgt_data = []
    ref_data = []
    sim_data = []
    frag_data = []
    with multiprocessing.Pool(processes=processes) as pool:  # 使用 with 上下文管理器时，multiprocessing.Pool 会自动处理资源清理
        # list() 会立即消费迭代器，触发 tqdm 的进度更新
        results = list(tqdm(pool.imap(multi_process, [(d, retriever) for d in data]), total=len(data),
                            desc="Processing reactions"))
    edit_distances, src2frag_edit_distances, tgt2frag_edit_distances = [], [], []
    for result in tqdm(results):
        if character:  # 字符级 tokenize 默认为false
            for i in range(len(result['src_data'])):
                result['src_data'][i] = " ".join([char for char in "".join(result['src_data'][i].split())])
            for i in range(len(result['tgt_data'])):
                result['tgt_data'][i] = " ".join([char for char in "".join(result['tgt_data'][i].split())])
            if retriever:
                for i in range(len(result['ref_data'])):
                    result['ref_data'][i] = " ".join([char for char in "".join(result['ref_data'][i].split())])
                for i in range(len(result['frag_data'])):
                    result['frag_data'][i] = " ".join([char for char in "".join(result['frag_data'][i].split())])
        edit_distances.append(result['edit_distance'])
        src_data.extend(result['src_data'])
        tgt_data.extend(result['tgt_data'])
        if retriever:
            src2frag_edit_distances.append(result['src2frag_edit_distances'])
            tgt2frag_edit_distances.append(result['tgt2frag_edit_distances'])
            ref_data.extend(result['ref_data'])
            sim_data.extend(result['sim_data'])
            frag_data.extend(result['frag_data'])
    print('size', len(src_data))
    print("Avg. src2tgt edit distance:", np.mean(edit_distances))
    if retriever:
        print("Avg. src2frag edit distance:", np.mean(src2frag_edit_distances))
        print("Avg. tgt2frag edit distance:", np.mean(tgt2frag_edit_distances))
    print('size', len(src_data))
    # if augmentation != 999:
    if augmentation > 0:
        with open(os.path.join(save_dir, '{}.src'.format(set_name)), 'w') as f:
            for src in src_data:
                f.write('{}\n'.format(src))
        with open(os.path.join(save_dir, '{}.tgt'.format(set_name)), 'w') as f:
            for tgt in tgt_data:
                f.write('{}\n'.format(tgt))
        if retriever:
            with open(os.path.join(save_dir, '{}.ref'.format(set_name)), 'w') as f:
                for ref in ref_data:
                    f.write('{}\n'.format(ref))
            with open(os.path.join(save_dir, '{}.sim'.format(set_name)), 'w') as f:
                for sim in sim_data:
                    f.write('{}\n'.format(sim))
            with open(os.path.join(save_dir, '{}.frag'.format(set_name)), 'w') as f:
                for frag in frag_data:
                    f.write('{}\n'.format(frag))
    return src_data, tgt_data, ref_data, sim_data, frag_data


def multi_process(data):
    data, retriever = data
    shuffle = args.shuffle  # False
    mixed = args.mixed  # False
    product = data['product']
    reactant = data['reactant']
    augmentation = data['augmentation']
    if retriever:
        ref_cano_reactant = data['ref_cano_reactant']
        fragment = data['fragment'].split(".")
        similarity = data['similarity']
    pro_mol = Chem.MolFromSmiles(product)
    return_status = {
        "status": 0,
        "src_data": [],
        "tgt_data": [],
        "ref_data": [],
        "sim_data": [],
        "frag_data": [],
        "edit_distance": 0,
        "src2frag_edit_distances": 0,
        "tgt2frag_edit_distances": 0,
    }

    pro_atom_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", product)))
    reactant = reactant.split(".")
    if data['root_aligned']:
        reversable = False  # no shuffle # TODO:
        if augmentation == 999:
            product_roots = pro_atom_map_numbers
            times = len(product_roots)
        else:
            product_roots = [-1]
            max_times = len(pro_atom_map_numbers)
            times = min(augmentation, max_times)
            if times < augmentation:  # times = max_times
                product_roots.extend(pro_atom_map_numbers)
                product_roots.extend(
                    random.choices(product_roots,
                                   k=augmentation - len(product_roots)))
            else:  # times = augmentation
                while len(product_roots) < times:
                    product_roots.append(
                        random.sample(pro_atom_map_numbers, 1)[0])
                    if product_roots[-1] in product_roots[:-1]:
                        product_roots.pop()
            times = len(product_roots)
            assert times == augmentation
            if reversable:
                times = int(times / 2)
        for k in range(times):
            pro_root_atom_map = product_roots[k]
            pro_root = get_root_id(pro_mol,
                                   root_map_number=pro_root_atom_map)
            cano_atom_map = get_cano_map_number(product, root=pro_root)
            if cano_atom_map is None:
                return_status["status"] = "error_mapping"
                return return_status
            pro_smi = clear_map_canonical_smiles(product,
                                                 canonical=True,
                                                 root=pro_root)
            aligned_reactants = []
            aligned_reactants_order = []
            rea_atom_map_numbers = [
                list(map(int, re.findall(r"(?<=:)\d+", rea)))
                for rea in reactant
            ]
            used_indices = []
            for i, rea_map_number in enumerate(rea_atom_map_numbers):
                for j, map_number in enumerate(cano_atom_map):
                    # select mapping reactans
                    if map_number in rea_map_number:
                        rea_root = get_root_id(Chem.MolFromSmiles(
                            reactant[i]),
                            root_map_number=map_number)
                        rea_smi = clear_map_canonical_smiles(
                            reactant[i], canonical=True, root=rea_root)
                        aligned_reactants.append(rea_smi)
                        aligned_reactants_order.append(j)
                        used_indices.append(i)
                        break
            if retriever:
                frag_map_numbers = [list(map(int, re.findall(r"(?<=:)\d+", frag))) for frag in fragment]
                aligned_fragments = []
                aligned_fragments_order = []
                for i, frag_map_number in enumerate(frag_map_numbers):
                    for j, map_number in enumerate(cano_atom_map):
                        if map_number in frag_map_number:
                            # 片段处理
                            frag_mol = Chem.MolFromSmiles(fragment[i])
                            if frag_mol:
                                frag_root = get_root_id(frag_mol, root_map_number=map_number)
                                frag_smi = clear_map_canonical_smiles(fragment[i], canonical=True, root=frag_root)
                                aligned_fragments.append(frag_smi)
                                aligned_fragments_order.append(j)
                            break
                aligned_fragments = [f for _, f in sorted(zip(aligned_fragments_order, aligned_fragments))]
                fragment_smi = '.'.join(aligned_fragments)
                frag_tokens = smi_tokenizer(fragment_smi, args.spe, args.self, args.dropout)
                return_status['frag_data'].append(frag_tokens)
                ref_tokens = smi_tokenizer(ref_cano_reactant, args.spe, args.self, args.dropout)
                return_status['ref_data'].append(ref_tokens)
                return_status['sim_data'].append(similarity)
            sorted_reactants = sorted(list(
                zip(aligned_reactants, aligned_reactants_order)),
                key=lambda x: x[1])
            aligned_reactants = [item[0] for item in sorted_reactants]
            if shuffle:
                random.shuffle(aligned_reactants)
            reactant_smi = ".".join(aligned_reactants)
            product_tokens = smi_tokenizer(pro_smi, args.spe, args.self, args.dropout)
            reactant_tokens = smi_tokenizer(reactant_smi, args.spe, args.self, args.dropout)
            return_status['src_data'].append(product_tokens)
            return_status['tgt_data'].append(reactant_tokens)
            if mixed:
                return_status['src_data'].append(
                    smi_tokenizer(rea_smi, args.spe, args.self, args.dropout))
                return_status['tgt_data'].append(
                    smi_tokenizer(pro_smi, args.spe, args.self, args.dropout))
            if reversable:
                reactant_smi = ".".join(aligned_reactants)
                product_tokens = smi_tokenizer(pro_smi, args.spe, args.self, args.dropout)
                reactant_tokens = smi_tokenizer(reactant_smi, args.spe, args.self, args.dropout)
                product_tokens_list = product_tokens.split(' ')
                product_tokens_list.reverse()
                product_tokens = ' '.join(product_tokens_list)
                return_status['src_data'].append(product_tokens)
                return_status['tgt_data'].append(reactant_tokens)
                if retriever:
                    ref_tokens = smi_tokenizer(ref_cano_reactant, args.spe, args.self, args.dropout)
                    frag_tokens = smi_tokenizer(fragment, args.spe, args.self, args.dropout)
                    return_status['ref_data'].append(ref_tokens)
                    return_status['sim_data'].append(similarity)
                    return_status['frag_data'].append(frag_tokens)
                if mixed:
                    return_status['src_data'].append(
                        smi_tokenizer(rea_smi, args.spe, args.self, args.dropout))
                    return_status['tgt_data'].append(
                        smi_tokenizer(pro_smi, args.spe, args.self, args.dropout))
    else:
        cano_product = clear_map_canonical_smiles(product)
        cano_reactant = ".".join([
            clear_map_canonical_smiles(rea) for rea in reactant if len(
                set(map(int, re.findall(r"(?<=:)\d+", rea)))
                & set(pro_atom_map_numbers)) > 0
        ])
        return_status['src_data'].append(
            smi_tokenizer(cano_product, args.spe, args.self, args.dropout))
        return_status['tgt_data'].append(
            smi_tokenizer(cano_reactant, args.spe, args.self, args.dropout))
        if retriever:
            fragment = ".".join([clear_map_canonical_smiles(frag) for frag in fragment if
                                 len(set(map(int, re.findall(r"(?<=:)\d+", frag))) & set(
                                     pro_atom_map_numbers)) > 0])
            return_status['ref_data'].append(smi_tokenizer(ref_cano_reactant, args.spe, args.self, args.dropout))
            return_status['sim_data'].append(similarity)
            return_status['frag_data'].append(smi_tokenizer(fragment, args.spe, args.self, args.dropout))
            frag_mols = [Chem.MolFromSmiles(frag) for frag in fragment.split(".")]

        pro_mol = Chem.MolFromSmiles(cano_product)
        rea_mols = [
            Chem.MolFromSmiles(rea) for rea in cano_reactant.split(".")
        ]
        for i in range(int(augmentation - 1)):
            pro_smi = Chem.MolToSmiles(pro_mol, doRandom=True)
            rea_smi = [
                Chem.MolToSmiles(rea_mol, doRandom=True)
                for rea_mol in rea_mols
            ]
            if shuffle:
                random.shuffle(rea_smi)
            rea_smi = ".".join(rea_smi)
            return_status['src_data'].append(
                smi_tokenizer(pro_smi, args.spe, args.self, args.dropout))
            return_status['tgt_data'].append(
                smi_tokenizer(rea_smi, args.spe, args.self, args.dropout))
            if retriever:
                frag_smi = [Chem.MolToSmiles(frag_mol, doRandom=True) for frag_mol in frag_mols]
                frag_smi = ".".join(frag_smi)
                return_status['ref_data'].append(smi_tokenizer(ref_cano_reactant, args.spe, args.self, args.dropout))
                return_status['sim_data'].append(similarity)
                return_status['frag_data'].append(smi_tokenizer(frag_smi, args.spe, args.self, args.dropout))
            if mixed:  # 加入正向信息
                return_status['src_data'].append(
                    smi_tokenizer(rea_smi, args.spe, args.self, args.dropout))
                return_status['tgt_data'].append(
                    smi_tokenizer(pro_smi, args.spe, args.self, args.dropout))

    edit_distances = []
    for src, tgt in zip(return_status['src_data'],
                        return_status['tgt_data']):
        edit_distances.append(
            textdistance.levenshtein.distance(src.split(), tgt.split()))
    return_status['edit_distance'] = np.mean(edit_distances)
    if retriever:
        src2frag_edit_distances, tgt2frag_edit_distances = [], []
        for src, frag in zip(return_status['src_data'], return_status['frag_data']):
            src2frag_edit_distances.append(textdistance.levenshtein.distance(src.split(), frag.split()))
        return_status['src2frag_edit_distances'] = np.mean(src2frag_edit_distances)
        for tgt, frag in zip(return_status['tgt_data'], return_status['frag_data']):
            tgt2frag_edit_distances.append(textdistance.levenshtein.distance(tgt.split(), frag.split()))
        return_status['tgt2frag_edit_distances'] = np.mean(tgt2frag_edit_distances)

    return return_status


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default='USPTO_50K')
    parser.add_argument("-augmentation", type=int, default=1)
    parser.add_argument("-seed", type=int, default=33)
    parser.add_argument("-processes", type=int, default=-1)  # -1自动计算cpu核心数
    parser.add_argument("-test_only", action="store_true")
    parser.add_argument("-train_only", action="store_true")
    parser.add_argument("-test_except", action="store_true")
    parser.add_argument("-train_except", action="store_true")
    parser.add_argument("-validastrain", action="store_true")
    parser.add_argument("-character", action="store_true")
    parser.add_argument("-canonical", action="store_true")
    parser.add_argument("-postfix", type=str, default="")
    parser.add_argument('-spe', default="spe", action="store_true")  # SPE Tokenization
    parser.add_argument('-self', action="store_true")
    parser.add_argument('-dropout', type=float, default=0.0)
    parser.add_argument('-shuffle', action='store_true')
    parser.add_argument('-mixed', action='store_true')
    parser.add_argument('-batch', type=int, default=-1)
    parser.add_argument("-method", type=str, default="")  # 这里只控制读取路径 保存全部在一个目录 一次只能一个指纹
    parser.add_argument("-retriever", action="store_true")  # 布尔类型的标志（flag）
    args = parser.parse_args()
    print('preprocessing dataset {}...'.format(args.dataset))
    assert args.dataset in ['USPTO_50K', 'USPTO_FULL']
    print(args)
    if args.test_only:
        datasets = ['test']
    elif args.train_only:
        datasets = ['train']
    elif args.test_except:
        datasets = ['val', 'train']
    elif args.train_except:
        datasets = ['val', 'test']
    elif args.validastrain:
        datasets = ['test', 'val', 'train']
    else:
        datasets = ['test', 'val', 'train']

    random.seed(args.seed)

    # datadir = '../datasets/{}/raw'.format(args.dataset)
    datadir = '../datasets/{}'.format(args.dataset)
    if args.spe:
        savedir = '../datasets/{}/aug{}'.format(args.dataset, args.augmentation)
    elif args.self:
        savedir = '../datasets/{}/aug{}_self'.format(args.dataset, args.augmentation)
    else:
        savedir = '../datasets/{}/aug{}_token'.format(args.dataset, args.augmentation)

    savedir += args.postfix
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    for i, data_set in enumerate(datasets):
        if args.retriever:  # 这里只控制读取路径 保存全部在一个目录 一次只能一个指纹
            csv_path = f"{datadir}/processed/{args.method}/processed_{data_set}.csv"
        else:
            csv_path = f"{datadir}/cleaned/cleaned_{data_set}.csv"

        csv = pd.read_csv(csv_path)
        reactant_smarts_list = list(csv["reactant"])
        product_smarts_list = list(csv["product"])
        product_cano_smarts_list = list(csv["cano_product"])
        ref_cano_reactant_smarts_list = []
        similarity_list = []
        if args.retriever:
            ref_cano_reactant_smarts_list = list(csv["sim_cano_reactant"])
            similarity_list = list(csv["similarity"])

        if args.validastrain and data_set == "train":
            csv_path = f"{datadir}/processed/processed_val.csv"
            csv = pd.read_csv(csv_path)
            reactant_smarts_list += list(csv["reactant"])
            product_smarts_list += list(csv["product"])
            if args.retriever:
                ref_cano_reactant_smarts_list += list(csv["sim_cano_reactant"])
                similarity_list += list(csv["similarity"])

        print("Total Data Size", len(reactant_smarts_list))
        save_dir = savedir

        src_data, tgt_data, ref_data, sim_data, frag_data = preprocess(
            save_dir,
            reactant_smarts_list,
            product_smarts_list,
            product_cano_smarts_list,
            ref_cano_reactant_smarts_list,
            similarity_list,
            data_set,
            args.augmentation,
            root_aligned=not args.canonical,
            character=args.character,
            processes=args.processes,
            retriever=args.retriever
        )
