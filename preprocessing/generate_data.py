import argparse
import multiprocessing
import os
import random
import re

import numpy as np
import pandas as pd
import textdistance
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.BRICS import BRICSDecompose, FindBRICSBonds, BreakBRICSBonds
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')


def smi_tokenizer(smi):
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
        return Chem.MolToSmiles(mol, isomericSmiles=True, rootedAtAtom=root, canonical=canonical)
    else:
        return smi


def get_cano_map_number(smi, root=-1):
    atommap_mol = Chem.MolFromSmiles(smi)
    canonical_mol = Chem.MolFromSmiles(clear_map_canonical_smiles(smi, root=root))
    cano2atommapIdx = atommap_mol.GetSubstructMatch(canonical_mol)
    correct_mapped = [canonical_mol.GetAtomWithIdx(i).GetSymbol() == atommap_mol.GetAtomWithIdx(index).GetSymbol() for
                      i, index in enumerate(cano2atommapIdx)]
    atom_number = len(canonical_mol.GetAtoms())
    if np.sum(correct_mapped) < atom_number or len(cano2atommapIdx) < atom_number:
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


#
def process_single_product(product):
    mol = Chem.MolFromSmiles(product)
    if mol is not None:
        try:
            for atom in mol.GetAtoms():
                atom.SetIntProp("_OrigIdx", atom.GetIdx())
            orig_mapping = {atom.GetIdx(): atom.GetAtomMapNum() for atom in mol.GetAtoms()}
            bonds = list(FindBRICSBonds(mol))
            if not bonds:
                return Chem.MolToSmiles(mol, True)

            frag_mol = BreakBRICSBonds(mol, bonds)
            idx_map = {}
            for atom in frag_mol.GetAtoms():
                if atom.HasProp("_OrigIdx"):
                    idx_map[atom.GetIdx()] = atom.GetUnsignedProp("_OrigIdx")
            frags = Chem.GetMolFrags(frag_mol, asMols=True)
            frag_indices = Chem.GetMolFrags(frag_mol, asMols=False)
            fragment_smiles_list = []
            for frag, atom_indices in zip(frags, frag_indices):
                editable_frag = Chem.RWMol(frag)
                for i, frag_idx in enumerate(atom_indices):
                    if frag_idx in idx_map:
                        orig_idx = idx_map[frag_idx]
                        atom = editable_frag.GetAtomWithIdx(i)
                        if atom.GetAtomMapNum() == 0:
                            atom.SetAtomMapNum(orig_mapping[orig_idx])
                frag_smiles = Chem.MolToSmiles(editable_frag, True)
                smilesH = re.sub(r'\[\d+\*]', '[H]', frag_smiles)
                processed_mol = Chem.MolFromSmiles(smilesH)
                if processed_mol is not None:
                    processed_mol = Chem.RemoveHs(processed_mol)
                    Chem.SanitizeMol(processed_mol)
                    final_smiles = Chem.MolToSmiles(processed_mol)
                    fragment_smiles_list.append(final_smiles)
                else:
                    fragment_smiles_list.append(smilesH)
            return '.'.join(fragment_smiles_list)
        except Exception as e:
            print(f"片段化 {product} 时出错: {e}")
    else:
        return ''


def process_single_cano_product(product):
    mol = Chem.MolFromSmiles(product)
    if mol is not None:
        fragments = list(BRICSDecompose(mol))
        processed_fragments = []

        for frag in fragments:
            processed_smiles = re.sub(r'\[\d+\*]', '[H]', frag)
            processed_mol = Chem.MolFromSmiles(processed_smiles)
            if processed_mol is not None:
                processed_mol = Chem.RemoveHs(processed_mol)
                Chem.SanitizeMol(processed_mol)
                final_smiles = Chem.MolToSmiles(processed_mol)
                processed_fragments.append(final_smiles)
            else:
                processed_fragments.append(processed_smiles)

        return '.'.join(processed_fragments)
    else:
        return ''


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
    if retriever:
        with multiprocessing.Pool(processes=processes) as pool:
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
    with multiprocessing.Pool(processes=processes) as pool:
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

    print("Avg. edit distance:", np.mean(edit_distances))
    if retriever:
        print("Avg. src2frag edit distance:", np.mean(src2frag_edit_distances))
        print("Avg. tgt2frag edit distance:", np.mean(tgt2frag_edit_distances))
    print('size', len(src_data))
    if augmentation > 0:
        with open(os.path.join(save_dir, 'src-{}.txt'.format(set_name)), 'w') as f:
            for src in src_data:
                f.write('{}\n'.format(src))
        with open(os.path.join(save_dir, 'tgt-{}.txt'.format(set_name)), 'w') as f:
            for tgt in tgt_data:
                f.write('{}\n'.format(tgt))
        if retriever:
            with open(os.path.join(save_dir, 'ref-{}.txt'.format(set_name)), 'w') as f:
                for ref in ref_data:
                    f.write('{}\n'.format(ref))
            with open(os.path.join(save_dir, 'sim-{}.txt'.format(set_name)), 'w') as f:
                for sim in sim_data:
                    f.write('{}\n'.format(sim))
            with open(os.path.join(save_dir, 'frag-{}.txt'.format(set_name)), 'w') as f:
                for frag in frag_data:
                    f.write('{}\n'.format(frag))
    return src_data, tgt_data, ref_data, sim_data, frag_data


def multi_process(data):
    data, retriever = data
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
    pro_atom_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", product)))  # 记录所有映射编号 用来选根 计算次数
    reactant = reactant.split(".")
    if data['root_aligned']:
        reversable = False
        if augmentation == 999:
            product_roots = pro_atom_map_numbers
            times = len(product_roots)
        else:
            product_roots = [-1]
            max_times = len(pro_atom_map_numbers)
            times = min(augmentation, max_times)
            if times < augmentation:  # times = max_times
                product_roots.extend(pro_atom_map_numbers)
                product_roots.extend(random.choices(product_roots, k=augmentation - len(product_roots)))
            else:  # times = augmentation
                while len(product_roots) < times:
                    product_roots.append(random.sample(pro_atom_map_numbers, 1)[0])
                    if product_roots[-1] in product_roots[:-1]:
                        product_roots.pop()
            times = len(product_roots)
            assert times == augmentation
            if reversable:
                times = int(times / 2)
        for k in range(times):
            pro_root_atom_map = product_roots[k]
            pro_root = get_root_id(pro_mol, root_map_number=pro_root_atom_map)
            cano_atom_map = get_cano_map_number(product, root=pro_root)
            if cano_atom_map is None:
                return_status["status"] = "error_mapping"
                return return_status
            pro_smi = clear_map_canonical_smiles(product, canonical=True, root=pro_root)
            aligned_reactants = []
            aligned_reactants_order = []
            rea_atom_map_numbers = [list(map(int, re.findall(r"(?<=:)\d+", rea))) for rea in reactant]
            for i, rea_map_number in enumerate(rea_atom_map_numbers):
                for j, map_number in enumerate(cano_atom_map):
                    # select mapping reactans
                    if map_number in rea_map_number:
                        rea_root = get_root_id(Chem.MolFromSmiles(reactant[i]), root_map_number=map_number)
                        rea_smi = clear_map_canonical_smiles(reactant[i], canonical=True, root=rea_root)
                        aligned_reactants.append(rea_smi)
                        aligned_reactants_order.append(j)
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
                frag_tokens = smi_tokenizer(fragment_smi)
                return_status['frag_data'].append(frag_tokens)
                ref_tokens = smi_tokenizer(ref_cano_reactant)
                return_status['ref_data'].append(ref_tokens)
                return_status['sim_data'].append(similarity)
            sorted_reactants = sorted(list(zip(aligned_reactants, aligned_reactants_order)), key=lambda x: x[1])
            aligned_reactants = [item[0] for item in sorted_reactants]
            reactant_smi = ".".join(aligned_reactants)
            product_tokens = smi_tokenizer(pro_smi)
            reactant_tokens = smi_tokenizer(reactant_smi)
            return_status['src_data'].append(product_tokens)
            return_status['tgt_data'].append(reactant_tokens)
            if reversable:
                aligned_reactants.reverse()
                reactant_smi = ".".join(aligned_reactants)
                product_tokens = smi_tokenizer(pro_smi)
                reactant_tokens = smi_tokenizer(reactant_smi)
                return_status['src_data'].append(product_tokens)
                return_status['tgt_data'].append(reactant_tokens)
                if retriever:
                    ref_tokens = smi_tokenizer(ref_cano_reactant)
                    frag_tokens = smi_tokenizer(fragment)
                    return_status['ref_data'].append(ref_tokens)
                    return_status['sim_data'].append(similarity)
                    return_status['frag_data'].append(frag_tokens)
        assert len(return_status['src_data']) == data['augmentation']
    else:
        cano_product = clear_map_canonical_smiles(product)
        cano_reactant = ".".join([clear_map_canonical_smiles(rea) for rea in reactant if
                                  len(set(map(int, re.findall(r"(?<=:)\d+", rea))) & set(
                                      pro_atom_map_numbers)) > 0])

        return_status['src_data'].append(smi_tokenizer(cano_product))
        return_status['tgt_data'].append(smi_tokenizer(cano_reactant))
        if retriever:
            fragment = ".".join([clear_map_canonical_smiles(frag) for frag in fragment if
                                 len(set(map(int, re.findall(r"(?<=:)\d+", frag))) & set(
                                     pro_atom_map_numbers)) > 0])
            return_status['ref_data'].append(smi_tokenizer(ref_cano_reactant))
            return_status['sim_data'].append(similarity)
            return_status['frag_data'].append(smi_tokenizer(fragment))
            frag_mols = [Chem.MolFromSmiles(frag) for frag in fragment.split(".")]
        pro_mol = Chem.MolFromSmiles(cano_product)
        rea_mols = [Chem.MolFromSmiles(rea) for rea in cano_reactant.split(".")]

        for i in range(int(augmentation - 1)):
            pro_smi = Chem.MolToSmiles(pro_mol, doRandom=True)
            rea_smi = [Chem.MolToSmiles(rea_mol, doRandom=True) for rea_mol in rea_mols]
            rea_smi = ".".join(rea_smi)
            return_status['src_data'].append(smi_tokenizer(pro_smi))
            return_status['tgt_data'].append(smi_tokenizer(rea_smi))
            if retriever:
                frag_smi = [Chem.MolToSmiles(frag_mol, doRandom=True) for frag_mol in frag_mols]
                frag_smi = ".".join(frag_smi)
                return_status['ref_data'].append(smi_tokenizer(ref_cano_reactant))
                return_status['sim_data'].append(similarity)
                return_status['frag_data'].append(smi_tokenizer(frag_smi))
    edit_distances = []

    for src, tgt in zip(return_status['src_data'], return_status['tgt_data']):
        edit_distances.append(textdistance.levenshtein.distance(src.split(), tgt.split()))

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
    parser.add_argument("-processes", type=int, default=-1)
    parser.add_argument("-test_only", action="store_true")
    parser.add_argument("-train_only", action="store_true")
    parser.add_argument("-test_except", action="store_true")
    parser.add_argument("-validastrain", action="store_true")
    parser.add_argument("-character", action="store_true")
    parser.add_argument("-canonical", action="store_true")
    parser.add_argument("-postfix", type=str, default="")
    parser.add_argument("-method", type=str, default="")
    parser.add_argument("-retriever", action="store_true")
    args = parser.parse_args()

    print(args)
    if args.test_only:
        datasets = ['test']
    elif args.train_only:
        datasets = ['train']
    elif args.test_except:
        datasets = ['val', 'train']
    elif args.validastrain:
        datasets = ['test', 'val', 'train']
    else:
        datasets = ['test', 'val', 'train']
    random.seed(args.seed)
    datadir = '../datasets/{}'.format(args.dataset)
    savedir = '../datasets/{}/aug{}/{}'.format(args.dataset, args.augmentation, args.method)
    savedir += args.postfix
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    for i, data_set in enumerate(datasets):
        if args.retriever:
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

        save_dir = os.path.join(savedir, data_set)

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
