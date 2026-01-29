import multiprocessing
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from rdkit import Chem, rdBase

rdBase.DisableLog('rdApp.warning')


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


def check_reaction_quality(reactant, product):
    return_status = {"status": 0}
    pt = re.compile(r':(\d+)]')
    rids = sorted(re.findall(pt, reactant))
    pids = sorted(re.findall(pt, product))
    # 解析 SMILES
    pro_mol = Chem.MolFromSmiles(product)
    rea_mol = Chem.MolFromSmiles(reactant)
    # 如果去重后长度不等于原长度，说明有重复 mapping is not 1:1
    if len(set(rids)) != len(rids):  # duplicate atom mapping
        return_status["status"] = "error_mapping"
    if len(set(pids)) != len(pids):  # duplicate atom mapping
        return_status["status"] = "error_mapping"
    if "" == product:
        return_status["status"] = "empty_p"
    if "" == reactant:
        return_status["status"] = "empty_r"
    if rea_mol is None:
        return_status["status"] = "invalid_r"
    if len(rea_mol.GetAtoms()) < 5:  # 总反应物
        return_status["status"] = "small_r"
    if pro_mol is None:
        return_status["status"] = "invalid_p"
    if len(pro_mol.GetAtoms()) == 1:
        return_status["status"] = "small_p"
    if not all([a.HasProp('molAtomMapNumber') for a in pro_mol.GetAtoms()]):
        return_status["status"] = "error_mapping_p"
    """finishing checking data quality"""

    return return_status["status"]


def cano_rea_product(data):
    product = data['product']
    reactant = data['reactant']
    reactant_original = reactant
    # 先检查然后标准化
    if check_reaction_quality(reactant, product) != 0:
        return None
    cano_product = clear_map_canonical_smiles(product)
    pro_atom_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", product)))
    reactant = reactant.split(".")
    cano_reactant = ".".join([
        clear_map_canonical_smiles(rea) for rea in reactant if len(
            set(map(int, re.findall(r"(?<=:)\d+", rea)))
            & set(pro_atom_map_numbers)) > 0
    ])
    return cano_product, cano_reactant, product, reactant_original


def build_rag_db(data_50k, data_mit, data_full):
    # 转换为 DataFrame
    df_50k = pd.DataFrame(data_50k, columns=['cano_product', 'cano_reactant', 'product', 'reactant'])
    df_mit = pd.DataFrame(data_mit, columns=['cano_product', 'cano_reactant', 'product', 'reactant'])
    df_full = pd.DataFrame(data_full, columns=['cano_product', 'cano_reactant', 'product', 'reactant'])

    common_full = pd.merge(
        df_mit, df_full, how='inner', on=['cano_product', 'cano_reactant'], suffixes=('', '_full')
    )
    common_50k = pd.merge(
        df_mit, df_50k, how='inner', on=['cano_product', 'cano_reactant'], suffixes=('', '_50k')
    )
    unique_common_full = common_full[['cano_product', 'cano_reactant']].drop_duplicates()
    unique_common_50k = common_50k[['cano_product', 'cano_reactant']].drop_duplicates()
    intersection = pd.merge(unique_common_full, unique_common_50k, on=['cano_product', 'cano_reactant'])

    all_duplicates = pd.concat([
        common_full[['cano_product', 'cano_reactant']],
        common_50k[['cano_product', 'cano_reactant']]
    ]).drop_duplicates()

    filtered_mit = df_mit.merge(all_duplicates, on=['cano_product', 'cano_reactant'],
                                how='left', indicator=True)
    filtered_mit = filtered_mit[filtered_mit['_merge'] == 'left_only'].drop('_merge', axis=1)
    filtered_mit = filtered_mit.drop_duplicates(subset=['cano_product', 'cano_reactant'])

    filtered_mit = filtered_mit.sort_values(
        by=['cano_product', 'cano_reactant']
    ).reset_index(drop=True)
    print("\n=== 比较完整的反应（反应物+产物）===")
    print(f"USPTO_50K_cleaned 反应数量: {len(data_50k)}")
    print(f"USPTO_MIT_cleaned 反应数量: {len(data_mit)}")
    print(f"USPTO_FULL_cleaned 反应数量: {len(data_full)}")
    print(f"与 FULL_cleaned 重复的反应数量: {len(common_full)}")
    print(f"与 50K_cleaned 重复的反应数量: {len(common_50k)}")
    print("common_full 中有重复反应对数量：", common_full.duplicated(subset=['cano_product', 'cano_reactant']).sum())
    print("common_50k 中有重复反应对数量：", common_50k.duplicated(subset=['cano_product', 'cano_reactant']).sum())
    print(f"common_full 唯一反应数: {len(unique_common_full)}")
    print(f"common_50k 唯一反应数: {len(unique_common_50k)}")
    print(f"unique_common_full 与 unique_common_50k 的交集重复反应数量: {len(intersection)}")  # 应该是 185451
    print(f"合并后唯一重复反应总数: {len(all_duplicates)}")
    print(f"去重后 USPTO_MIT_cleaned 反应数量: {len(filtered_mit)}")

    filtered_mit.to_csv('../datasets/USPTO_MIT/RAG.csv', index=False)
    print(f"去重后的数据已保存到 ../datasets/USPTO_MIT/RAG.csv，共 {len(filtered_mit)} 条记录")


def get_data(reaction_list, dataset="USPTO_50K"):
    if dataset == "USPTO_MIT":
        reactant_smarts_list = list(
            map(lambda x: x.split('>>')[0], reaction_list))
        product_smarts_list = list(
            map(lambda x: x.split('>>')[1], reaction_list))
        product_smarts_list = list(
            map(lambda x: x.split(' ')[0], product_smarts_list))
    else:
        reactant_smarts_list = list(
            map(lambda x: x.split('>')[0], reaction_list))
        reactant_smarts_list = list(
            map(lambda x: x.split(' ')[0], reactant_smarts_list))  # remove ' |f:1...'
        product_smarts_list = list(
            map(lambda x: x.split('>')[2], reaction_list))
        product_smarts_list = list(
            map(lambda x: x.split(' ')[0], product_smarts_list))  # remove ' |f:1...'
    print(f"Total {dataset} Size", len(reaction_list))
    multiple_product_indices = [
        i for i in range(len(product_smarts_list)) if "." in product_smarts_list[i]]
    for index in multiple_product_indices:
        products = product_smarts_list[index].split(".")
        for product in products:
            reactant_smarts_list.append(reactant_smarts_list[index])
            product_smarts_list.append(product)
    for index in multiple_product_indices[::-1]:
        del reactant_smarts_list[index]
        del product_smarts_list[index]
    data = [{
        "reactant": i,
        "product": j,
    } for i, j in zip(reactant_smarts_list, product_smarts_list)]

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results = list(tqdm(pool.imap(cano_rea_product, data), total=len(data), desc="Processing split reactions"))
    results = set(list(filter(None, results)))
    sorted_results = sorted(list(results), key=lambda x: (x[0], x[1]))
    print(f"Cleand {dataset} Size", len(sorted_results))
    return sorted_results


def process_uspto_dataset(dataset_path, dataset_name):
    reaction_list, processed_data = [], []

    for filename in ['raw_test.csv', 'raw_train.csv', 'raw_val.csv']:
        csv = pd.read_csv(dataset_path / 'raw' / filename)
        current_reactions = list(csv["reactants>reagents>production"])

        result = get_data(current_reactions, dataset_name + "_" + filename.split('.')[0])
        save_path = dataset_path / 'cleaned' / filename.replace('raw_', 'cleaned_')
        save_path.parent.mkdir(parents=True, exist_ok=True)

        df_result = pd.DataFrame(result, columns=['cano_product', 'cano_reactant', 'product', 'reactant'])
        df_result.to_csv(save_path, index=False)
        processed_data.extend(list(result))

    return set(processed_data)


if __name__ == '__main__':
    uspto_50k_path = Path('../datasets/USPTO_50K')
    uspto_mit_path = Path('../datasets/USPTO_MIT')
    uspto_full_path = Path('../datasets/USPTO_FULL')

    reaction_list_mit = []
    for filename in ['test.txt', 'train.txt', 'valid.txt']:
        with open(uspto_mit_path / 'raw' / filename, 'r') as file:
            for line in file:
                reaction_list_mit.append(line)
    data_mit = get_data(reaction_list_mit, "USPTO_MIT")
    data_50k = process_uspto_dataset(uspto_50k_path, "USPTO_50K")
    data_full = process_uspto_dataset(uspto_full_path, "USPTO_FULL")
    build_rag_db(data_50k, data_mit, data_full)  # 生成 RAG 数据库
