from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from tqdm import tqdm


class FingerprintGenerator:
    @staticmethod
    def get_fingerprint_function(fp_type):
        fp_functions = {
            'maccs': lambda mol: MACCSkeys.GenMACCSKeys(mol),
            'morgan': lambda mol: GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol),
            'rdk': lambda mol: AllChem.RDKFingerprint(mol, maxPath=5),
        }
        return fp_functions.get(fp_type)


def process_smiles(args):
    idx, smiles, fp_type = args
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp_func = FingerprintGenerator.get_fingerprint_function(fp_type)
        if fp_func:
            return {
                'smiles': smiles,
                'fingerprint': fp_func(mol),
                'original_index': idx
            }
    return None


def normalize_reactant(smi: str) -> str:
    parts = [p.strip() for p in smi.split('.') if p.strip()]
    parts.sort()
    return '.'.join(parts)


def calculate_similarity_batch(args):
    query_batch, rag_data, rag_reactants, query_reactants, is_test = args  # 增加 is_test 标志
    results = []

    rag_fps = [entry['fingerprint'] for entry in rag_data]
    rag_indices = [entry['original_index'] for entry in rag_data]
    rag_products = [entry['smiles'] for entry in rag_data]

    for query in query_batch:
        if query is None:
            results.append((0.0, -1))
            continue

        query_fp = query['fingerprint']
        query_idx = query['original_index']
        query_prod = query['smiles']
        query_react = query_reactants[query_idx]

        similarities = np.array(DataStructs.BulkTanimotoSimilarity(query_fp, rag_fps))
        sorted_idx = np.argsort(similarities)[::-1]

        best_idx = sorted_idx[0]
        best_sim = similarities[best_idx]

        if is_test:
            for idx in sorted_idx:
                rag_prod = rag_products[idx]
                if query_prod == rag_prod:
                    query_react_norm = normalize_reactant(query_react)
                    rag_react_norm = normalize_reactant(rag_reactants[idx])
                    if query_react_norm == rag_react_norm:
                        continue
                best_idx = idx
                best_sim = similarities[idx]
                break

        results.append((query_idx, (best_sim, rag_indices[best_idx])))
    return results


def add_fingerprint_similarity(dataset_path, rag_path, fp_type):
    fp_dir = dataset_path / 'processed' / f'{fp_type}'
    fp_dir.mkdir(parents=True, exist_ok=True)

    print(f"读取并处理RAG数据 ({fp_type})...")
    rag_df = pd.read_csv(rag_path)

    with Pool() as pool:
        rag_inputs = [(idx, smiles, fp_type) for idx, smiles in enumerate(rag_df['cano_product'])]
        rag_results = []
        for result in tqdm(
                pool.imap_unordered(process_smiles, rag_inputs),
                total=len(rag_df),
                desc="处理RAG分子"
        ):
            if result is not None:
                rag_results.append(result)

        print(f"有效RAG分子: {len(rag_results)}/{len(rag_df)}")
        success_rag_indices = [r['original_index'] for r in rag_results]
        rag_reactants = [rag_df.loc[i, 'cano_reactant'] for i in success_rag_indices]
        for filename in ['test.csv', 'train.csv', 'val.csv']:
            print(f"\n处理 {filename}...")
            filepath = dataset_path / 'cleaned' / f"cleaned_{filename}"
            df = pd.read_csv(filepath)

            query_inputs = [(idx, smiles, fp_type) for idx, smiles in enumerate(df['cano_product'])]
            query_results = list(tqdm(
                pool.imap_unordered(process_smiles, query_inputs),
                total=len(df),
                desc="处理查询分子"
            ))
            # batch_size = 1500 if filename == 'train.csv' else 300  # -50k
            batch_size = 5000 if filename == 'train.csv' else 2000  # -full
            is_test = filename.startswith('test')
            query_reactants = df['cano_reactant'].astype(str).tolist()  # 记录query反应物
            similarity_batches = [
                (query_results[i:i + batch_size], rag_results, rag_reactants, query_reactants, is_test)
                for i in range(0, len(query_results), batch_size)
            ]

            sim_results = {}
            for batch_results in tqdm(
                    pool.imap_unordered(calculate_similarity_batch, similarity_batches),
                    total=len(similarity_batches),
                    desc="计算相似度"
            ):
                for query_idx, (sim, rag_idx) in batch_results:
                    if query_idx not in sim_results or sim_results[query_idx][0] < sim:
                        sim_results[query_idx] = (sim, rag_idx)

            result_data = []
            for i in range(len(df)):
                if i in sim_results:
                    sim, rag_idx = sim_results[i]
                    rag_data = rag_df.iloc[rag_idx].to_dict()
                    rag_data['similarity'] = sim
                    rag_data['rag_index'] = rag_idx
                else:
                    rag_data = {'similarity': 0.0, 'rag_index': -1}
                result_data.append(rag_data)
            # 创建结果DataFrame
            result_df = pd.DataFrame(result_data)
            result_df.columns = [
                f'sim_{col}' if col not in ['similarity', 'rag_index']
                else col for col in result_df.columns
            ]

            final_df = pd.concat([df, result_df], axis=1)
            save_path = fp_dir / f"processed_{filename}"
            final_df.to_csv(save_path, index=False)

            stats = result_df['similarity'].agg(['mean', 'max', 'min'])
            print(f"{fp_type}相似度统计: ")
            print(f"  平均值: {stats['mean']:.4f}")
            print(f"  最大值: {stats['max']:.4f}")
            print(f"  最小值: {stats['min']:.4f}")


if __name__ == '__main__':
    uspto_50k_path = Path('../datasets/USPTO_50K')
    uspto_full_path = Path('../datasets/USPTO_FULL')
    rag_path = Path('../datasets/USPTO_MIT/RAG.csv')

    fingerprint_types = ['morgan']  # 'morgan', 'maccs', 'rdk'
    for fingerprint_type in fingerprint_types:
        print(f"\n处理指纹类型: {fingerprint_type}")
        add_fingerprint_similarity(uspto_50k_path, rag_path, fingerprint_type)
        add_fingerprint_similarity(uspto_full_path, rag_path, fingerprint_type)
