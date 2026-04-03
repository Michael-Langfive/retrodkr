import time
from multiprocessing import Pool
from pathlib import Path

import faiss
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

    @staticmethod
    def get_fp_size(fp_type):
        sizes = {
            'maccs': 167,
            'morgan': 2048,
            'rdk': 2048,
        }
        return sizes.get(fp_type, 2048)


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


def fp_to_numpy(fp, fp_size):
    arr = np.zeros(fp_size, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def normalize_reactant(smi: str) -> str:
    parts = [p.strip() for p in smi.split('.') if p.strip()]
    parts.sort()
    return '.'.join(parts)


def calculate_similarity_batch(args):
    query_batch, rag_data, rag_reactants, query_reactants, is_test = args
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


def build_faiss_index(rag_results, fp_type, index_type='flat'):
    """构建 FAISS binary 索引"""
    fp_size = FingerprintGenerator.get_fp_size(fp_type)
    # 需要对齐到8的倍数
    d = ((fp_size + 7) // 8) * 8

    binary_vecs = []
    for entry in rag_results:
        arr = fp_to_numpy(entry['fingerprint'], fp_size)
        if fp_size < d:
            arr = np.pad(arr, (0, d - fp_size), 'constant')
        packed = np.packbits(arr)
        binary_vecs.append(packed)

    binary_vecs = np.array(binary_vecs, dtype=np.uint8)

    if index_type == 'flat':
        index = faiss.IndexBinaryFlat(d)
        index.add(binary_vecs)
    elif index_type == 'ivf':
        nlist = min(256, len(binary_vecs) // 10)
        quantizer = faiss.IndexBinaryFlat(d)
        index = faiss.IndexBinaryIVF(quantizer, d, nlist)
        index.train(binary_vecs)
        index.add(binary_vecs)
        index.nprobe = 16
    else:
        raise ValueError(f"Unknown index_type: {index_type}")

    return index, binary_vecs, d


def faiss_search(index, query_results, fp_type, d, rag_results, rag_df,
                 rag_reactants, query_reactants, is_test, k=10):
    """使用 FAISS 检索，然后用 Tanimoto 重排"""
    fp_size = FingerprintGenerator.get_fp_size(fp_type)

    # 构建查询向量
    query_binary = []
    query_indices = []
    for entry in query_results:
        if entry is None:
            continue
        arr = fp_to_numpy(entry['fingerprint'], fp_size)
        if fp_size < d:
            arr = np.pad(arr, (0, d - fp_size), 'constant')
        packed = np.packbits(arr)
        query_binary.append(packed)
        query_indices.append(entry['original_index'])

    if not query_binary:
        return {}

    query_binary = np.array(query_binary, dtype=np.uint8)
    # ================= 性能测试计时开始 =================
    # print(f"\n[性能测试] 开始进行 FAISS 批量检索 (k={k}, queries={len(query_binary)})...")
    # start_time = time.time()

    # FAISS 批量检索 top-k 候选
    D, I = index.search(query_binary, k)

    # end_time = time.time()

    # total_search_time = end_time - start_time
    # latency_per_query = (total_search_time / len(query_binary)) * 1000  # 换算成毫秒
    #
    # print(f"[性能测试] FAISS 批量检索总耗时: {total_search_time:.4f} 秒")
    # print(f"[性能测试] *** FAISS 平均单句检索延迟: {latency_per_query:.4f} ms/query ***\n")
    rag_fps = [entry['fingerprint'] for entry in rag_results]
    rag_original_indices = [entry['original_index'] for entry in rag_results]
    rag_products = [entry['smiles'] for entry in rag_results]

    sim_results = {}

    for qi in range(len(query_indices)):
        query_idx = query_indices[qi]
        query_entry = None
        for entry in query_results:
            if entry is not None and entry['original_index'] == query_idx:
                query_entry = entry
                break

        if query_entry is None:
            continue

        query_fp = query_entry['fingerprint']
        query_prod = query_entry['smiles']
        query_react = query_reactants[query_idx]

        # 对 FAISS 返回的 top-k 候选用 Tanimoto 精确重排
        candidates = I[qi]
        candidate_sims = []
        for cand_idx in candidates:
            if cand_idx < 0 or cand_idx >= len(rag_fps):
                candidate_sims.append(-1.0)
                continue
            sim = DataStructs.TanimotoSimilarity(query_fp, rag_fps[cand_idx])
            candidate_sims.append(sim)

        # 按 Tanimoto 相似度降序排列
        sorted_pairs = sorted(zip(candidates, candidate_sims), key=lambda x: -x[1])

        best_rag_idx = -1
        best_sim = 0.0

        for cand_idx, sim in sorted_pairs:
            if cand_idx < 0:
                continue

            if is_test:
                rag_prod = rag_products[cand_idx]
                if query_prod == rag_prod:
                    query_react_norm = normalize_reactant(query_react)
                    rag_react_norm = normalize_reactant(rag_reactants[cand_idx])
                    if query_react_norm == rag_react_norm:
                        continue

            best_rag_idx = cand_idx
            best_sim = sim
            break

        if best_rag_idx >= 0:
            orig_rag_idx = rag_original_indices[best_rag_idx]
            sim_results[query_idx] = (best_sim, orig_rag_idx)

    return sim_results


def add_fingerprint_similarity(dataset_path, rag_path, fp_type, retrieval_method='brute_force'):
    """
    为数据集添加指纹相似度信息
    retrieval_method: 'brute_force', 'faiss_flat', 'faiss_ivf'
    """
    fp_dir = dataset_path / 'processed' / f'{fp_type}'
    fp_dir.mkdir(parents=True, exist_ok=True)

    print(f"读取并处理RAG数据 ({fp_type}, method={retrieval_method})...")
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

        # 构建 FAISS 索引（如果需要）
        faiss_index = None
        faiss_binary = None
        faiss_d = None
        if retrieval_method in ['faiss_flat', 'faiss_ivf']:
            index_type = 'flat' if retrieval_method == 'faiss_flat' else 'ivf'
            print(f"构建 FAISS {index_type} 索引...")
            faiss_index, faiss_binary, faiss_d = build_faiss_index(
                rag_results, fp_type, index_type=index_type
            )
            print(f"FAISS 索引构建完成, d={faiss_d}")

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

            is_test = filename.startswith('test')
            query_reactants = df['cano_reactant'].astype(str).tolist()

            if retrieval_method == 'brute_force':
                # 原有的暴力搜索逻辑
                batch_size = 1500 if filename == 'train.csv' else 300
                # batch_size = 5000 if filename == 'train.csv' else 2000  # -full
                similarity_batches = [
                    (query_results[i:i + batch_size], rag_results, rag_reactants, query_reactants, is_test)
                    for i in range(0, len(query_results), batch_size)
                ]

                sim_results = {}
                for batch_results in tqdm(
                        pool.imap_unordered(calculate_similarity_batch, similarity_batches),
                        total=len(similarity_batches),
                        desc="计算相似度 (brute-force)"
                ):
                    for query_idx, (sim, rag_idx) in batch_results:
                        if query_idx not in sim_results or sim_results[query_idx][0] < sim:
                            sim_results[query_idx] = (sim, rag_idx)

            else:
                # FAISS 检索 + Tanimoto 重排
                print(f"使用 FAISS ({retrieval_method}) 检索...")
                sim_results = faiss_search(
                    faiss_index, query_results, fp_type, faiss_d,
                    rag_results, rag_df, rag_reactants, query_reactants,
                    is_test, k=50
                )

            # 处理结果
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

            result_df = pd.DataFrame(result_data)
            result_df.columns = [
                f'sim_{col}' if col not in ['similarity', 'rag_index']
                else col for col in result_df.columns
            ]

            final_df = pd.concat([df, result_df], axis=1)
            save_path = fp_dir / f"processed_{filename}"
            final_df.to_csv(save_path, index=False)

            stats = result_df['similarity'].agg(['mean', 'max', 'min'])
            print(f"{fp_type}相似度统计 ({retrieval_method}): ")
            print(f"  平均值: {stats['mean']:.4f}")
            print(f"  最大值: {stats['max']:.4f}")
            print(f"  最小值: {stats['min']:.4f}")


if __name__ == '__main__':
    uspto_50k_path = Path('../datasets/USPTO_50K')
    uspto_full_path = Path('../datasets/USPTO_FULL')
    rag_path = Path('../datasets/USPTO_MIT/RAG.csv')

    fp_type = 'morgan'

    # 选择检索方式: 'brute_force', 'faiss_flat', 'faiss_ivf'
    retrieval_method = 'brute_force'

    print(f"\n处理指纹类型: {fp_type}, 检索方式: {retrieval_method}")
    add_fingerprint_similarity(uspto_50k_path, rag_path, fp_type, retrieval_method)
    add_fingerprint_similarity(uspto_full_path, rag_path, fp_type, retrieval_method)
