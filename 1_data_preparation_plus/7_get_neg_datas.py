import gc
import pandas as pd
from tqdm import tqdm
from rdkit.DataStructs import TanimotoSimilarity
from rdkit import DataStructs
from multiprocessing import Pool

# 读取数据
pos_data = pd.read_csv('./6_positive_database/positive_database.csv')
all_drug_data = pd.read_pickle('./5_normalize_and_pad_encodings/normalized_and_no_padded_drug_encodings.pkl')
all_drug_data = all_drug_data[['compound_id', 'fingerprint']]

# 创建 drug_data 字典，键为 compound_id，值为 fingerprint
drug_data = {}
for idx, row in all_drug_data.iterrows():
    fingerprint = ''.join(map(str, row['fingerprint']))
    fingerprint = DataStructs.CreateFromBitString(fingerprint)
    drug_data[row['compound_id']] = fingerprint
del all_drug_data
gc.collect()
drug_keys = set(drug_data.keys())

# 创建 pos_dict 字典，键为 target_protein_id 的不同值，值为对应的 compound_id 列表
pos_dict = {}
for target_protein_id, group in pos_data.groupby('target_protein_id'):
    pos_dict[target_protein_id] = list(group['compound_id'].unique())
del pos_data
gc.collect()
# 准备结果记录
result_data = []

# 定义计算函数
def check_compound_similarity(compound_id):
    fingerprint = drug_data[compound_id]
    max_score = 0
    for cmp_id in compound_id_list:
        tanimoto_similarity = TanimotoSimilarity(fingerprint, drug_data[cmp_id])
        if tanimoto_similarity > max_score:
            max_score = tanimoto_similarity
    return [compound_id, max_score]  # 如果都低于阈值，返回 compound_id 表示符合条件

# 遍历 pos_dict 字典中的每个 target_protein_id
for target_protein_id, compound_id_list in tqdm(pos_dict.items(), desc='Processing...'):
    compound_id_set = set(compound_id_list)
    inputs = list(drug_keys - compound_id_set)

    # 用于存储符合条件的 compound_id
    neg_infos = []

    # 使用多进程池并行计算
    with Pool(96) as pool:
        neg_infos = list(pool.imap(check_compound_similarity, inputs)) # tqdm(, total=len(inputs))

    neg_compounds = [neg_compound[0] for neg_compound in neg_infos]
    neg_max_scores = [str(neg_max_score[1]) for neg_max_score in neg_infos]

    # 保存结果
    result_data.append([
        target_protein_id,
        ';'.join(neg_compounds),
        ';'.join(neg_max_scores),
        len(neg_infos),
        len(compound_id_list)
    ])

# 将结果保存为 CSV 文件
result_df = pd.DataFrame(result_data, columns=['target_protein_id', 'compound_id', 'max_tanimoto_score', 'all_neg_count', 'pos_count'])
result_df.to_csv('./7_get_neg_datas/neg_pre_data.csv', index=False)

