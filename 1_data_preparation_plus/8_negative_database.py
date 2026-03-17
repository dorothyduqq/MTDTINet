# 选择相似性分数最低的
import pandas as pd
from tqdm import tqdm

# 读取原始数据文件
input_file = './7_get_neg_datas/neg_pre_data.csv'
df = pd.read_csv(input_file)

pos_sample = pd.read_csv('./6_positive_database/positive_database.csv')
total_pos_count = pos_sample.shape[0]
print(f'positive sample: {total_pos_count}')

# 创建一个列表来保存新的数据
new_data = []

compound_counts = dict()
max_neg_compound = 5

# 遍历每一行
for _, row in tqdm(df.iterrows(), total=len(df)):
    pos_count = int(row['pos_count'])  # 获取pos_count
    compound_ids = row['compound_id'].split(';')  # 获取compound_id并分割
    tanimoto_scores = row['max_tanimoto_score'].split(';')  # 获取对应的max_tanimoto_score并分割

    # 将compound_id和tanimoto_score成对打包为元组列表，并按tanimoto_score升序排序
    compound_score_pairs = list(zip(compound_ids, map(float, tanimoto_scores)))
    compound_score_pairs.sort(key=lambda x: x[1])  # 根据tanimoto_score排序

    # 选择得分最低的pos_count个compound_id
    num = -1
    selected_pairs = []
    while len(selected_pairs) < pos_count and len(compound_score_pairs) > num + 1:
        num += 1
        the_compound = compound_score_pairs[num][0]
        if the_compound not in compound_counts:
            compound_counts[the_compound] = 1
            selected_pairs.append(compound_score_pairs[num])
        if the_compound in compound_counts and compound_counts[the_compound] < max_neg_compound:
            compound_counts[the_compound] += 1
            selected_pairs.append(compound_score_pairs[num])
        elif the_compound in compound_counts and compound_counts[the_compound] >= max_neg_compound:
            continue

    # 把数据加入到new_data列表中
    for compound_id, tanimoto_score in selected_pairs:
        new_data.append({
            'target_protein_id': row['target_protein_id'],
            'compound_id': compound_id,
            'tanimoto_score': tanimoto_score, 
            'pchembl': 0
        })

print(f'balanced neg sample: {len(new_data)}')

# 转换为DataFrame
new_df = pd.DataFrame(new_data)

# extra_counts = 0
# # 如果new_df的行数不足total_pos_count，从df中随机选择不重复的行补足
# while len(new_df) < total_pos_count: 
#     extra_counts += 1
#     print(extra_counts)
#     # 随机选择一行数据
#     random_row = df.sample(n=1).iloc[0]
#     target_protein_id = random_row['target_protein_id']
#     compound_ids = random_row['compound_id'].split(';')
#     tanimoto_scores = random_row['max_tanimoto_score'].split(';')  # 获取对应的max_tanimoto_score并分割

#     compound_score_pairs = list(zip(compound_ids, map(float, tanimoto_scores)))
#     compound_score_pairs.sort(key=lambda x: x[1])  # 根据 tanimoto_score 升序排序

#     # 获取 new_df 中已有的相同 target_protein_id 的 compound_id 集合
#     existing_ids = set(new_df[new_df['target_protein_id'] == target_protein_id]['compound_id'])
    
#     # 选择与已有的 compound_id 不重复的 ID
#     unused_ids = set(compound_ids) - existing_ids
    
#     if unused_ids:
#         # 筛选出 compound_score_pairs 中在 unused_ids 里的项
#         valid_pairs = [pair for pair in compound_score_pairs if pair[0] in unused_ids]
#         if valid_pairs:
#             new_df = pd.concat([new_df, pd.DataFrame([{
#                 'target_protein_id': target_protein_id,
#                 'compound_id': valid_pairs[0][0],
#                 'max_tanimoto_score': valid_pairs[0][1],
#                 'pchembl': 0
#             }])], ignore_index=True)

# 保存为新的CSV文件
output_file = './8_negative_database/negative_database.csv'
new_df.to_csv(output_file, index=False)
print(f'negative sample: {new_df.shape[0]}')
