import pandas as pd
import os
from tqdm import tqdm

RANDOM_SEED = 1234

# 读取正负样本并添加标签列
pos_sample = pd.read_csv('./6_positive_database/positive_database.csv')
pos_sample['label'] = 1  # 正样本标签为 1
print(f'positive sample: {pos_sample.shape}')

neg_sample = pd.read_csv('./8_negative_database/negative_database.csv')[['compound_id', 'target_protein_id', 'pchembl']]
neg_sample['label'] = 0  # 负样本标签为 0
print(f'negative sample: {neg_sample.shape}')

# 定义每种靶点的正负样本数量限制
MAX_SAMPLES_PER_TARGET = 100  # 每种靶点测试集正负样本限制

# 合并所有样本，便于后续处理
all_samples = pd.concat([pos_sample, neg_sample], ignore_index=True)

# 初始化测试集和训练集
test_data = pd.DataFrame(columns=['compound_id', 'target_protein_id', 'pchembl', 'label'])
train_data = all_samples.copy()
test_compounds = set()  # 用于记录测试集中涉及的化合物

# 计算每个 group 的大小，并按从小到大排序
group_sizes = all_samples.groupby('target_protein_id').size().sort_values()

# 按照排序后的顺序进行遍历
for target_id in tqdm(group_sizes.index, total=len(group_sizes)):
    group = all_samples[all_samples['target_protein_id'] == target_id]
    # 分离正负样本
    pos_group = group[group['label'] == 1]
    neg_group = group[group['label'] == 0]
    
    # 优先从 test_compounds 中抽取正负样本
    test_pos_from_existing = pos_group[pos_group['compound_id'].isin(test_compounds)].sample(n=min(len(pos_group[pos_group['compound_id'].isin(test_compounds)]), MAX_SAMPLES_PER_TARGET), random_state=RANDOM_SEED)
    test_neg_from_existing = neg_group[neg_group['compound_id'].isin(test_compounds)].sample(n=min(len(neg_group[neg_group['compound_id'].isin(test_compounds)]), MAX_SAMPLES_PER_TARGET), random_state=RANDOM_SEED)
    
    # 补充不够的部分从 unique_compounds 中抽取
    remaining_pos = MAX_SAMPLES_PER_TARGET - len(test_pos_from_existing)
    remaining_neg = MAX_SAMPLES_PER_TARGET - len(test_neg_from_existing)
    
    if remaining_pos > 0:
        test_pos_from_unique = pos_group.sample(n=min(remaining_pos, len(pos_group)), random_state=RANDOM_SEED)
    else:
        test_pos_from_unique = pd.DataFrame(columns=pos_group.columns)

    if remaining_neg > 0:
        test_neg_from_unique = neg_group.sample(n=min(remaining_neg, len(neg_group)), random_state=RANDOM_SEED)
    else:
        test_neg_from_unique = pd.DataFrame(columns=neg_group.columns)

    # 合并测试集样本
    test_pos = pd.concat([test_pos_from_existing, test_pos_from_unique])
    test_neg = pd.concat([test_neg_from_existing, test_neg_from_unique])
    # print(len(test_pos), len(test_neg))
    
    # 更新测试集和化合物集合
    test_data = pd.concat([test_data, test_pos, test_neg])
    test_compounds.update(test_pos['compound_id'])
    test_compounds.update(test_neg['compound_id'])

# 从总数据中移除测试集化合物的所有记录，作为训练集
train_data = all_samples[~all_samples['compound_id'].isin(test_compounds)]

# 保存训练集和测试集
output_dir = './9_split_database/'
os.makedirs(output_dir, exist_ok=True)
train_data.to_csv(os.path.join(output_dir, 'training_dataset.csv'), index=False)
test_data.to_csv(os.path.join(output_dir, 'test_dataset.csv'), index=False)

# 打印数据集信息
print(f"Training set size: {train_data.shape}")
print(f"Test set size: {test_data.shape}")

train_label_counts = train_data['label'].value_counts()
print("Train positive samples:", train_label_counts.get(1, 0))
print("Train negative samples:", train_label_counts.get(0, 0))

test_label_counts = test_data['label'].value_counts()
print("Test positive samples:", test_label_counts.get(1, 0))
print("Test negative samples:", test_label_counts.get(0, 0))

# 验证测试集化合物是否在训练集中
test_compounds_set = set(test_data['compound_id'])
train_compounds_set = set(train_data['compound_id'])
assert len(test_compounds_set.intersection(train_compounds_set)) == 0, "测试集化合物不应出现在训练集中"
print("✅ 验证成功：测试集化合物完全不在训练集中")
