import matplotlib.pyplot as plt
import pandas as pd

# 1. 从CSV文件中读取target_seq列，统计蛋白质序列长度
csv_path = "./1_get_full_datas/pre_database.csv"
df = pd.read_csv(csv_path)

# 获取除标题外的行数
num_rows = len(df)
print(f'CSV文件共有 {num_rows} 行（不包括标题）')

# 统计target_chembl_id列中不同值的数量
unique_target_chembl_ids = df['target_chembl_id'].nunique()
print(f'target_chembl_id列中有 {unique_target_chembl_ids} 个不同的值')

unique_uniprot_ids = df['uniprot_id'].nunique()
print(f'uniprot_id列中有 {unique_uniprot_ids} 个不同的值')

unique_target_seqs = df['target_seq'].nunique()
print(f'target_seq列中有 {unique_target_seqs} 个不同的值')

# 统计parent_chembl_id列中不同值的数量
unique_parent_chembl_ids = df['parent_chemblid'].nunique()
print(f'parent_chembl_id列中有 {unique_parent_chembl_ids} 个不同的值')

# 蛋白去重前的直方图-----------------------------------------------------------------------------
# 获取target_seq列中的序列，并计算每个序列的长度
lengths = df['target_seq'].apply(len).tolist()

# 2. 定义长度区间
bin_edges = list(range(0, max(lengths) + 50, 50))

# 3. 绘制直方图
plt.figure(figsize=(30, 6))
counts, bins, patches = plt.hist(lengths, bins=bin_edges, color='skyblue', edgecolor='black')
plt.xticks(range(0, int(max(bins)) + 100, 100))

# 4. 在直方图柱上方标注蛋白质个数
for count, patch in zip(counts, patches):
    plt.text(patch.get_x() + patch.get_width() / 2, count + 1, int(count), ha='center', va='bottom')

# 5. 设置标题和标签
plt.title(f"Protein Length Distribution: protein number is {len(lengths)}")
plt.xlabel("Protein Length")
plt.ylabel("Number of Proteins")

# 6. 保存为PDF文件
output_path = "./2_statistic_full_datas/length_distribution.pdf"
plt.savefig(output_path, format='pdf')



# 蛋白去重后的直方图---------------------------------------------------------------------------
unique_lengths = df['target_seq'].drop_duplicates().apply(len).tolist()

# 2. 定义长度区间
bin_edges = list(range(0, max(unique_lengths) + 50, 50))

# 3. 绘制直方图
plt.figure(figsize=(30, 6))
counts, bins, patches = plt.hist(unique_lengths, bins=bin_edges, color='skyblue', edgecolor='black')
plt.xticks(range(0, int(max(bins)) + 100, 100))

# 4. 在直方图柱上方标注蛋白质个数
for count, patch in zip(counts, patches):
    plt.text(patch.get_x() + patch.get_width() / 2, count + 1, int(count), ha='center', va='bottom')

# 5. 设置标题和标签
plt.title(f"Protein Length Distribution: protein number is {len(unique_lengths)}")
plt.xlabel("Protein Length")
plt.ylabel("Number of Proteins")

# 6. 保存为PDF文件
output_path = "./2_statistic_full_datas/length_distribution_remove.pdf"
plt.savefig(output_path, format='pdf')


# 不同靶蛋白长度-个数
target_stats = (
    df['target_seq']
    .value_counts()  # 统计每个元素的出现次数
    .reset_index()   # 重置索引，以便转换成DataFrame
    .rename(columns={'index': 'target_seq', 'target_seq': 'count'})  # 重命名列
)

target_stats.columns = ['target_seq', 'count']  
# 计算每个target_seq的长度
target_stats['length'] = target_stats['target_seq'].apply(len)

# 保存统计结果到新的CSV文件
output_csv_path = "./2_statistic_full_datas/target_seq_statistics.csv"
target_stats.to_csv(output_csv_path, index=False)

plt.figure(figsize=(10, 6))
plt.scatter(target_stats['count'], target_stats['length'], alpha=0.6)
plt.xlabel("Count")
plt.ylabel("Length")
plt.title("Scatter Plot of Count vs Length for target_seq Elements")
plt.grid(True)
plt.savefig('target_count_length.pdf', format='pdf')
