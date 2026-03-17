import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm

# 字体设置
font_path_bold = './ARIALBD.TTF'
font_prop_bold = fm.FontProperties(fname=font_path_bold)
fm.fontManager.addfont(font_path_bold)

font_path = './ARIAL.TTF'
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['font.size'] = 16
plt.rcParams['axes.unicode_minus'] = False


positive_data = pd.read_csv(f'./6_positive_database/positive_database.csv')

bins = np.arange(4, 12)  # 区间为 [2, 3), [3, 4), ..., [11, 12)
plt.figure(figsize=(4.6, 6))
plt.hist(positive_data['pchembl'], bins=bins, color=(95/255, 133/255, 189/255), edgecolor='black', linewidth=0.5)
plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f'{x/1000:.0f}'))
plt.xticks(np.arange(4, 12, 1))
plt.xlabel('pChEMBL')
plt.ylabel('Frequency (×10³)')

# 显示图形
plt.tight_layout()
plt.savefig(f'6_positive_database/pchembl_stat.pdf')
plt.close()


target_counts = positive_data['target_protein_id'].value_counts()
target_counts = target_counts.sort_values()

# 生成排名（1 为最少出现的）
target_counts = target_counts.reset_index()
target_counts.columns = ['target_protein_id', 'count']
target_counts['rank'] = range(1, len(target_counts) + 1)
target_counts['log2_count'] = np.log2(target_counts['count'])

# 绘制散点图
plt.figure(figsize=(8, 6))
plt.scatter(target_counts['rank'], target_counts['log2_count'], s=5, color=(95/255, 133/255, 189/255))
plt.xlabel("Protein")
plt.ylabel("Number of Compounds ($\log_2$scale)")
plt.xticks([1] + list(range(100, len(target_counts), 100)) + [len(target_counts['rank'])])
plt.tight_layout()
plt.savefig(f'6_positive_database/target_compound_count.pdf')


compound_id = set(positive_data['compound_id'].tolist())

drug_encodings = pd.read_pickle('./4_characterize_pos_datas/drug_encodings.pkl')
drug_encodings = drug_encodings[drug_encodings['compound_id'].isin(compound_id)].drop_duplicates(subset=['compound_id'])
# 绘制node_features数量分布直方图
node_counts = drug_encodings['node_features'].apply(lambda x: x.shape[0])

plt.figure(figsize=(4.7, 6))
# 定义 0～60 的区间边界
bins_first = np.arange(0, 61, 10)  # 得到 [0, 10, 20, 30, 40, 50, 60]
# 计算 0～60 内各区间的计数，注意 np.histogram 默认区间为左闭右开
counts_first, _ = np.histogram(node_counts, bins=bins_first)
# 计算 >= 60 的数据计数
count_last = np.sum(node_counts >= 60)
# 绘制 0-10, 10-20, 20-30, 30-40, 40-50, 50-60 的柱状图，条宽均为 10
plt.bar(bins_first[:-1], counts_first, width=10, align='edge',
        color=(95/255, 133/255, 189/255), edgecolor='black', linewidth=0.5)
# 绘制最后一个 bin：将所有 >= 60 的数据合并，条宽固定为 10，显示在 60～70 区间
plt.bar(60, count_last, width=10, align='edge',
        color=(95/255, 133/255, 189/255), edgecolor='black', linewidth=0.5)

xtick_labels = list(bins_first) + [node_counts.max()]
xtick_positions = list(bins_first) + [70]
plt.xticks(xtick_positions, labels=xtick_labels)
plt.xlim(-5, 75)
plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f'{x/1000:.0f}'))
plt.xlabel('Atom Count')
plt.ylabel('Frequency (×10³)')
plt.tight_layout()
plt.savefig('./6_positive_database/node_counts.pdf')
plt.close()


target_protein_id = set(positive_data['target_protein_id'].tolist())

protein_encodings = pd.read_pickle('./4_characterize_pos_datas/protein_encodings.pkl')
protein_encodings = protein_encodings[protein_encodings['target_protein_id'].isin(target_protein_id)].drop_duplicates(subset=['target_protein_id'])

prot_length = protein_encodings['onehot_encoding'].apply(lambda x: x.shape[0])

plt.figure(figsize=(8, 6))
plt.hist(prot_length, bins=range(0, int(prot_length.max()) + 10, 100), color=(95/255, 133/255, 189/255), edgecolor='black', linewidth=0.5)
plt.xlabel('Protein Length')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('./6_positive_database/prot_length.pdf')
plt.close()
