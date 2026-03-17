import pandas as pd

# 读取CSV文件 
input_file = './1_get_full_datas/pre_database.csv'
short_output_file = './3_get_pos_datas/pos_pre_short_data.csv'
long_output_file = './3_get_pos_datas/pos_pre_long_data.csv'
short_stats_file = './3_get_pos_datas/statistics_short_data.csv'
long_stats_file = './3_get_pos_datas/statistics_long_data.csv'

# 加载数据
df = pd.read_csv(input_file)

# 为canonical_smiles列分配compound_id
unique_smiles = df['canonical_smiles'].unique()
smiles_id_map = {smiles: f"d{str(i+1).zfill(6)}" for i, smiles in enumerate(unique_smiles)}
df['compound_id'] = df['canonical_smiles'].map(smiles_id_map)

# 为target_seq列分配target_protein_id
unique_targets = df['target_seq'].unique()
target_id_map = {target: f"tp{str(i+1).zfill(4)}" for i, target in enumerate(unique_targets)}
df['target_protein_id'] = df['target_seq'].map(target_id_map)

# 条件 1：target_seq 的长度小于等于 1400
# 条件 2：canonical_smiles 的长度小于等于 350
# 条件 3：target_seq 不包含 U, Z, O, B, X
short_data = df[
    (df['target_seq'].str.len() <= 1400) &
    (df['canonical_smiles'].str.len() <= 350) &
    (~df['target_seq'].str.contains(r'[UZOXB]'))
]
# 其余的数据属于 long_data
long_data = df[~(
    (df['target_seq'].str.len() <= 1400) &
    (df['canonical_smiles'].str.len() <= 350) &
    (~df['target_seq'].str.contains(r'[UZOXB]'))
)]

# 按指定顺序排列列
column_order = ['compound_id', 'parent_chemblid', 'canonical_smiles', 'target_protein_id', 
                'target_chembl_id', 'uniprot_id', 'target_seq', 'pchembl']
short_data = short_data[column_order]
long_data = long_data[column_order]

# 保存short和long数据集
short_data.to_csv(short_output_file, index=False)
long_data.to_csv(long_output_file, index=False)

# 统计每个target_seq的长度和数量
def generate_statistics(data, output_file):
    # 计算长度并去重
    statistics = data['target_seq'].apply(len).value_counts().reset_index()
    statistics.columns = ['Length', 'Count']
    statistics = statistics.sort_values(by='Length').reset_index(drop=True)
    
    # 保存统计信息
    statistics.to_csv(output_file, index=False)

# 生成并保存统计数据
generate_statistics(short_data, short_stats_file)
generate_statistics(long_data, long_stats_file)
