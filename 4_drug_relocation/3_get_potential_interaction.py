import pandas as pd
import os

# 定义路径
input_folder = './2_add_ids/'  # 输入文件夹路径
output_file = './3_get_potential_interaction/interaction_results.csv'

pos_data_path = '../1_data_preparation_plus/3_get_pos_datas/pos_pre_short_data.csv'
pos_data = pd.read_csv(pos_data_path)
pos_data = pos_data.copy()
pos_data.loc[:, 'comp_key'] = pos_data['compound_id'] + '_' + pos_data['target_protein_id']

# 初始化结果数据框
final_df = pd.DataFrame()

# 遍历文件夹中的所有CSV文件
for file_name in os.listdir(input_folder):
    if file_name.endswith('.csv'):  # 只处理CSV文件
        input_file = os.path.join(input_folder, file_name)
        print(input_file)
        
        # 读取数据
        df = pd.read_csv(input_file)

        # 筛选数据
        filtered_df = df[(df['output_class'] >= 0.5) & (df['output_reg'] > 9)]
        filtered_df = filtered_df.copy()
        filtered_df.loc[:, 'comp_key'] = filtered_df['compound_id'] + '_' + filtered_df['target_protein_id']
        # 筛选出不在pos_data中的行
        filtered_df = filtered_df[~filtered_df['comp_key'].isin(pos_data['comp_key'])]
        filtered_df.drop(columns = ['comp_key'], inplace = True)

        # 合并相同的smiles
        grouped = filtered_df.groupby('smiles').agg({
            'output_class': lambda x: '；'.join(map(str, x)),
            'output_reg': lambda x: '；'.join(map(str, x)),
            'compound_id': 'first',
            'parent_chemblid': 'first',
            'target_protein_id': lambda x: '；'.join(map(str, x)),
            'target_chembl_id': lambda x: '；'.join(map(str, x)),
            'uniprot_id': lambda x: '；'.join(map(str, x)),
        }).reset_index()
        grouped['count'] = filtered_df.groupby('smiles').size().values
        # 合并结果
        final_df = pd.concat([final_df, grouped], ignore_index=True)

compound_data_path = '../0_database/ChEMBL34_CTI_literature_only/ChEMBL34_CTI_literature_only_full_dataset.csv'
compound_data = pd.read_csv(compound_data_path, delimiter = ';')
compound_data = compound_data[['canonical_smiles', 'parent_pref_name', 'max_phase', 'first_approval', 'usan_year']].drop_duplicates(subset = ['canonical_smiles'])

final_df = pd.merge(final_df, compound_data, left_on='smiles', right_on='canonical_smiles', how='left')
final_df.drop(columns = ['canonical_smiles'], inplace = True)

# 保存最终结果
final_df = final_df.sort_values(by='count', ascending=False)
final_df.to_csv(output_file, index=False)
