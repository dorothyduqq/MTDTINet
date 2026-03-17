import pandas as pd
import os
from rdkit import Chem

def standardize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False) if mol else None
    except Exception:
        return None

# 定义路径
input_folder = './2_add_ids/'  # 输入文件夹路径
output_file = './3_get_potential_interaction/interaction_results_by_protein.xlsx'

df = pd.read_csv(f'./2_add_ids/predict_results_batch_0_with_ids.csv')

# 筛选数据
filtered_df = df[(df['output_class'] >= 0.5)]# & (df['output_reg'] >= 7)]

# 筛选出超过 50% 的 Smiles 并删除
smiles_counts = filtered_df.groupby("Smiles")["target_protein_id"].nunique()
# total_targets = filtered_df["target_protein_id"].nunique()
# smiles_to_remove = smiles_counts[smiles_counts / total_targets >= 0.5].index
smiles_to_remove = smiles_counts[smiles_counts >= 100].index
filtered_df = filtered_df[~filtered_df["Smiles"].isin(smiles_to_remove)]

# 合并相同的smiles
grouped = filtered_df.groupby('target_protein_id').agg({
    'canonical_smiles': lambda x: ';'.join(map(str, x)),
    'output_class': lambda x: ';'.join(map(str, x)),
    'output_reg': lambda x: ';'.join(map(str, x)),
    'ID': lambda x: ';'.join(map(str, x)),
    'Smiles': lambda x: ';'.join(map(str, x)),
}).reset_index()

grouped['count'] = filtered_df.groupby('target_protein_id').size().values

pos_data = pd.read_csv(f'../1_data_preparation_plus/3_get_pos_datas/pos_pre_short_data.csv')
pos_data_prot = pos_data[['target_protein_id', 'target_chembl_id', 'uniprot_id']].drop_duplicates()

prot_data_path = '../0_database/ChEMBL34_CTI_literature_only/ChEMBL34_CTI_literature_only_full_dataset.csv'
prot_data = pd.read_csv(prot_data_path, delimiter = ';')
prot_data = prot_data[['target_chembl_id', 'target_pref_name', 'target_type', 'organism']].drop_duplicates(subset = ['target_chembl_id'])

prot_data = pd.merge(prot_data, pos_data_prot, on='target_chembl_id', how='outer')
prot_data = prot_data.drop_duplicates(subset=['target_chembl_id'])

prot_data = (prot_data.groupby('target_protein_id', as_index=False).agg({
    'target_chembl_id': lambda x: ';'.join(x.dropna().astype(str)), 
    'uniprot_id': lambda x: ';'.join(x.dropna().astype(str)), 
    'target_pref_name': lambda x: ';'.join(x.dropna().astype(str)), 
    'target_type': lambda x: ';'.join(x.dropna().astype(str)), 
    'organism': lambda x: ';'.join(x.dropna().astype(str))
    }))

final_df = pd.merge(grouped, prot_data, on='target_protein_id', how='left')

final_df = final_df.sort_values(by='count', ascending=False)
final_df.to_excel(output_file, index=False)
