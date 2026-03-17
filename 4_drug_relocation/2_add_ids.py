import os
import pandas as pd

pos_data_path = '../1_data_preparation_plus/3_get_pos_datas/pos_pre_short_data.csv'
pos_data = pd.read_csv(pos_data_path)
pos_data_prot = pos_data[['target_protein_id', 'target_chembl_id', 'uniprot_id']].drop_duplicates()
pos_data_prot = (pos_data_prot.groupby('target_protein_id', as_index=False).agg({'target_chembl_id': lambda x: ';'.join(x.dropna().astype(str)), 'uniprot_id': lambda x: ';'.join(x.dropna().astype(str))}))

pos_data_drug = pos_data[['compound_id', 'parent_chemblid', 'canonical_smiles']].drop_duplicates(subset = ['canonical_smiles'])

# Step 5: Process CSV files in ./1_tcm_target_predict/
input_directory = "./1_drug_target_predict/"
output_directory = "./2_add_ids/"
os.makedirs(output_directory, exist_ok=True)

for filename in os.listdir(input_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_directory, filename)
        data = pd.read_csv(file_path)
        data = pd.merge(data, pos_data_drug, left_on="smiles", right_on="canonical_smiles", how="left")
        data.drop(columns = ['canonical_smiles'], inplace = True)
        data = pd.merge(data, pos_data_prot, on='target_protein_id', how='left')
        output_path = os.path.join(output_directory, filename.replace(".csv", "_with_ids.csv"))
        data.to_csv(output_path, index=False)
