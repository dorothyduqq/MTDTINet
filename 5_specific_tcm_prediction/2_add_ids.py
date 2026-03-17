import os
import pandas as pd
from rdkit import Chem

def standardize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False) if mol else None
    except Exception:
        return None

tcm_path = './tcm_data.txt'
tcm_df = pd.read_csv(tcm_path, delimiter='\t', header=None, names=['Smiles', 'ID'])
tcm_df["canonical_smiles"] = tcm_df["Smiles"].apply(standardize_smiles)

tcm_df = tcm_df.groupby('canonical_smiles', as_index=False).agg({'ID': lambda x: ';'.join(x.dropna().astype(str)),
                                                                'Smiles': lambda x: ';'.join(x.dropna().astype(str))})

# Step 5: Process CSV files in ./1_tcm_target_predict/
input_directory = "./1_tcm_target_predict/"
output_directory = "./2_add_ids/"
os.makedirs(output_directory, exist_ok=True)

for filename in os.listdir(input_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_directory, filename)
        data = pd.read_csv(file_path)
        data = pd.merge(data, tcm_df, left_on="smiles", right_on="canonical_smiles", how="left")
        data.drop(columns = ['smiles'], inplace = True)
        output_path = os.path.join(output_directory, filename.replace(".csv", "_with_ids.csv"))
        data.to_csv(output_path, index=False)
