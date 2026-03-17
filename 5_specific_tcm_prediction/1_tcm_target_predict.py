import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch
from model_structure import DrugProteinModel
from rdkit import Chem
from tqdm import trange

def standardize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False) if mol else None
    except Exception:
        return None

# Helper functions remain unchanged
def get_atom_features(atom):
    return {
        'atomic_num': atom.GetAtomicNum(),
        'degree': atom.GetDegree(),
        'num_hs': atom.GetTotalNumHs(),
        'implicit_valence': atom.GetImplicitValence(),
        'is_aromatic': int(atom.GetIsAromatic()),
        'mass': atom.GetMass(),
        'is_in_ring': int(atom.IsInRing())
    }

def get_drug_encodings(smiles):
    mol = Chem.MolFromSmiles(smiles)
    node_features = np.array([list(get_atom_features(atom).values()) for atom in mol.GetAtoms()])
    node_features = torch.tensor((node_features - norm_mins) / (norm_maxs - norm_mins), dtype=torch.float)
    adj_matrix = torch.tensor(Chem.GetAdjacencyMatrix(mol), dtype=torch.long)
    edge_index = adj_matrix.nonzero(as_tuple=False).t().contiguous()
    drug_graph = Data(x=node_features, edge_index=edge_index)
    return drug_graph

def is_valid_smiles(smiles):
    if pd.isna(smiles) or smiles in [None, '', 'nan']:
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False

# Helper function for batch encoding
def get_batch_drug_encodings(smiles_list):
    drug_graphs = [get_drug_encodings(smiles) for smiles in smiles_list]
    return Batch.from_data_list(drug_graphs)

# Resume mechanism
def get_last_processed_batch(output_dir):
    files = [f for f in os.listdir(output_dir) if f.startswith("predict_results_batch_")]
    if files:
        last_batch = max(int(f.split("_")[-1].split(".")[0]) for f in files)
        return last_batch + 1
    return 0

# 批量预测函数
def predict_in_batches(models, drug_batch, protein_batch, batch_size):
    predictions_class = []
    predictions_reg = []

    # 将 drug_batch 按 predict_batch_size 分割
    num_samples = drug_batch.num_graphs
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        
        # 创建子批次的 drug_batch 和 protein_batch
        sub_drug_batch = Batch.from_data_list(drug_batch.to_data_list()[start:end]).to(device)
        sub_protein_batch = protein_batch[start:end].to(device)

        # 模型预测
        sub_predictions_class = []
        sub_predictions_reg = []
        for model in models:
            with torch.no_grad():
                output_class, output_reg = model(sub_drug_batch, sub_protein_batch)
                sub_predictions_class.append(output_class.cpu().numpy())
                sub_predictions_reg.append(output_reg.cpu().numpy())

        # 合并每个模型的子批次结果
        predictions_class.append(sum(sub_predictions_class) / len(sub_predictions_class))
        predictions_reg.append(sum(sub_predictions_reg) / len(sub_predictions_reg))

    # 将所有子批次结果拼接
    predictions_class = np.concatenate(predictions_class, axis=0)
    predictions_reg = np.concatenate(predictions_reg, axis=0)

    return predictions_class, predictions_reg

# Paths
protein_path = '../1_data_preparation_plus/5_normalize_and_pad_encodings/normalized_and_padded_protein_encodings.pkl'
evaluation_path = '../2_train_model_plus/evaluation/effect_classification_targets.csv'
tcm_path = './tcm_data.txt'
output_dir = './1_tcm_target_predict/'
drug_norm_path = '../1_data_preparation_plus/5_normalize_and_pad_encodings/node_features_normalization_params.csv'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Data loading and preprocessing
protein_df = pd.read_pickle(protein_path)
protein_df = protein_df[['target_protein_id', 'aaindex_encoding']].drop_duplicates(subset=['target_protein_id'])

evaluation_df = pd.read_csv(evaluation_path)
filtered_targets = evaluation_df['Target protein id'].unique()
filtered_protein_df = protein_df[protein_df['target_protein_id'].isin(filtered_targets)]

tcm_df = pd.read_csv(tcm_path, delimiter='\t', header=None, names=['Smiles', 'ID'])
tcm_df["canonical_smiles"] = tcm_df["Smiles"].apply(standardize_smiles)

tcm_df = tcm_df.groupby('canonical_smiles')['ID'].apply(lambda x: ';'.join(map(str, x))).reset_index()
tcm_df = tcm_df[tcm_df['canonical_smiles'].apply(is_valid_smiles)]
tcm_df = tcm_df[tcm_df['canonical_smiles'].str.len() <= 350]

drug_norm_df = pd.read_csv(drug_norm_path)
norm_mins = drug_norm_df['min'].to_numpy()
norm_maxs = drug_norm_df['max'].to_numpy()

# Model loading
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
best_models = [f"../2_train_model_plus/best_models/best_model_fold_{i}.pth" for i in range(1, 6)]
models = [DrugProteinModel().to(device) for _ in best_models]
for model, path in zip(models, best_models):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

# Batch processing
batch_size = 10000
predict_batch_size = 500  # 调整此值以适应显存容量
start_batch = get_last_processed_batch(output_dir)
tcm_smiles = tcm_df['canonical_smiles'].tolist()
target_protein_ids = filtered_protein_df['target_protein_id'].tolist()

protein_encodings = torch.tensor(
    [protein_row['aaindex_encoding'] for _, protein_row in filtered_protein_df.iterrows()],
    dtype=torch.float
).permute(0, 2, 1).unsqueeze(1)

# 按批次处理药物数据
for batch_idx in trange(start_batch, len(tcm_smiles) // batch_size + 1):
    batch_start = batch_idx * batch_size
    batch_end = min((batch_idx + 1) * batch_size, len(tcm_smiles))
    batch_smiles = tcm_smiles[batch_start:batch_end]
    tcm_batch = get_batch_drug_encodings(batch_smiles)

    results = []
    for j, target_protein_id in enumerate(target_protein_ids):
        protein_batch = protein_encodings[j].squeeze(0).expand(len(batch_smiles), -1, -1)

        # 使用分批次预测函数
        avg_class, avg_reg = predict_in_batches(models, tcm_batch, protein_batch, predict_batch_size)

        for k, smiles in enumerate(batch_smiles):
            results.append({
                'smiles': smiles,
                'target_protein_id': target_protein_id,
                'output_class': avg_class[k],
                'output_reg': avg_reg[k]
            })

    # 保存结果至文件
    result_df = pd.DataFrame(results)
    result_df.to_csv(os.path.join(output_dir, f"predict_results_batch_{batch_idx}.csv"), index=False)

