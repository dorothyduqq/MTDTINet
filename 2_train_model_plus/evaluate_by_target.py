import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.data import Batch
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, mean_squared_error, roc_auc_score
from scipy.stats import pearsonr
from model_structure import DrugProteinModel
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
font_path_bold = './ARIALBD.TTF'
font_prop_bold = fm.FontProperties(fname=font_path_bold)
fm.fontManager.addfont(font_path_bold)

font_path = './ARIAL.TTF'
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['font.size'] = 16
plt.rcParams['axes.unicode_minus']=False

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

class DynamicDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # 动态生成数据
        node_features = torch.tensor(row['node_features'], dtype=torch.float)
        adj_matrix = torch.tensor(row['adj_matrix'], dtype=torch.long)
        edge_index = adj_matrix.nonzero(as_tuple=False).t().contiguous()
        drug_graph = Data(x=node_features, edge_index=edge_index)
        aaindex_encoding = torch.tensor(row['aaindex_encoding'], dtype=torch.float)
        label = torch.tensor(row['pchembl'], dtype=torch.float)
        return drug_graph, aaindex_encoding, label

# 使用动态数据集加载器
def create_dataloader(data, batch_size, shuffle=True):
    dataset = DynamicDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=2, prefetch_factor=2, pin_memory=True, persistent_workers=True)

def collate_fn(batch):
    drug_graphs, aaindex_encodings, labels = zip(*batch)
    drug_graphs = Batch.from_data_list(drug_graphs)
    aaindex_encodings = torch.stack(aaindex_encodings).permute(0, 2, 1)
    labels = torch.tensor(labels)
    return drug_graphs, aaindex_encodings, labels

def evaluate_metrics(true_labels, predictions):
    true_labels = (true_labels >= 4).astype(int).reshape(-1)
    pred_labels = (predictions >= 0.5).astype(int).reshape(-1)
    acc = accuracy_score(true_labels, pred_labels)
    auc = roc_auc_score(true_labels, predictions)
    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
    sn = tp / (tp + fn)
    sp = tn / (tn + fp)
    prec = tp / (tp + fp)
    f1 = 2 * tp / (2 * tp + fn + fp)
    mcc = matthews_corrcoef(true_labels, pred_labels)
    return acc, sn, sp, prec, f1, auc, mcc

if __name__ == '__main__':
    batch_size = 256

    original_data_path = '../0_database/ChEMBL34_CTI_literature_only/ChEMBL34_CTI_literature_only_full_dataset.csv'
    train_data_path = '../1_data_preparation_plus/9_split_database/training_dataset.csv'
    test_data_path = '../1_data_preparation_plus/9_split_database/test_dataset.csv'
    drug_encodings_path = '../1_data_preparation_plus/5_normalize_and_pad_encodings/normalized_and_no_padded_drug_encodings.pkl'
    protein_encodings_path = '../1_data_preparation_plus/5_normalize_and_pad_encodings/normalized_and_padded_protein_encodings.pkl'
    pos_data_path = '../1_data_preparation_plus/3_get_pos_datas/pos_pre_short_data.csv'
    model_save_dir = './best_models'
    evaluation_results_dir = './evaluation'

    # 加载数据
    drug_encodings = pd.read_pickle(drug_encodings_path)
    protein_encodings = pd.read_pickle(protein_encodings_path)
    test_data = pd.read_csv(test_data_path)
    pos_data = pd.read_csv(pos_data_path)  # 加载 pos 数据
    original_data = pd.read_csv(original_data_path, sep=';')
    original_data = original_data[['target_chembl_id', 'target_pref_name', 'organism', 'target_class_l1', 'target_class_l2']].drop_duplicates(subset = ['target_chembl_id'])

    test_data = test_data.merge(drug_encodings, on='compound_id').merge(protein_encodings, on='target_protein_id')

    fold_no = 0
    best_models = [f"{model_save_dir}/best_model_fold_{i}.pth" for i in range(1, 6)]

    model = DrugProteinModel().to(device)

    evaluation_results = []

    # 按 target_protein_id 分组
    grouped_data = test_data.groupby('target_protein_id')

    for target_protein_id, group in grouped_data:
        group_labels = group['pchembl'].values
        group_loader = create_dataloader(group, batch_size=batch_size, shuffle=False)
        group_preds_class = np.zeros(group.shape[0])
        group_preds_reg = np.zeros(group.shape[0])

        for model_path in best_models:
            fold_no += 1
            model.load_state_dict(torch.load(model_path, weights_only=True))
            model.eval()
            fold_preds_reg, fold_preds_class = [], []

            with torch.no_grad():
                for drug_data, protein_data, labels in group_loader:
                    drug_data = drug_data.to(device)
                    protein_data = protein_data.to(device)
                    labels = labels.to(device)
                    output_class, output_reg = model(drug_data, protein_data)
                    fold_preds_class.extend(output_class.cpu().numpy())
                    fold_preds_reg.extend(output_reg.cpu().numpy())
                fold_preds_class = np.array(fold_preds_class)
                fold_preds_reg = np.array(fold_preds_reg)
            group_preds_class += fold_preds_class
            group_preds_reg += fold_preds_reg

        # 取模型预测均值
        group_preds_class_mean = group_preds_class / len(best_models)
        group_preds_reg_mean = group_preds_reg / len(best_models)

        try:
            group_acc, group_sn, group_sp, group_prec, group_f1, group_auc, group_mcc = evaluate_metrics(group_labels, group_preds_class_mean)
        except:
            continue

        # 筛选分类预测为正样本的数据
        valid_mask = group_labels >= 4
        valid_labels = group_labels[valid_mask]
        valid_preds = group_preds_reg_mean[valid_mask]

        if len(valid_labels) >= 2:  # 确保有有效样本
            test_mse = mean_squared_error(valid_labels, valid_preds)
            test_corr, test_corr_p_value = pearsonr(valid_labels, valid_preds)
            sample_counts = len(valid_labels)
            print(target_protein_id, group_acc, group_sn, group_sp, group_prec, group_f1, group_auc, group_mcc, test_mse, test_corr, test_corr_p_value, sample_counts)
            evaluation_results.append([target_protein_id, group_acc, group_sn, group_sp, group_prec, group_f1, group_auc, group_mcc, test_mse, test_corr, test_corr_p_value, sample_counts])

            # 如果相关系数和 p 值满足条件，绘制散点图
            if group_acc >= 0.9 and test_corr >= 0.75 and test_mse <= 0.8 and test_corr_p_value < 0.001:
                plt.figure(figsize=(7, 6))
                plt.scatter(valid_labels, valid_preds, alpha=1, s=4, color=(59/255,108/255,197/255), rasterized=True, zorder=0)
                plt.plot([min(min(valid_labels), min(valid_preds)), max(max(valid_labels), max(valid_preds))], [min(min(valid_labels), min(valid_preds)), max(max(valid_labels), max(valid_preds))], linestyle='-', lw=2, color=(252/255,73/255,106/255), alpha=0.8, rasterized=True, zorder=1)
                plt.title(f"Target Protein ID: {target_protein_id.upper()}")
                plt.xlabel("True pChEMBL Value")
                plt.ylabel("Predicted pChEMBL Value")
                plt.grid(alpha=0.3)
                plt.text(0.05, 0.95, f"r = {test_corr:.2f}",
                            transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
                plt.tight_layout()
                # 保存图像
                save_path = os.path.join(evaluation_results_dir, 'evaluate_by_target', f"{target_protein_id}.pdf")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=1200) #bbox_inches='tight', 
                plt.close()

    # 保存评价结果
    results_df = pd.DataFrame(evaluation_results, columns=['Target protein id', "ACC", "Sn", "Sp", "Precision", "F1-score", "AUC", "MCC", 'Positive MSE', 'Positive R', 'Positive R P value', 'Positive sample count'])
    results_df = results_df.sort_values(by='Positive R', ascending=False)

    # 合并 pos 数据中的关系信息
    pos_data = pos_data[['target_protein_id', 'target_chembl_id', 'uniprot_id']].drop_duplicates()
    pos_data = pd.merge(pos_data, original_data, on='target_chembl_id', how='left')
    pos_data = (pos_data.groupby('target_protein_id', as_index=False).agg({'target_chembl_id': lambda x: ';'.join(x.dropna().astype(str)), 'uniprot_id': lambda x: ';'.join(x.dropna().astype(str)), 
                'target_pref_name': lambda x: ';'.join(x.dropna().astype(str)), 'organism': lambda x: ';'.join(x.dropna().astype(str)), 'target_class_l1': lambda x: ';'.join(x.dropna().astype(str)), 
                'target_class_l2': lambda x: ';'.join(x.dropna().astype(str))}))
    merged_results = pd.merge(results_df, pos_data, left_on='Target protein id', right_on='target_protein_id', how='left')

    # 保存合并后的结果
    results_path = os.path.join(evaluation_results_dir, 'evaluate_with_pos_data.csv')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    merged_results.to_csv(results_path, index=False)
