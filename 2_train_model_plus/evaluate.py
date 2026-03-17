import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.data import Batch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, mean_squared_error, roc_curve, roc_auc_score
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

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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

def plot_detail(type):
    plt.plot([0, 100], [0, 100], linestyle='--', lw=2, color='r', label='Random Guess', alpha=0.8)
    plt.tick_params(axis='both', direction='in', top=True, right=True, width=1.5, which='major')
    plt.xlabel('1-Specificity(%)')
    plt.ylabel('Sensitivity(%)')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xticks(np.arange(0, 110, 10))
    plt.yticks(np.arange(0, 110, 10))
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', linewidth=1.5)

    ax = plt.gca()
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    plt.savefig(f'{evaluation_results_dir}/{type}_auc.pdf')

if __name__ == '__main__':
    batch_size = 256

    train_data_path = '../1_data_preparation_plus/9_split_database/training_dataset.csv'
    test_data_path = '../1_data_preparation_plus/9_split_database/test_dataset.csv'
    drug_encodings_path = '../1_data_preparation_plus/5_normalize_and_pad_encodings/normalized_and_no_padded_drug_encodings.pkl'
    protein_encodings_path = '../1_data_preparation_plus/5_normalize_and_pad_encodings/normalized_and_padded_protein_encodings.pkl'
    model_save_dir = './best_models'
    evaluation_results_dir = './evaluation'

    drug_encodings = pd.read_pickle(drug_encodings_path)
    protein_encodings = pd.read_pickle(protein_encodings_path)

    train_data = pd.read_csv(train_data_path)
    train_data = train_data.merge(drug_encodings, on='compound_id').merge(protein_encodings, on='target_protein_id')
    # train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
    # train_data = train_data.head(1000)
    train_labels = np.array(train_data[['pchembl']]).flatten()

    test_data = pd.read_csv(test_data_path)
    test_data = test_data.merge(drug_encodings, on='compound_id').merge(protein_encodings, on='target_protein_id')
    test_labels = np.array(test_data[['pchembl']]).flatten()

    # 在每折交叉验证中评估并保存结果
    results = []
    fold_no = 0
    train_fprs, train_tprs, train_aucs = [], [], []
    test_fprs, test_tprs, test_aucs = [], [], []
    best_models = [f"{model_save_dir}/best_model_fold_{i}.pth" for i in range(1, 6)]

    model = DrugProteinModel().to(device)

    train_loader = create_dataloader(train_data, batch_size=batch_size, shuffle=False)
    test_loader = create_dataloader(test_data, batch_size=batch_size, shuffle=False)

    # 测试阶段
    final_train_predictions_class = np.zeros(train_data.shape[0])
    final_test_predictions_class = np.zeros(test_data.shape[0])
    final_train_predictions_reg = np.zeros(train_data.shape[0])
    final_test_predictions_reg = np.zeros(test_data.shape[0])

    for model_path in best_models:
        # if '3' not in model_path:
        #     continue
        fold_no += 1
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        train_preds_class, test_preds_class, train_preds_reg, test_preds_reg = [], [], [], []
        with torch.no_grad():
            for drug_data, protein_data, labels in tqdm(train_loader, desc=f'Fold {fold_no}', ncols=0, bar_format='{l_bar}{bar}'):
            # for drug_data, protein_data, labels in train_loader:
                drug_data = drug_data.to(device)
                protein_data = protein_data.to(device)
                labels = labels.to(device)
                output_class, output_reg = model(drug_data, protein_data)
                train_preds_class.extend(output_class.cpu().numpy())
                train_preds_reg.extend(output_reg.cpu().numpy())
                # print(min(output_reg.cpu().numpy()))
            train_preds_class = np.array(train_preds_class)
            train_preds_reg = np.array(train_preds_reg)
            train_acc, train_sn, train_sp, train_prec, train_f1, train_auc, train_mcc = evaluate_metrics(train_labels, train_preds_class)
            train_fpr, train_tpr, _ = roc_curve((train_labels >= 4).astype(int).reshape(-1), train_preds_class)
            train_fprs.append(train_fpr)
            train_tprs.append(train_tpr)
            train_aucs.append(train_auc)

            train_mask = train_labels >= 4
            train_preds_reg_pos = train_preds_reg[train_mask]
            train_labels_pos = train_labels[train_mask]
            train_mse = mean_squared_error(train_labels_pos, train_preds_reg_pos)
            train_corr, _ = pearsonr(train_labels_pos, train_preds_reg_pos)

            print(f"Train Fold -> ACC: {train_acc}, Sn: {train_sn}, Sp: {train_sp}, Precision: {train_prec}, F1: {train_f1}, AUC: {train_auc}, MCC: {train_mcc}, MSE: {train_mse}, R: {train_corr}")
            results.append([f"Train Fold {fold_no}", train_acc, train_sn, train_sp, train_prec, train_f1, train_auc, train_mcc, train_mse, train_corr])

            for drug_data, protein_data, labels in tqdm(test_loader, desc=f'Fold {fold_no}', ncols=0, bar_format='{l_bar}{bar}'):
            # for drug_data, protein_data, labels in test_loader:
                drug_data = drug_data.to(device)
                protein_data = protein_data.to(device)
                labels = labels.to(device)
                output_class, output_reg = model(drug_data, protein_data)
                test_preds_class.extend(output_class.cpu().numpy())
                test_preds_reg.extend(output_reg.cpu().numpy())
                # print(min(output_reg.cpu().numpy()))
            test_preds_class = np.array(test_preds_class)
            test_preds_reg = np.array(test_preds_reg)
            test_acc, test_sn, test_sp, test_prec, test_f1, test_auc, test_mcc = evaluate_metrics(test_labels, test_preds_class)
            test_fpr, test_tpr, _ = roc_curve((test_labels >= 4).astype(int).reshape(-1), test_preds_class)
            test_fprs.append(test_fpr)
            test_tprs.append(test_tpr)
            test_aucs.append(test_auc)

            test_mask = test_labels >= 4
            test_preds_reg_pos = test_preds_reg[test_mask]
            test_labels_pos = test_labels[test_mask]
            test_mse = mean_squared_error(test_labels_pos, test_preds_reg_pos)
            test_corr, _ = pearsonr(test_labels_pos, test_preds_reg_pos)

            print(f"Test Fold -> ACC: {test_acc}, Sn: {test_sn}, Sp: {test_sp}, Precision: {test_prec}, F1: {test_f1}, AUC, {test_auc}, MCC: {test_mcc}, MSE: {test_mse}, R: {test_corr}")
            results.append([f"Test Fold {fold_no}", test_acc, test_sn, test_sp, test_prec, test_f1, test_auc, test_mcc, test_mse, test_corr])
        final_train_predictions_class += train_preds_class
        final_test_predictions_class += test_preds_class
        final_train_predictions_reg += train_preds_reg
        final_test_predictions_reg += test_preds_reg

    # final train evaluation
    final_train_predictions_class /= len(best_models)
    final_train_predictions_reg /= len(best_models)
    train_acc, train_sn, train_sp, train_prec, train_f1, train_auc, train_mcc = evaluate_metrics(train_labels, final_train_predictions_class)
    train_fpr, train_tpr, _ = roc_curve((train_labels >= 4).astype(int).reshape(-1), final_train_predictions_class)
    train_fprs.append(train_fpr)
    train_tprs.append(train_tpr)
    train_aucs.append(train_auc)

    train_mask = train_labels >= 4
    train_preds_reg_pos = final_train_predictions_reg[train_mask]
    train_labels_pos = train_labels[train_mask]
    train_mse = mean_squared_error(train_labels_pos, train_preds_reg_pos)
    train_corr, _ = pearsonr(train_labels_pos, train_preds_reg_pos)

    plt.figure(figsize=(7, 6))
    plt.scatter(train_labels_pos, train_preds_reg_pos, alpha=1, s=4, color=(59/255,108/255,197/255), linewidths=0.01, edgecolors=(188/255,190/255,193/255), rasterized=True, zorder=0)
    plt.plot([min(min(train_labels_pos), min(train_preds_reg_pos)), max(max(train_labels_pos), max(train_preds_reg_pos))], [min(min(train_labels_pos), min(train_preds_reg_pos)), max(max(train_labels_pos), max(train_preds_reg_pos))], linestyle='-', lw=2, color=(252/255,73/255,106/255), alpha=0.8, rasterized=True, zorder=1)
    plt.xlabel("True pChEMBL Value")
    plt.ylabel("Predicted pChEMBL Value")
    plt.grid(alpha=0.3)
    plt.text(0.05, 0.95, f"r = {train_corr:.2f}",
                transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
    # 保存图像
    plt.tight_layout()
    plt.savefig(f'./evaluation/train_r.pdf', dpi=1200)
    plt.close()

    print(f"Train -> ACC: {train_acc}, Sn: {train_sn}, Sp: {train_sp}, Precision: {train_prec}, F1: {train_f1}, AUC: {train_auc}, MCC: {train_mcc}, MSE: {train_mse}, R: {train_corr}")
    results.append([f"Train", train_acc, train_sn, train_sp, train_prec, train_f1, train_auc, train_mcc, train_mse, train_corr])

    # final test evaluation
    final_test_predictions_class /= len(best_models)
    final_test_predictions_reg /= len(best_models)
    test_acc, test_sn, test_sp, test_prec, test_f1, test_auc, test_mcc = evaluate_metrics(test_labels, final_test_predictions_class)
    test_fpr, test_tpr, _ = roc_curve((test_labels >= 4).astype(int).reshape(-1), final_test_predictions_class)
    test_fprs.append(test_fpr)
    test_tprs.append(test_tpr)
    test_aucs.append(test_auc)

    test_mask = test_labels >= 4
    test_preds_reg_pos = final_test_predictions_reg[test_mask]
    test_labels_pos = test_labels[test_mask]
    test_mse = mean_squared_error(test_labels_pos, test_preds_reg_pos)
    test_corr, _ = pearsonr(test_labels_pos, test_preds_reg_pos)

    plt.figure(figsize=(7, 6))
    plt.scatter(test_labels_pos, test_preds_reg_pos, alpha=1, s=4, color=(59/255,108/255,197/255), linewidths=0.01, edgecolors=(188/255,190/255,193/255), rasterized=True, zorder=0)
    plt.plot([min(min(test_labels_pos), min(test_preds_reg_pos)), max(max(test_labels_pos), max(test_preds_reg_pos))], [min(min(test_labels_pos), min(test_preds_reg_pos)), max(max(test_labels_pos), max(test_preds_reg_pos))], linestyle='-', lw=2, color=(252/255,73/255,106/255), alpha=0.8, rasterized=True, zorder=1)
    plt.xlabel("True pChEMBL Value")
    plt.ylabel("Predicted pChEMBL Value")
    plt.grid(alpha=0.3)
    plt.text(0.05, 0.95, f"r = {test_corr:.2f}",
                transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
    # 保存图像
    plt.tight_layout()
    plt.savefig(f'./evaluation/test_r.pdf', dpi=1200)
    plt.close()

    print(f"Test -> ACC: {test_acc}, Sn: {test_sn}, Sp: {test_sp}, Precision: {test_prec}, F1: {test_f1}, AUC, {test_auc}, MCC: {test_mcc}, MSE: {test_mse}, R: {test_corr}")
    results.append([f"Test", test_acc, test_sn, test_sp, test_prec, test_f1, test_auc, test_mcc, test_mse, test_corr])

    # 保存评估结果至CSV
    df = pd.DataFrame(results, columns=["Type", "ACC", "Sn", "Sp", "Precision", "F1-score", "AUC", "MCC", 'MSE', 'R'])
    df = df.round(4)
    print(df)
    df.to_csv(f'{evaluation_results_dir}/evaluation_results.csv', index=False)

    plt.figure(figsize=(10, 8))
    plt.plot(train_fprs[0]*100, train_tprs[0]*100, lw=2, alpha=0.8, color=(81/255,161/255,93/255), label=f'Fold 1: AUC={train_aucs[0]:.4f}')
    plt.plot(train_fprs[1]*100, train_tprs[1]*100, lw=2, alpha=0.8, color=(254/255,55/255,149/255), label=f'Fold 2: AUC={train_aucs[1]:.4f}')
    plt.plot(train_fprs[2]*100, train_tprs[2]*100, lw=2, alpha=0.8, color=(176/255,74/255,70/255), label=f'Fold 3: AUC={train_aucs[2]:.4f}')
    plt.plot(train_fprs[3]*100, train_tprs[3]*100, lw=2, alpha=0.8, color=(255/255,172/255,55/255), label=f'Fold 4: AUC={train_aucs[3]:.4f}')
    plt.plot(train_fprs[4]*100, train_tprs[4]*100, lw=2, alpha=0.8, color=(198/255,26/255,165/255), label=f'Fold 5: AUC={train_aucs[4]:.4f}')
    plt.plot(train_fprs[5]*100, train_tprs[5]*100, lw=2, alpha=0.8, color=(18/255,15/255,253/255), label=f'Final Model: AUC={train_aucs[5]:.4f}')
    plot_detail('train')

    plt.figure(figsize=(10, 8))
    plt.plot(test_fprs[0]*100, test_tprs[0]*100, lw=2, alpha=0.8, color=(81/255,161/255,93/255), label=f'Fold 1: AUC={test_aucs[0]:.4f}')
    plt.plot(test_fprs[1]*100, test_tprs[1]*100, lw=2, alpha=0.8, color=(254/255,55/255,149/255), label=f'Fold 2: AUC={test_aucs[1]:.4f}')
    plt.plot(test_fprs[2]*100, test_tprs[2]*100, lw=2, alpha=0.8, color=(176/255,74/255,70/255), label=f'Fold 3: AUC={test_aucs[2]:.4f}')
    plt.plot(test_fprs[3]*100, test_tprs[3]*100, lw=2, alpha=0.8, color=(255/255,172/255,55/255), label=f'Fold 4: AUC={test_aucs[3]:.4f}')
    plt.plot(test_fprs[4]*100, test_tprs[4]*100, lw=2, alpha=0.8, color=(198/255,26/255,165/255), label=f'Fold 5: AUC={test_aucs[4]:.4f}')
    plt.plot(test_fprs[5]*100, test_tprs[5]*100, lw=2, alpha=0.8, color=(18/255,15/255,253/255), label=f'Final Model: AUC={test_aucs[5]:.4f}')
    plot_detail('test')

    plt.figure(figsize=(10, 8))
    plt.plot(train_fprs[5]*100, train_tprs[5]*100, lw=2, alpha=0.8, color=(18/255,15/255,253/255), label=f'Train: AUC={train_aucs[5]:.4f}')
    plt.plot(test_fprs[5]*100, test_tprs[5]*100, lw=2, alpha=0.8, color=(198/255,26/255,165/255), label=f'Test: AUC={test_aucs[5]:.4f}')
    plot_detail('compare')
