import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.data import Batch
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from tqdm import tqdm
from model_structure import DrugProteinModel
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
torch.autograd.set_detect_anomaly(True)

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
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=4, prefetch_factor=4, pin_memory=True, persistent_workers=True)

def collate_fn(batch):
    drug_graphs, aaindex_encodings, labels = zip(*batch)
    drug_graphs = Batch.from_data_list(drug_graphs)
    aaindex_encodings = torch.stack(aaindex_encodings).permute(0, 2, 1)
    labels = torch.tensor(labels)
    return drug_graphs, aaindex_encodings, labels

def cold_start_split(data, num_groups=5):
    # 按化合物频率降序排序
    compound_counts = data['compound_id'].value_counts()
    sorted_compounds = compound_counts.index.tolist()

    # 创建分组
    compound_groups = [[] for _ in range(num_groups)]
    for i, compound in enumerate(sorted_compounds):
        group_index = i % num_groups
        compound_groups[group_index].append(compound)
    
    # 准备训练和验证集
    train_compounds = []
    val_compounds = []
    for i in range(num_groups):
        # 选择当前组作为验证集，其他组作为训练集
        val_group = compound_groups[i]
        train_group = [c for j, group in enumerate(compound_groups) if j != i for c in group]
        
        train_compounds.append(train_group)
        val_compounds.append(val_group)
    
    return train_compounds, val_compounds

def masked_huber_loss(y_pred, y_true, delta=1.0):
    mask = (y_true >= 4).float()  # 掩码：标签为2-11
    # 计算预测值与真实值之间的差异
    diff = torch.abs(y_pred - y_true)
    # Huber损失计算
    huber_loss = torch.where(
        diff <= delta,
        0.5 * diff ** 2,
        delta * (diff - 0.5 * delta)
    )
    # 应用掩码并计算平均损失
    mask_sum = torch.sum(mask)
    if mask_sum == 0:  # 无正样本的情况
        return torch.tensor(0.0, device=y_pred.device), 0  # 返回 0 或其他默认值
    masked_loss = torch.sum(huber_loss * mask) / mask_sum  # 只计算正样本
    return masked_loss, mask_sum

# def masked_mse_loss(y_pred, y_true):
#     mask = (y_true >= 4).float()  # 掩码：只计算 y_true >= 4 的样本
#     mse_loss = (y_pred - y_true) ** 2  # 计算 MSE 损失
#     mask_sum = torch.sum(mask)

#     if mask_sum == 0:  # 如果没有符合条件的样本
#         return torch.tensor(0.0, device=y_pred.device), 0  
    
#     masked_loss = torch.sum(mse_loss * mask) / mask_sum  # 只计算正样本
#     return masked_loss, mask_sum

# 训练和验证函数（与之前代码保持一致）
def train_and_evaluate_model(model, train_loader, val_loader, optimizer, patience, num_epochs, delta):
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_total_loss, train_class_loss, train_reg_loss, train_count_reg = 0.0, 0.0, 0.0, 0.0
        train_outputs_class, train_outputs_reg, train_labels_all = [], [], []
        
        # for drug_data, protein_data, labels in tqdm(train_loader, desc=f'Training Epoch {epoch}'):
        for drug_data, protein_data, labels in train_loader:
            drug_data = drug_data.to(device)
            protein_data = protein_data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output_class, output_reg = model(drug_data, protein_data)
            # print(min(output_reg.detach().cpu().numpy()))
            loss_class = nn.BCELoss()(output_class, (labels >= 4).float()) # 分类损失
            loss_reg, count_reg = masked_huber_loss(output_reg, labels, delta = delta) # 回归损失（掩码机制）
            total_loss = 0.2 * loss_class + 0.8 * loss_reg # 总损失
            total_loss.backward()
            optimizer.step()
            train_total_loss += total_loss.item() * labels.size(0)
            train_class_loss += loss_class.item() * labels.size(0)
            train_reg_loss += loss_reg.item() * count_reg
            train_count_reg += count_reg
            
            train_outputs_class.append(output_class.detach())
            train_outputs_reg.append(output_reg.detach())
            train_labels_all.append(labels)
        
        train_total_loss /= len(train_loader.dataset)
        train_class_loss /= len(train_loader.dataset)
        train_reg_loss /= train_count_reg # 除以正样本的数量
        train_outputs_class = torch.cat(train_outputs_class).cpu().numpy()
        train_outputs_reg = torch.cat(train_outputs_reg).cpu().numpy()
        train_labels_all = torch.cat(train_labels_all).cpu().numpy()

        train_predictions_binary = (train_outputs_class >= 0.5).astype(int)  # 分类阈值为 0.5
        train_labels_binary = (train_labels_all >= 4).astype(int)  # 标签 >= 4 为正样本
        train_accuracy = (train_predictions_binary == train_labels_binary).mean()

        train_mask = train_labels_all >= 4
        train_outputs_pos = train_outputs_reg[train_mask]
        train_labels_pos = train_labels_all[train_mask]
        train_mse_pos = mean_squared_error(train_labels_pos, train_outputs_pos)
        train_corr_pos, _ = pearsonr(train_labels_pos, train_outputs_pos)

        # 验证集
        model.eval()
        val_total_loss, val_class_loss, val_reg_loss, val_count_reg = 0.0, 0.0, 0.0, 0.0
        val_outputs_class, val_outputs_reg, val_labels_all = [], [], []
        
        with torch.no_grad():
            # for drug_data, protein_data, labels in tqdm(val_loader, desc=f'Val Epoch {epoch}'):
            for drug_data, protein_data, labels in val_loader:
                drug_data = drug_data.to(device)
                protein_data = protein_data.to(device)
                labels = labels.to(device)
                output_class, output_reg = model(drug_data, protein_data)
                # print(min(output_reg.detach().cpu().numpy()))
                loss_class = nn.BCELoss()(output_class, (labels >= 4).float()) # 分类损失
                loss_reg, count_reg = masked_huber_loss(output_reg, labels, delta=delta) # 回归损失（掩码机制）
                total_loss = 0.2 * loss_class + 0.8 * loss_reg # 总损失
                val_total_loss += total_loss.item() * labels.size(0)
                val_class_loss += loss_class.item() * labels.size(0)
                val_reg_loss += loss_reg.item() * count_reg
                val_count_reg += count_reg
                
                val_outputs_class.append(output_class.detach())
                val_outputs_reg.append(output_reg.detach())
                val_labels_all.append(labels)
            
            val_total_loss /= len(val_loader.dataset)
            val_class_loss /= len(val_loader.dataset)
            val_reg_loss /= val_count_reg
            val_outputs_class = torch.cat(val_outputs_class).cpu().numpy()
            val_outputs_reg = torch.cat(val_outputs_reg).cpu().numpy()
            val_labels_all = torch.cat(val_labels_all).cpu().numpy()

            val_predictions_binary = (val_outputs_class >= 0.5).astype(int)  # 分类阈值为 0.5
            val_labels_binary = (val_labels_all >= 4).astype(int)  # 标签 >= 4 为正样本
            val_accuracy = (val_predictions_binary == val_labels_binary).mean()

            val_mask = val_labels_all >= 4
            val_outputs_pos = val_outputs_reg[val_mask]
            val_labels_pos = val_labels_all[val_mask]
            val_mse_pos = mean_squared_error(val_labels_pos, val_outputs_pos)
            val_corr_pos, _ = pearsonr(val_labels_pos, val_outputs_pos)
        
        # 打印结果
        print(f"Epoch {epoch}\nTrain: Total Loss: {train_total_loss:.4f}, Class Loss: {train_class_loss:.4f}, Reg Loss: {train_reg_loss:.4f}, Accuracy: {train_accuracy:.4f}, MSE: {train_mse_pos:.4f}, R: {train_corr_pos:.4f}\n"
              f"Val: Total Loss: {val_total_loss:.4f}, Class Loss: {val_class_loss:.4f}, Reg Loss: {val_reg_loss:.4f}, Accuracy: {val_accuracy:.4f}, MSE: {val_mse_pos:.4f}, R: {val_corr_pos:.4f}\n", flush=True)

        # 早停
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break
        
        torch.cuda.empty_cache()

    return best_model_state

if __name__ == '__main__':
    # 可调参数设置
    batch_size = 256
    num_epochs = 500
    learning_rate = 0.005
    patience = 10
    delta = 1
    # node_feat_dim = 7
    # protein_feat_dim = 31
    # hidden_dim = 256
    # output_dim = 2
    k_folds = 5

    # 路径设置
    train_data_path = '../1_data_preparation_plus/9_split_database/training_dataset.csv'
    drug_encodings_path = '../1_data_preparation_plus/5_normalize_and_pad_encodings/normalized_and_no_padded_drug_encodings.pkl'
    protein_encodings_path = '../1_data_preparation_plus/5_normalize_and_pad_encodings/normalized_and_padded_protein_encodings.pkl'
    model_save_dir = './best_models'
    os.makedirs(model_save_dir, exist_ok=True)

    # 读取数据
    train_data = pd.read_csv(train_data_path)
    drug_encodings = pd.read_pickle(drug_encodings_path)
    protein_encodings = pd.read_pickle(protein_encodings_path)

    # 数据合并
    train_data = train_data.merge(drug_encodings, on='compound_id').merge(protein_encodings, on='target_protein_id')
    # train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
    # train_data = train_data.head(1000)

    # 冷启动交叉验证
    train_compounds, val_compounds = cold_start_split(train_data, num_groups=k_folds)

    for fold in range(k_folds):
        if fold not in [3]:
            continue
        print(f"Fold {fold + 1}", flush=True)
        
        # 根据化合物分组筛选数据索引
        train_mask = train_data['compound_id'].isin(train_compounds[fold])
        val_mask = train_data['compound_id'].isin(val_compounds[fold])
        
        # 根据筛选结果创建动态数据集
        train_subset = train_data[train_mask]
        val_subset = train_data[val_mask]
        
        train_loader = create_dataloader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = create_dataloader(val_subset, batch_size=batch_size, shuffle=True)

        # 模型初始化
        model = DrugProteinModel().to(device) # node_feat_dim=node_feat_dim, hidden_dim=hidden_dim, protein_feat_dim=protein_feat_dim, output_dim=output_dim
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 训练和保存模型
        best_model_state = train_and_evaluate_model(model, train_loader, val_loader, optimizer, patience, num_epochs, delta)
        
        model_path = os.path.join(model_save_dir, f"best_model_fold_{fold + 1}.pth")
        torch.save(best_model_state, model_path)
