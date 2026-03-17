import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_max_pool

class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        # 第一个残差块的两层GINConv
        self.gin1_1 = GINConv(nn.Sequential(nn.Linear(7, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU()))
        self.gin1_2 = GINConv(nn.Sequential(nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU()))
        
        # 第二个残差块的两层GINConv
        self.gin2_1 = GINConv(nn.Sequential(nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU()))
        self.gin2_2 = GINConv(nn.Sequential(nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU()))
        
        # 用于残差连接的维度调整的GINConv
        self.residual1_gin = nn.Linear(7, 64)
        self.residual2_gin = nn.Linear(64, 128)
    
    def forward(self, x, edge_index, batch):
        identity1 = self.residual1_gin(x)
        out1 = self.gin1_1(x, edge_index)
        out1 = self.gin1_2(out1, edge_index)
        x = torch.add(out1, identity1)
        x = torch.relu(x)
        
        identity2 = self.residual2_gin(x)
        out2 = self.gin2_1(x, edge_index)
        out2 = self.gin2_2(out2, edge_index)
        x = torch.add(out2, identity2)
        x = torch.relu(x)
        x = global_max_pool(x, batch)
        return x

class ChannelAttention(nn.Module):
    """
    CBAM混合注意力机制的通道注意力
    """
    def __init__(self, in_channels, ratio):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels // ratio, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out * x

class SpatialAttention(nn.Module):
    """
    CBAM混合注意力机制的空间注意力
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(out))
        return out * x

class CBAM(nn.Module):
    """
    CBAM混合注意力机制
    """
    def __init__(self, in_channels, ratio, kernel_size):
        super(CBAM, self).__init__()
        self.channelattention = ChannelAttention(in_channels, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.channelattention(x)
        x = self.spatialattention(x)
        return x

class ProteinEncoder(nn.Module): 
    def __init__(self):
        super(ProteinEncoder, self).__init__()
        self.conv1_1 = nn.Conv1d(31, 64, kernel_size=3, padding=1)
        self.batch_norm1_1 = nn.BatchNorm1d(64)
        self.conv1_2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.batch_norm1_2 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.cbam1 = CBAM(64, ratio = 4, kernel_size = 3)

        self.conv2_1 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.batch_norm2_1 = nn.BatchNorm1d(128)
        self.conv2_2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.batch_norm2_2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.cbam2 = CBAM(128, ratio = 4, kernel_size = 3)

        self.conv3_1 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.batch_norm3_1 = nn.BatchNorm1d(256)
        self.conv3_2 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.batch_norm3_2 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.cbam3 = CBAM(256, ratio = 4, kernel_size = 3)

        self.residue1_conv = nn.Sequential(nn.Conv1d(31, 64, kernel_size=1, padding=0), nn.BatchNorm1d(64))
        self.residue1_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.residue2_conv = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, padding=0), nn.BatchNorm1d(128))
        self.residue2_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.residue3_conv = nn.Sequential(nn.Conv1d(128, 256, kernel_size=1, padding=0), nn.BatchNorm1d(256))
        self.residue3_pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        residual1 = self.residue1_conv(x)
        residual1 = self.residue1_pool(residual1)
        out1 = self.conv1_1(x)
        out1 = self.batch_norm1_1(out1)
        out1 = torch.relu(out1)
        out1 = self.conv1_2(out1)
        out1 = self.batch_norm1_2(out1)
        out1 = torch.relu(out1)
        out1 = self.pool1(out1)
        out1 = self.cbam1(out1)
        x = torch.add(out1, residual1)  # 残差连接
        x = torch.relu(x)

        residual2 = self.residue2_conv(x)
        residual2 = self.residue2_pool(residual2)
        out2 = self.conv2_1(x)
        out2 = self.batch_norm2_1(out2)
        out2 = torch.relu(out2)
        out2 = self.conv2_2(out2)
        out2 = self.batch_norm2_2(out2)
        out2 = torch.relu(out2)
        out2 = self.pool2(out2)
        out2 = self.cbam2(out2)
        x = torch.add(out2, residual2)  # 残差连接
        x = torch.relu(x)

        residual3 = self.residue3_conv(x)
        residual3 = self.residue3_pool(residual3)
        out3 = self.conv3_1(x)
        out3 = self.batch_norm3_1(out3)
        out3 = torch.relu(out3)
        out3 = self.conv3_2(out3)
        out3 = self.batch_norm3_2(out3)
        out3 = torch.relu(out3)
        out3 = self.pool3(out3)
        out3 = self.cbam3(out3)
        x = torch.add(out3, residual3)  # 残差连接
        x = torch.relu(x)

        x = self.global_max_pool(x).squeeze(-1)
        return x

# class DrugProteinModel(nn.Module):
#     def __init__(self):
#         super(DrugProteinModel, self).__init__()
#         self.gnn = GNN()
#         self.protein_encoder = ProteinEncoder()
#         self.fc1 = nn.Linear(256 + 128, 128)
#         self.bn1 = nn.BatchNorm1d(128)
#         self.fc2 = nn.Linear(128, 64)
#         self.bn2 = nn.BatchNorm1d(64)
#         self.fc3 = nn.Linear(64, 2)
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, drug_data, protein_data):
#         drug_features = self.gnn(drug_data.x, drug_data.edge_index, drug_data.batch)
#         protein_features = self.protein_encoder(protein_data)
#         combined_features = torch.cat((drug_features, protein_features), dim=1)

#         x = self.fc1(combined_features)
#         x = self.bn1(x)
#         x = torch.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.bn2(x)
#         x = torch.relu(x)
#         x = self.dropout(x)
#         output = self.fc3(x)
#         output_class = torch.sigmoid(output[:, 0])  # 分类任务
#         output_reg = torch.relu(output[:, 1])  # 回归任务
#         return output_class, output_reg

class DrugProteinModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn = GNN()
        self.protein_encoder = ProteinEncoder()
        
        # 共享的特征提取层
        self.fc1 = nn.Linear(256 + 128, 128)
        self.bn_shared = nn.BatchNorm1d(128)  # 共享的批归一化
        
        # 分类任务分支（输出0-1概率）
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),  # 输出单节点用于二分类
            nn.Sigmoid()  # 直接在这里添加Sigmoid
        )
        
        # 回归任务分支（输出非负值）
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),  # 回归任务通常需要更少正则化
            nn.Linear(64, 1),
            # nn.Softplus()  # 确保输出非负
        )

    def forward(self, drug_data, protein_data):
        # 特征提取
        drug_features = self.gnn(drug_data.x, drug_data.edge_index, drug_data.batch)
        protein_features = self.protein_encoder(protein_data)
        combined_features = torch.cat((drug_features, protein_features), dim=1)
        
        # 共享层处理
        shared_out = self.fc1(combined_features)
        shared_out = self.bn_shared(shared_out)
        shared_out = torch.relu(shared_out)
        
        # 并行分支处理
        class_output = self.classifier(shared_out).squeeze(-1)  # 形状 [batch_size]
        reg_output = self.regressor(shared_out).squeeze(-1)     # 形状 [batch_size]
        
        return class_output, reg_output