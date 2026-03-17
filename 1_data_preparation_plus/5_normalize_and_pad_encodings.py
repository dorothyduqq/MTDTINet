from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def standardize_column(df, column_name):
    min_vals = np.min(np.vstack(df[column_name]), axis=0)  # 按列计算最小值
    max_vals = np.max(np.vstack(df[column_name]), axis=0)  # 按列计算最大值
    # 保存标准化参数为 CSV 文件
    save_normalization_params(column_name, pd.Series(min_vals), pd.Series(max_vals), './5_normalize_and_pad_encodings')
    # 标准化每个二维数组的列
    updated_data = []
    for feature_vector in tqdm(df[column_name], total=len(df), desc='Standarizing...'):
        for j in range(feature_vector.shape[1]):  # 按列遍历
            # 获取每列的最小值和最大值
            min_val = min_vals[j]
            max_val = max_vals[j]
            feature_vector[:, j] = (feature_vector[:, j] - min_val) / (max_val - min_val)
        updated_data.append(feature_vector)
        # 将标准化结果重新赋值回原 dataframe
    df[column_name] = updated_data
    return df

def save_normalization_params(feature_name, min_vals, max_vals, path):
    params_df = pd.DataFrame({
        'feature': min_vals.index,
        'min': min_vals.values,
        'max': max_vals.values
    })
    params_df.to_csv(f"{path}/{feature_name}_normalization_params.csv", index=False)

def pad_array_rowwise(row, column_name, max_rows):
    array = row[column_name]
    if array.shape[0] < max_rows:
        padding_size = max_rows - array.shape[0]
        array = np.pad(array, ((0, padding_size), (0, 0)), mode='constant', constant_values=0)
    return array

def pad_array_square(row, column_name, max_size):
    array = row[column_name]
    if array.shape[0] < max_size:
        padding_size = max_size - array.shape[0]
        array = np.pad(array, ((0, padding_size), (0, padding_size)), mode='constant', constant_values=0)
    return array

drug_encodings = pd.read_pickle('./4_characterize_pos_datas/drug_encodings.pkl')
max_atom = drug_encodings['node_features'].apply(lambda x: x.shape[0]).max()
print(max_atom)

protein_encodings = pd.read_pickle('./4_characterize_pos_datas/protein_encodings.pkl')
max_aa = protein_encodings['onehot_encoding'].apply(lambda x: x.shape[0]).max()
print(max_aa)

# 绘制node_features数量分布直方图
node_counts = drug_encodings['node_features'].apply(lambda x: x.shape[0])

plt.figure(figsize=(10, 6))
plt.hist(node_counts, bins=range(0, int(node_counts.max()) + 10, 10), edgecolor='black')
plt.title('Distribution of Node Features Counts')
plt.xlabel('Number of Node Features')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('./5_normalize_and_pad_encodings/node_counts.pdf')
plt.close()

# 标准化 node_features 列
drug_encodings = standardize_column(drug_encodings, 'node_features')
# 使用apply函数对每一行进行填充
# drug_encodings['node_features'] = drug_encodings.apply(pad_array_rowwise, column_name='node_features', max_rows=max_atom, axis=1)
# drug_encodings['adj_matrix'] = drug_encodings.apply(pad_array_square, column_name='adj_matrix', max_size=max_atom, axis=1)
# 保存标准化后的 drug_encodings 数据为 CSV
drug_encodings.to_pickle('./5_normalize_and_pad_encodings/normalized_and_no_padded_drug_encodings.pkl')

# 处理 protein_encodings.pkl
protein_encodings = standardize_column(protein_encodings, 'blosum_encoding')
protein_encodings = standardize_column(protein_encodings, 'aaindex_encoding')

protein_encodings['onehot_encoding'] = protein_encodings.apply(pad_array_rowwise, column_name='onehot_encoding', max_rows=max_aa, axis=1)
protein_encodings['blosum_encoding'] = protein_encodings.apply(pad_array_rowwise, column_name='blosum_encoding', max_rows=max_aa, axis=1)
protein_encodings['aaindex_encoding'] = protein_encodings.apply(pad_array_rowwise, column_name='aaindex_encoding', max_rows=max_aa, axis=1)
protein_encodings['prottrans_encoding_per_residue'] = protein_encodings.apply(pad_array_rowwise, column_name='prottrans_encoding_per_residue', max_rows=max_aa, axis=1)

# 保存标准化后的 protein_encodings 数据为 CSV
protein_encodings.to_pickle('./5_normalize_and_pad_encodings/normalized_and_padded_protein_encodings.pkl')
