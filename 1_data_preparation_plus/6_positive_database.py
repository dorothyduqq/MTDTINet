import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')

pos_pre_short_data = pd.read_csv('./3_get_pos_datas/pos_pre_short_data.csv')
print(len(pos_pre_short_data))

positive_data = pos_pre_short_data[['compound_id', 'target_protein_id', 'pchembl']]

# 按照 'compound_id' 和 'target_protein_id' 分组，并计算 'pchembl' 的平均值
positive_data = positive_data.groupby(['compound_id', 'target_protein_id'], as_index=False).median()
print(len(positive_data))
positive_data = positive_data[positive_data['pchembl']>=4]
print(len(positive_data))
positive_data.to_csv('./6_positive_database/positive_database.csv', index=False)
