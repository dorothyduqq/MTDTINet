import pandas as pd

data = pd.read_csv('./evaluation/evaluate_with_pos_data.csv', sep=',')

filtered_data = data[(data['ACC'] >= 0.9)]
filtered_data.to_csv('./evaluation/effect_classification_targets.csv', index=False)