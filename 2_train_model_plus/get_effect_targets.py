import pandas as pd

data = pd.read_csv('./evaluation/evaluate_with_pos_data.csv', sep=',')

filtered_data = data[(data['Positive R'] >= 0.75) & (data['Positive MSE'] <= 0.8) & (data['ACC'] >= 0.9) & (data['Positive R P value'] < 0.001)]
filtered_data.to_csv('./evaluation/effect_targets.csv', index=False)