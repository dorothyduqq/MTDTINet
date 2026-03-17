import pandas as pd

def expand_rows(df, columns_to_split):
    expanded_rows = []
    for _, row in df.iterrows():
        split_values = [str(row[col]).split('；') for col in columns_to_split]
        max_length = max(map(len, split_values))
        for values in zip(*[val if len(val) == max_length else val * max_length for val in split_values]):
            new_row = row.copy()
            for col, value in zip(columns_to_split, values):
                new_row[col] = value
            expanded_rows.append(new_row)
    return pd.DataFrame(expanded_rows)

def merge_data(df_expanded, pos_file, final_output):
    df_pos = pd.read_csv(pos_file, usecols=['target_protein_id', 'target_seq']).drop_duplicates()
    
    # 合并target_protein_id
    df_merged_protein = df_expanded.merge(df_pos, on='target_protein_id', how='left')
    
    df_merged_protein.to_csv(final_output, index=False)

def process_csv(file_path, pos_file, final_output):
    df = pd.read_csv(file_path)
    df_filtered = df[df['max_phase'].notna()]
    columns_to_expand = ['output_class', 'output_reg', 'target_protein_id', 'target_chembl_id', 'uniprot_id']
    df_expanded = expand_rows(df_filtered, columns_to_expand)    
    # 进行合并
    merge_data(df_expanded, pos_file, final_output)

# 使用示例
input_file = "./3_get_potential_interaction/interaction_results_9.csv"
pos_file = "../1_data_preparation_plus/3_get_pos_datas/pos_pre_short_data.csv"
final_output = "./3_get_potential_interaction/drug_interaction.csv"
process_csv(input_file, pos_file, final_output)
