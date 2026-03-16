import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 输入输出文件路径
input_file = './0_database/chembl_uniprot_mapping.txt'
output_file = './1_get_prot_seqs/target_sequences.fasta'

# 先统计行数
with open(input_file, 'r') as infile:
    total_lines = sum(1 for _ in infile)

# 函数定义同上
def fetch_fasta_data(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    max_retries = 5
    retry_count = 0
    
    while retry_count < max_retries:
        response = requests.get(url)
        if response.ok:
            return uniprot_id, response.text
        else:
            retry_count += 1
            print(f"Error fetching {uniprot_id}. Retrying {retry_count}/{max_retries}...")
            time.sleep(0.1)
    
    print(f"Failed to fetch sequence for UniProt ID {uniprot_id} after {max_retries} retries.")
    return uniprot_id, None

# 打开文件并逐行读取所有的UniProt ID
with open(input_file, 'r') as infile:
    next(infile)  # 跳过标题行
    uniprot_ids = [line.strip().split('\t')[0] for line in infile]

# 使用ThreadPoolExecutor并行处理
with ThreadPoolExecutor(max_workers=64) as executor, open(output_file, 'w') as outfile:
    # 提交所有的任务
    future_to_id = {executor.submit(fetch_fasta_data, uniprot_id): uniprot_id for uniprot_id in uniprot_ids}
    
    # 使用tqdm显示进度条
    for future in tqdm(as_completed(future_to_id), total=total_lines-1):
        uniprot_id = future_to_id[future]
        try:
            uniprot_id, fasta_data = future.result()
            if fasta_data:
                outfile.write(fasta_data)
        except Exception as e:
            print(f"Exception occurred while processing {uniprot_id}: {e}")
