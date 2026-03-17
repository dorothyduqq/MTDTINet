import pandas as pd
import numpy as np
from tqdm import tqdm
from Bio.Align import substitution_matrices
from transformers import T5Tokenizer, T5EncoderModel
import torch
from multiprocessing import Pool

standard_amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
aaindex_amino_acids='ARNDCQEGHILKMFPSTWYV'

# 初始化one-hot向量
one_hot_dict = {aa: i for i, aa in enumerate(standard_amino_acids)}
# 初始化BLOSUN62矩阵
blosum62 = substitution_matrices.load("BLOSUM62")
# 初始化AAindex索引
aaindex=pd.read_table('./4_characterize_pos_datas/aaindex31',sep='\t',header=None)
aa=[x for x in aaindex_amino_acids]
aaindex=aaindex.to_numpy().T
index={x:y for x,y in zip(aa,aaindex.tolist())}
# 初始化ProtTrans
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('../tools/prot_t5_xl_uniref50', do_lower_case=False)
# Load the model
model = T5EncoderModel.from_pretrained("../tools//prot_t5_xl_uniref50").to(device)
# only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
model.to(torch.float32) if device==torch.device("cpu") else model.to(torch.float64)

def one_hot_encode(sequence):
    # 初始化零矩阵
    encoding = np.zeros((len(sequence), len(standard_amino_acids)))
    # 对序列中的每个氨基酸进行编码
    for i, aa in enumerate(sequence):
        if aa in one_hot_dict:
            encoding[i, one_hot_dict[aa]] = 1
    return encoding

def blosum62_encode(sequence):
    # 创建编码矩阵
    encoding = np.zeros((len(sequence), len(standard_amino_acids)))
    # 使用BLOSUM62矩阵提取特征
    for i, aa in enumerate(sequence):
        encoding[i] = np.array([blosum62[aa][aa2] for aa2 in standard_amino_acids])
    return encoding

def aaindex_encode(sequence):
    encoding = np.zeros((len(sequence), 31))
    for i, x in enumerate(sequence):
        encoding[i] = index[x]
    return encoding

def prottrans_encode(sequence):
    # introduce white-space between all amino acids
    pad_sequence = [" ".join(sequence)]
    # tokenize sequences and pad up to the longest sequence in the batch
    ids = tokenizer(pad_sequence, add_special_tokens=True) # , padding="longest"
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    # generate embeddings
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
    # extract residue embeddings for the first ([0,:]) sequence in the batch and remove padded & special tokens ([0,:7]) 
    embedding_per_residue = embedding_repr.last_hidden_state[0,:len(sequence)].numpy() # shape (7 x 1024)
    embedding_per_protein = np.mean(embedding_per_residue, axis=0) # shape (1024)
    return embedding_per_residue, embedding_per_protein

# 定义处理每个序列的任务函数
def process_sequence(sequence_info):
    sequence, tp_id = sequence_info
    onehot_encoding = one_hot_encode(sequence)
    blosum_encoding = blosum62_encode(sequence)
    aaindex_encoding = aaindex_encode(sequence)
    prottrans_encoding_per_residue, prottrans_encoding_per_protein = prottrans_encode(sequence)
    return {
        'target_protein_id': tp_id,
        'onehot_encoding': onehot_encoding,
        'blosum_encoding': blosum_encoding,
        'aaindex_encoding': aaindex_encoding,
        'prottrans_encoding_per_residue': prottrans_encoding_per_residue,
        'prottrans_encoding_per_protein': prottrans_encoding_per_protein
    }

# 读取CSV文件并准备数据
df = pd.read_csv('./3_get_pos_datas/pos_pre_short_data.csv')
sequences_info = [
    (sequence, df.loc[df['target_seq'] == sequence, 'target_protein_id'].tolist()[0])
    for sequence in df['target_seq'].unique()
]

# 使用多进程处理
if __name__ == "__main__":
    # 设置进程池大小
    pool_size = 16  # 可根据实际情况调整进程数
    with Pool(pool_size) as pool:
        results = list(tqdm(pool.imap(process_sequence, sequences_info), total=len(sequences_info)))

    # 保存结果
    df_results = pd.DataFrame(results)
    df_results.to_pickle('./4_characterize_pos_datas/protein_encodings.pkl')