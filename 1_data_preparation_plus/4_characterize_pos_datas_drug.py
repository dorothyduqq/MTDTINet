from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

# 读取CSV文件
df = pd.read_csv('./3_get_pos_datas/pos_pre_short_data.csv')

# 初始化Morgan指纹生成器
morgan_generator = GetMorganGenerator(radius=2, fpSize=1024)

# 定义一个函数用于计算节点特征
def get_atom_features(atom):
    return {
        'atomic_num': atom.GetAtomicNum(),
        'degree': atom.GetDegree(),
        'num_hs': atom.GetTotalNumHs(),
        'implicit_valence': atom.GetImplicitValence(),
        'is_aromatic': int(atom.GetIsAromatic()),
        'mass': atom.GetMass(),
        'is_in_ring': int(atom.IsInRing())
    }

# 定义用于处理单个SMILES的函数
def process_smiles(smiles):
    result = {}
    try:
        # 获取当前SMILES对应的compound_id
        compound_id = df.loc[df['canonical_smiles'] == smiles, 'compound_id'].tolist()[0]
        
        # 生成分子对象
        mol = Chem.MolFromSmiles(smiles)
        
        # 节点特征矩阵
        node_features = np.array([list(get_atom_features(atom).values()) for atom in mol.GetAtoms()])
        
        # 邻接矩阵
        adj_matrix = Chem.GetAdjacencyMatrix(mol)
        
        # 生成Morgan指纹
        fingerprint = morgan_generator.GetFingerprint(mol)
        fingerprint_list = np.array(fingerprint)  # 转换为NumPy数组以方便存储
        
        # 保存结果
        result = {
            'compound_id': compound_id,
            'node_features': node_features,
            'adj_matrix': adj_matrix,
            'fingerprint': fingerprint_list
        }
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
    
    return result

# 主函数：使用多进程处理SMILES
if __name__ == '__main__':
    # 获取所有唯一的SMILES
    smiles_list = df['canonical_smiles'].unique()
    
    # 使用进程池并行处理
    with mp.Pool(128) as pool:
        results = list(tqdm(pool.imap(process_smiles, smiles_list), total=len(smiles_list)))

    # 过滤空结果
    results = [res for res in results if res]

    # 保存到DataFrame并写入文件
    df_results = pd.DataFrame(results)
    df_results.to_pickle('./4_characterize_pos_datas/drug_encodings.pkl')

# df_results = pd.read_pickle('./4_characterize_pos_datas/drug_encodings.pkl')
