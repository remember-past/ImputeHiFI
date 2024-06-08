import os
import time
from subprocess import  call
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sc
from matplotlib.colors import ListedColormap
import scipy.sparse as sparse
import importlib
import copy
import gc
import datetime
from scipy.stats import zscore
import sys
import scipy.stats as st
import math
from scipy import optimize
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
import scanpy as sc
import warnings
warnings.filterwarnings('ignore')

src_dir=r"/Path/to/ImputeHiFI/src"#r"/Path/to/ImputeHiFI/src"
sys.path.append(src_dir)
from function_basic import basic_function as basic
import ImputeHiFI_Algorithm as ImputeHiFI


def convert_array_to_bin_pair_df(chr_name,bin_length,one_cell_chr_array,data=0):
    matrix = np.ones((bin_length, bin_length))
    matrix = np.triu(matrix, k=1)

    from scipy.sparse import coo_matrix
    matrix_sparse = coo_matrix(matrix)

    df = pd.DataFrame({'col': matrix_sparse.col, 'row': matrix_sparse.row, 'data': matrix_sparse.data})

    df['col'] = df['col'].astype(str)
    df['row'] = df['row'].astype(str)
    df['data']=data
    df['chr_bin_bin']=chr_name+'_'+df['row']+'_'+df['col']
    Example_df=df.copy()
    Example_df.index=Example_df['chr_bin_bin']

    one_cell_chr_array=np.triu(one_cell_chr_array, k=1)
    one_cell_chr_array_sparse = coo_matrix(one_cell_chr_array)

    df = pd.DataFrame({'col': one_cell_chr_array_sparse.col, 'row': one_cell_chr_array_sparse.row, 'data': one_cell_chr_array_sparse.data})
    df['col'] = df['col'].astype(str)
    df['row'] = df['row'].astype(str)
    df['chr_bin_bin']=chr_name+'_'+df['row']+'_'+df['col']
    one_cell_chr_df=df.copy()
    one_cell_chr_df.index=one_cell_chr_df['chr_bin_bin']
    Example_df.loc[one_cell_chr_df.index,'data']=one_cell_chr_df['data'].values
    return Example_df
def generate_Example_bin_pair_df(chr_name,bin_length,data=0):
    matrix = np.ones((bin_length, bin_length))
    matrix = np.triu(matrix, k=1)

    from scipy.sparse import coo_matrix
    matrix_sparse = coo_matrix(matrix)

    df = pd.DataFrame({'col': matrix_sparse.col, 'row': matrix_sparse.row, 'data': matrix_sparse.data})
    df['data']=data
    df['col'] = df['col'].astype(str)
    df['row'] = df['row'].astype(str)
    df['chr_bin_bin']=chr_name+'_'+df['row']+'_'+df['col']
    Example_df=df.copy()
    Example_df.index=Example_df['chr_bin_bin']
    return Example_df

def convert_bin_pair_df_to_array(bin_length, Example_df):
    # 将'chr_bin_bin'列分解为'chr_name', 'row', 'col'
    Example_df[['chr_name', 'row', 'col']] = Example_df['chr_bin_bin'].str.split('_', expand=True)

    # 将'row'和'col'列的数据类型转换为整数，以便用于创建coo_matrix
    Example_df['row'] = Example_df['row'].astype(int)
    Example_df['col'] = Example_df['col'].astype(int)

    # 使用'row'，'col'，和'data'列创建coo_matrix
    from scipy.sparse import coo_matrix
    matrix_sparse = coo_matrix((Example_df['data'], (Example_df['row'], Example_df['col'])), shape=(bin_length, bin_length))

    # 将coo_matrix转换为数组
    array = matrix_sparse.toarray()

    return array
def generate_bin_pair_df_one_chr(chr_name,cell_name_array,one_cell_name_same_leiden_type_dict,FISH_nearest_neighbors_names_dic):
    print(chr_name)

    FISH_RMSD_nearest_neighbors_names_dic={}
    FISH_RMSD_nearest_neighbors_names_dic[chr_name]={}
    for one_cell_name in tqdm(cell_name_array):
        cells_with_same_leiden_type=one_cell_name_same_leiden_type_dict[one_cell_name]
        candidate_array=FISH_nearest_neighbors_names_dic[chr_name][one_cell_name]
        used_array=candidate_array[np.isin(candidate_array,cells_with_same_leiden_type)]
        FISH_RMSD_nearest_neighbors_names_dic[chr_name][one_cell_name]=used_array

    all_result={}
    all_result['nearest_neighbors_names_dic']=FISH_RMSD_nearest_neighbors_names_dic
    chr_all_result={}
    chr_all_result[chr_name]=all_result
    return chr_name,chr_all_result

data_dir=r"/mnt/disk1/scfan/data/Cai_21_bioRxiv"
if __name__ == '__main__':
    similar_FISH_cell_path = os.path.join(
        r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/ImputHiFI/first_seen_probe_data",
        "All_cell_RMSD.pkl")
    FISH_nearest_neighbors_dic = basic.load_variable_from_pikle_file(similar_FISH_cell_path)
    FISH_nearest_neighbors_names_dic = FISH_nearest_neighbors_dic['nearest_neighbors_names_dic']

    data_dir = r"/mnt/disk1/scfan/data/Cai_21_bioRxiv"
    adata = ImputeHiFI.get_RNA_FISH_leiden(data_dir)


    cell_name_array = adata.obs_names
    one_cell_name_same_leiden_type_dict = {}
    for one_cell_name in tqdm(cell_name_array):
        leiden_type_of_one_cell = adata.obs.loc[adata.obs['cell_name'] == one_cell_name, 'leiden'].values[0]
        # 找到所有具有相同 leiden 类型的细胞
        cells_with_same_leiden_type = adata.obs[adata.obs['leiden'] == leiden_type_of_one_cell]['cell_name'].values
        one_cell_name_same_leiden_type_dict[one_cell_name] = cells_with_same_leiden_type

    temp_path = os.path.join(r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/ImputHiFI", "first_seen_probe_data.pkl")
    data = basic.load_variable_from_pikle_file(temp_path)

    chr_length_file = r"/mnt/disk1/scfan/data/Genome/mm10_chr_length.txt"
    resolution_name = '1Mb'
    resolution = 1000000
    chr_length_dic = basic.get_chr_length_info(chr_length_file)
    chr_bin_length_dic = {}
    for one_chr in chr_length_dic.keys():
        if one_chr != 'chrY' and one_chr != 'chrX':
            bin_length = chr_length_dic[one_chr] // resolution + 1
            chr_bin_length_dic[one_chr] = bin_length

    chr_name_list=list(chr_bin_length_dic.keys())
    kernerl_number=19
    chr_name_split_list = np.array_split(chr_name_list, kernerl_number)
    store_dir = r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/ImputHiFI/first_seen_probe_data"
    import multiprocessing as mp
    num_threads = kernerl_number

    FISH_name_list = []
    for ids in data['chrom_ids']:
        FISH_name_list.append(str(ids))
    FISH_name_array = np.array(FISH_name_list)



    _domain_args = [( chr_name_part[0], cell_name_array,
                      one_cell_name_same_leiden_type_dict, FISH_nearest_neighbors_names_dic)
                    for chr_name_part in chr_name_split_list]
    with mp.Pool(num_threads) as domain_pool:
        domain_results = domain_pool.starmap(generate_bin_pair_df_one_chr, _domain_args)
        domain_pool.close()
        domain_pool.join()
        domain_pool.terminate()


    result_dic = {}

    result_dic['nearest_neighbors_names_dic']={}

    for result in domain_results:
        print("Collect result")
        chr_name,one_chr_corr_dict=result
        corr_dict=one_chr_corr_dict[chr_name]
        for key,item in corr_dict.items():
            result_dic[key][chr_name]=item[chr_name]

    temp_path = os.path.join(store_dir, "All_cell_RNA_FISH_RMSD.pkl")
    # temp_path = r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/ImputHiFI/first_seen_probe_data_Impute/ImputeHiFI_v2_proximity_score_adata_"+str(para)+".h5ad"
    basic.store_variable_from_pikle_file(temp_path, result_dic)