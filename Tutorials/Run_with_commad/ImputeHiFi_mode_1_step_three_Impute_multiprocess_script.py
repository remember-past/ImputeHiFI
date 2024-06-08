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
from scipy.spatial import cKDTree

import warnings
warnings.filterwarnings('ignore')
src_dir=r"/Path/to/ImputeHiFI/src"#r"/Path/to/ImputeHiFI/src"
sys.path.append(src_dir)
from function_basic import basic_function as basic
import ImputeHiFI_Algorithm as ImputeHiFI


def generate_bin_pair_df_one_chr(data,chr_name,store_dir,chr_bin_length_dic):

    # store_dir = r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/ImputHiFI/first_seen_probe_data"
    used_chr_name_list = data['chrom_name'][0:19]
    chr_index = used_chr_name_list.index(chr_name)
    cell_number = len(data['chrom_ids'])
    all_cell_dna_list = data['all_cell_dna_list']
    print(chr_name)
    dna_zxys=[]
    distmap_list_dic = {}
    for cell_index in tqdm(range(cell_number)):
        dna_zxys.append(all_cell_dna_list[cell_index][chr_index])
    distmap_list = Cai21BioRxivPreprocess.GetPassedData.get_distmap_list_from_dna_zxys(dna_zxys)
    distmap_list_dic[chr_name]=distmap_list

    all_cell_df_dic = {}
    for chr_name in distmap_list_dic.keys():
        print(chr_name)
        # chr_name='chr1'
        bin_length = chr_bin_length_dic[chr_name]
        Example_bin_pair_df = generate_Example_bin_pair_df(chr_name, bin_length)
        all_cell_one_chr_df = pd.DataFrame(index=Example_bin_pair_df.index)
        for cell_index, cell_ids in tqdm(enumerate(data['chrom_ids'])):
            one_cell_chr_array = distmap_list_dic[chr_name][cell_index].copy()
            one_cell_chr_df = convert_array_to_bin_pair_df(chr_name, bin_length, one_cell_chr_array)
            cell_name = str(cell_ids)
            all_cell_one_chr_df.loc[one_cell_chr_df.index, cell_name] = one_cell_chr_df['data'].values
        all_cell_df_dic[chr_name] = all_cell_one_chr_df

    temp_path=os.path.join(store_dir,chr_name+"_spatial_dist_bin_pair_df.pkl")
    basic.store_variable_from_pikle_file(temp_path,all_cell_df_dic)




def generate_Impute_result(chr_bin_length_dic, base_coef,short_range_HiC_weight_bound,
                      long_range_HiC_weight_bound,all_range_diag_sum_weight_HiC_bound,
                        short_range_diag,background_cell_num,
                               HiC_RNA_correlations_df, DNA_FISH_RNA_HiC_dic_name_tuple,
                               DNA_FISH_name_list, missing_rate_df, DNA_FISH_adata,
                               FISH_nearest_neighbors_names_dic,use_mean_count):
    Impute_DNA_FISH_df_dict = {}
    for one_FISH_name in tqdm(DNA_FISH_name_list):
        one_RNA_name, multi_HiC_name_list = DNA_FISH_RNA_HiC_dic_name_tuple[one_FISH_name]
        one_cell_DNA_RNA_confidence = 1
        multi_HiC_FISHconfid_array = np.array([one_cell_DNA_RNA_confidence] * len(multi_HiC_name_list))

        multi_HiC_RNAcoef_array = HiC_RNA_correlations_df.loc[multi_HiC_name_list, one_RNA_name].values
        multi_HiC_RNAcoef_array[multi_HiC_RNAcoef_array <= 0] = base_coef
        weight_multi_HiC_RNAcoef_array = multi_HiC_RNAcoef_array / np.sum(multi_HiC_RNAcoef_array)
        #     print(multi_HiC_RNAcoef_array,weight_multi_HiC_RNAcoef_array)
        #     break

        # chr_name='chr1'
        Impute_df_dict = {}
        for chr_name in chr_bin_length_dic.keys():
            DNA_FISH_df, diag_sum_prop, DNA_FISH_all_count, Common_Example_df, diag_array, bin_length = ImputeHiFI.generate_DNA_FISH_df(
                one_FISH_name, chr_name, missing_rate_df, DNA_FISH_adata)

            HiC_df, HiC_diag_sum_prop_dict = ImputeHiFI.generate_HiC_df(diag_array, Common_Example_df,
                                                                        multi_HiC_name_list, chr_name, bin_length,
                                                                        weight_multi_HiC_RNAcoef_array, HiC_adata)

            bg_df, bg_diag_sum_prop_dict, bg_all_count = ImputeHiFI.generate_background_df(diag_array,
                                                                                           Common_Example_df,
                                                                                           one_FISH_name, chr_name,
                                                                                           bin_length, DNA_FISH_adata,
                                                                                           FISH_nearest_neighbors_names_dic,
                                                                                           background_cell_num)

            DNA_Impute_FISH_df, diag_learning_weight_dict, diag_learning_weight_for_diag_sum_dict, diag_init_loss, diag_learning_loss = ImputeHiFI.optimize_and_impute_data(
                diag_array, Common_Example_df, DNA_FISH_df, HiC_df, bg_df,
                initial_guess, short_range_diag, short_range_HiC_weight_bound,
                long_range_HiC_weight_bound, all_range_diag_sum_weight_HiC_bound,
                HiC_diag_sum_prop_dict, bg_diag_sum_prop_dict, bg_all_count,use_mean_count)
            Impute_df_dict[chr_name] = DNA_Impute_FISH_df

        #         one_array=ImputeHiFI.convert_bin_pair_df_to_array(bin_length,DNA_FISH_df)
        #         cmap = plt.cm.seismic
        #         cmap.set_bad('#8a898a')
        #         sns.heatmap(one_array,cmap=cmap)
        #         plt.title(chr_name)
        #         plt.show()

        #         one_array=ImputeHiFI.convert_bin_pair_df_to_array(bin_length,DNA_Impute_FISH_df)
        #         cmap = plt.cm.seismic
        #         cmap.set_bad('#8a898a')
        #         sns.heatmap(one_array,cmap=cmap,vmax=1)
        #         plt.title(chr_name)
        #         plt.show()
        #         break
        one_cell_Impute_df = pd.concat(Impute_df_dict.values(), axis=0)
        Impute_DNA_FISH_df_dict[one_FISH_name] = one_cell_Impute_df
    return Impute_DNA_FISH_df_dict
data_dir=r"/mnt/disk1/scfan/data/Cai_21_bioRxiv"
if __name__ == '__main__':
    from datetime import datetime

    # Record the start time
    start_time = datetime.now()

    temp_path = r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/DNA_1Mb/cell_base_info/missing_rate_dic.pkl"
    temp_dic = basic.load_variable_from_pikle_file(temp_path)
    each_cell_missing_rate_df = temp_dic['each_cell_missing_rate_df']
    each_chr_missing_rate_df = temp_dic['each_chr_missing_rate_df']

    missing_rate_df = each_chr_missing_rate_df.set_index(['cell_name', 'chr_name'])

    one_cell_DNA_RNA_confidence = 1

    temp_path = r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/ImputHiFI/all_HiC_all_pair_adata.h5ad"
    HiC_adata = sc.read(temp_path)

    temp_path = r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/Integration_Xie_21/scGAD_HiC_RNA_correlations_df.pkl"
    HiC_RNA_correlations_df = basic.load_variable_from_pikle_file(temp_path)

    temp_path = r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/ImputHiFI/scGAD_kmeans_coef_DNA_FISH_RNA_HiC_dic_name_tuple_neighbor_5.pkl"
    DNA_FISH_RNA_HiC_dic_name_tuple = basic.load_variable_from_pikle_file(temp_path)

    temp_path = r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/ImputHiFI/first_seen_probe_data/ImputeHiFI_v2_proximity_score_bin_pair_adata.h5ad"
    DNA_FISH_adata = sc.read(temp_path)

    # temp_path = r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/ImputHiFI/first_seen_probe_data_spatial_dist_bin_pair_adata.h5ad"
    # DNA_FISH_spad_adata = sc.read(temp_path)

    DNA_FISH_name_list = list(DNA_FISH_RNA_HiC_dic_name_tuple.keys())
    chr_length_file = r"/mnt/disk1/scfan/data/Genome/mm10_chr_length.txt"
    resolution_name = '1Mb'
    resolution = 1000000
    chr_length_dic = basic.get_chr_length_info(chr_length_file)
    chr_bin_length_dic = {}
    for one_chr in chr_length_dic.keys():
        if one_chr != 'chrY' and one_chr != 'chrX':
            bin_length = chr_length_dic[one_chr] // resolution + 1
            chr_bin_length_dic[one_chr] = bin_length
    # RNA_FISH_nearest_neighbors_names_dic=ImputeHiFI.get_RNA_FISH_nearset_neighbors(data_dir)


    kernerl_number=32
    DNA_FISH_name_split_list = np.array_split(DNA_FISH_name_list, kernerl_number)
    store_dir = r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/ImputHiFI/first_seen_probe_data"
    import multiprocessing as mp
    num_threads = kernerl_number
    base_coef = 0.001
    HiC_weight = 0.4
    FISH_weight = 0.6
    short_range_HiC_weight_bound = 0.4
    long_range_HiC_weight_bound = 0.1
    all_range_diag_sum_weight_HiC_bound = 0.1
    short_range_diag = 20
    initial_guess = [HiC_weight, FISH_weight]
    verbose = False
    background_cell_num = 100
    if(len(sys.argv)>1):
        short_range_HiC_weight_bound = float(sys.argv[1])
    else:
        short_range_HiC_weight_bound = 0.4
    if(len(sys.argv)>2):
        long_range_HiC_weight_bound = float(sys.argv[2])
    else:
        long_range_HiC_weight_bound = 0.1
    if(len(sys.argv)>3):
        all_range_diag_sum_weight_HiC_bound = float(sys.argv[3])
    else:
        all_range_diag_sum_weight_HiC_bound = 0.1
    if(len(sys.argv)>4):
        short_range_diag = int(sys.argv[4])
    else:
        short_range_diag = 20
    if(len(sys.argv)>5):
        background_cell_num = int(sys.argv[5])
    else:
        background_cell_num = 100
    if(len(sys.argv)>6):
        temp_path = sys.argv[6]
        similar_FISH_cell_path = temp_path
    else:
        similar_FISH_cell_path =os.path.join(
            r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/ImputHiFI/first_seen_probe_data",
            "All_cell_RNA_FISH_RMSD.pkl")
    if(len(sys.argv)>7):
        version=sys.argv[7]
    else:
        version='ImputeHiFI mode 1'
    if(len(sys.argv)>8):
        use_mean_count=bool(int(sys.argv[8]))
    else:
        use_mean_count=False

    FISH_nearest_neighbors_dic=basic.load_variable_from_pikle_file(similar_FISH_cell_path)
    FISH_nearest_neighbors_names_dic=FISH_nearest_neighbors_dic['nearest_neighbors_names_dic']

    para=('bg_n'+str(background_cell_num)+'_sdn'+str(short_range_diag)+'_sd'+str(all_range_diag_sum_weight_HiC_bound)+
          '_sr'+str(short_range_HiC_weight_bound)+'_lr'+str(long_range_HiC_weight_bound))

    parameter_dic={}
    parameter_dic['base_coef']=base_coef
    parameter_dic['HiC_weight']=HiC_weight
    parameter_dic['FISH_weight'] = FISH_weight
    parameter_dic['short_range_HiC_weight_bound'] = short_range_HiC_weight_bound
    parameter_dic['long_range_HiC_weight_bound'] = long_range_HiC_weight_bound
    parameter_dic['all_range_diag_sum_weight_HiC_bound'] = all_range_diag_sum_weight_HiC_bound
    parameter_dic['short_range_diag'] = short_range_diag
    parameter_dic['background_cell_num']=background_cell_num


    _domain_args = [( chr_bin_length_dic, base_coef,short_range_HiC_weight_bound,
                      long_range_HiC_weight_bound,all_range_diag_sum_weight_HiC_bound,
                        short_range_diag,background_cell_num,
                               HiC_RNA_correlations_df, DNA_FISH_RNA_HiC_dic_name_tuple,
                               DNA_FISH_name_list, missing_rate_df, DNA_FISH_adata,
                               FISH_nearest_neighbors_names_dic,use_mean_count)
                    for DNA_FISH_name_list in DNA_FISH_name_split_list]
    with mp.Pool(num_threads) as domain_pool:
        domain_results = domain_pool.starmap(generate_Impute_result, _domain_args)
        domain_pool.close()
        domain_pool.join()
        domain_pool.terminate()

    result_dic = {}
    for result in domain_results:
        print("Collect result")
        for cell_name,df in result.items():
            result_dic[cell_name]=df
    Example_dict = {}
    for chr_name, bin_length in chr_bin_length_dic.items():
        Example_dict[chr_name] = ImputeHiFI.generate_Example_bin_pair_df(chr_name, bin_length)

    all_index_df = pd.concat(Example_dict.values(), axis=0)
    all_cell_df=pd.DataFrame(index=all_index_df.index)
    for cell_name,df in result_dic.items():
        all_cell_df[cell_name]=df.loc[all_cell_df.index,'data'].values
    Impute_adata=sc.AnnData(all_cell_df.T)
    Impute_adata.obs = DNA_FISH_adata.obs.loc[Impute_adata.obs_names]
    Impute_adata.uns['parameter'] = parameter_dic

    temp_path = r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/ImputHiFI/first_seen_probe_data_Impute/ImputeHiFI_"+str(version)+"_proximity_score_adata_"+str(para)+".h5ad"
    Impute_adata.write(temp_path)

    # Record the end time
    end_time = datetime.now()

    # Print the runtime
    runtime = end_time - start_time
    print(f"Code executed in {runtime}")