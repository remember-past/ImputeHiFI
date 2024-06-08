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

def get_evaluation_result(chr_bin_length_dic, chr_name,chr_name_list,all_FISH_name,
                          ImputeHiFI_adata,ImputeHiFI_diff_genes_adata,all_cell_dna_list,Linear_impute_all_cell_dna_list,
                          chr_cell_type_center_value_df,chrom_ids_to_index_dict,loci_pair_type,
                          ):
    print(chr_name)
    if loci_pair_type != 'no_diff':
        print("Fine tune with leiden find import loci pair")
    optimize_all_cell_dna_list=copy.deepcopy(Linear_impute_all_cell_dna_list)
    for one_FISH_name in tqdm(all_FISH_name):
        one_cell_type = ImputeHiFI_adata.obs.loc[one_FISH_name]['cell_type']
        sigma = chr_cell_type_center_value_df.loc[chr_name + one_cell_type]['Center_Value']

        proximity_array = ImputeHiFI.get_proximity_array(ImputeHiFI_adata, one_FISH_name, chr_name, chr_bin_length_dic,
                                                         set_min_limit=False, min_limit=0.001, set_max_limit=True,
                                                         max_limit=1.0)
        linear_zxys = ImputeHiFI.infer_3D_position.get_specific_dan_zxys(chrom_ids_to_index_dict, one_FISH_name, chr_name_list, chr_name,
                                            Linear_impute_all_cell_dna_list)
        not_imputed_zxys = ImputeHiFI.infer_3D_position.get_specific_dan_zxys(chrom_ids_to_index_dict, one_FISH_name, chr_name_list, chr_name,
                                                 all_cell_dna_list)

        K_imputed = proximity_array.copy()
        initial_guess = linear_zxys.copy()

        nan_position_array=ImputeHiFI.infer_3D_position.get_nan_positions(chrom_ids_to_index_dict,
                                                                          one_FISH_name, chr_name_list,
                                                                          chr_name, all_cell_dna_list)

        not_imputed_all_x = not_imputed_zxys.copy()
        need_imputed_x_initial_guess = initial_guess[nan_position_array].copy()
        # bounds = ImputeHiFI.infer_3D_position.get_bounds(not_nan_array, initial_guess)
        bounds = None
        # optimze_zxys = ImputeHiFI.infer_3D_position_all.optimize_coordinates(K_imputed, sigma, initial_guess,
        #                                                                      maxfun=5000, maxiter=5000, bounds=bounds,
        #                                                                      bias=None, verbose=False)
        optimze_zxys=ImputeHiFI.infer_3D_position.optimize_coordinates(K_imputed, sigma, need_imputed_x_initial_guess,
                                                          not_imputed_all_x, nan_position_array, maxfun=5000,
                                                          maxiter=5000, bounds=bounds, bias=None, verbose=False)
        if loci_pair_type != 'no_diff':
            # print("Fine tune with leiden find import loci pair")
            #fine tune with leiden find import loci pair
            proximity_array = ImputeHiFI.get_proximity_array(ImputeHiFI_diff_genes_adata, one_FISH_name, chr_name, chr_bin_length_dic,
                                                             set_min_limit=False, min_limit=0.001, set_max_limit=True,
                                                             max_limit=1.0)
            K_imputed = proximity_array.copy()
            initial_guess = optimze_zxys.copy()
            not_imputed_all_x = not_imputed_zxys.copy()
            need_imputed_x_initial_guess = initial_guess[nan_position_array].copy()
            optimze_zxys=ImputeHiFI.infer_3D_position.optimize_coordinates(K_imputed, sigma, need_imputed_x_initial_guess,
                                                              not_imputed_all_x, nan_position_array, maxfun=5000,
                                                              maxiter=5000, bounds=bounds, bias=None, verbose=False)


        cell_idx = chrom_ids_to_index_dict[one_FISH_name]
        chr_idx = chr_name_list.index(chr_name)
        optimize_all_cell_dna_list[cell_idx][chr_idx] = optimze_zxys

    corr_dict = {}
    corr_dict['optimize_all_cell_dna_list']=optimize_all_cell_dna_list
    one_chr_corr_dict={}
    one_chr_corr_dict[chr_name]=corr_dict
    return chr_name,one_chr_corr_dict
def generate_spatial_bin_pair_df_one_chr(data,chr_name,chr_bin_length_dic):

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
        Example_bin_pair_df = ImputeHiFI.generate_Example_bin_pair_df(chr_name, bin_length)
        all_cell_one_chr_df = pd.DataFrame(index=Example_bin_pair_df.index)
        for cell_index, cell_ids in tqdm(enumerate(data['chrom_ids'])):
            one_cell_chr_array = distmap_list_dic[chr_name][cell_index].copy()
            one_cell_chr_df = ImputeHiFI.convert_array_to_bin_pair_df(chr_name, bin_length, one_cell_chr_array)
            cell_name = str(cell_ids)
            all_cell_one_chr_df.loc[one_cell_chr_df.index, cell_name] = one_cell_chr_df['data'].values
        all_cell_df_dic[chr_name] = all_cell_one_chr_df

    return chr_name,all_cell_df_dic

def generate_proximity_bin_pair_df_one_chr(DNA_FISH_spad_adata,chr_name,chr_bin_length_dic,chr_cell_type_center_value_df):

    # store_dir = r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/ImputHiFI/first_seen_probe_data"


    all_cell_df_dic = {}

    print(chr_name)
    # chr_name='chr1'
    bin_length = chr_bin_length_dic[chr_name]
    Example_bin_pair_df = ImputeHiFI.generate_Example_bin_pair_df(chr_name, bin_length)
    all_cell_one_chr_df = pd.DataFrame(index=Example_bin_pair_df.index)
    for cell_index, one_cell_name in tqdm(enumerate(DNA_FISH_spad_adata.obs_names)):
        one_cell_type = DNA_FISH_spad_adata.obs.loc[one_cell_name]['cell_type']
        center_value = chr_cell_type_center_value_df.loc[chr_name + one_cell_type]['Center_Value']
        Example_df = ImputeHiFI.generate_Example_bin_pair_df(chr_name, bin_length)
        temp_array = DNA_FISH_spad_adata[one_cell_name, Example_df.index].copy().X[0]
        Example_df['data'] = temp_array
        matrix_array = ImputeHiFI.convert_bin_pair_df_to_array(bin_length, Example_df)
        matrix_array = matrix_array + matrix_array.T
        np.fill_diagonal(matrix_array, 0)
        dist_matrix = copy.deepcopy(matrix_array)
        sigma = center_value
        matrix_array = np.exp2(-dist_matrix * dist_matrix / (sigma * sigma))
        proxi_array = copy.deepcopy(matrix_array)

        one_cell_chr_array = proxi_array
        one_cell_chr_df = ImputeHiFI.convert_array_to_bin_pair_df(chr_name, bin_length, one_cell_chr_array)
        cell_name = one_cell_name
        all_cell_one_chr_df.loc[one_cell_chr_df.index, cell_name] = one_cell_chr_df['data'].values
    all_cell_df_dic[chr_name] = all_cell_one_chr_df

    return chr_name,all_cell_df_dic

data_dir=r"/mnt/disk1/scfan/data/Cai_21_bioRxiv"
if __name__ == '__main__':
    from datetime import datetime

    # Record the start time
    start_time = datetime.now()


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
        # adata = sc.read(temp_path)
    else:
        temp_path = r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/ImputHiFI/first_seen_probe_data/first_seen_probe_data_proximity_score_bin_pair_adata.h5ad"
        # adata = sc.read(temp_path)
    if(len(sys.argv)>7):
        data_dir = sys.argv[7]
    else:
        data_dir = r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/ImputHiFI"
    if(len(sys.argv)>8):
        ImputeHiFI_dir = sys.argv[8]
    else:
        ImputeHiFI_dir = r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/ImputHiFI/first_seen_probe_data_Impute"
    if(len(sys.argv)>9):
        Linear_impute_dir = sys.argv[9]
    else:
        Linear_impute_dir = r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/ImputHiFI/first_seen_probe_data_Linear/"
    if(len(sys.argv)>10):
        version=sys.argv[10]
    else:
        version='ImputeHiFI mode 1'
    if(len(sys.argv)>11):
        infer_3D_type = sys.argv[11]
    else:
        infer_3D_type = r"_part_"
    if(len(sys.argv)>12):
        n_loci_pair = int(sys.argv[12])
    else:
        n_loci_pair = 6000
    if(len(sys.argv)>13):
        loci_pair_type = sys.argv[13]
    else:
        loci_pair_type = 'leiden_diff'# or 'highly_variable_loci_pair','no_diff'

    temp_path = os.path.join(data_dir,r"first_seen_probe_data.pkl")
    data = basic.load_variable_from_pikle_file(temp_path)
    chrom_ids_to_index_dict = {}
    for index, chrom_ids in enumerate(data['chrom_ids']):
        chrom_ids_to_index_dict[str(chrom_ids)] = index
    chr_name_list = data['chrom_name']

    temp_path = '/mnt/disk1/scfan/data/Cai_21_bioRxiv/ImputHiFI/first_seen_probe_data/chr_cell_type_center_value_P50.txt'
    # 读取CSV文件
    chr_cell_type_center_value_df = pd.read_csv(temp_path, sep='\t')
    chr_cell_type_center_value_df = chr_cell_type_center_value_df.set_index('Chr_Cell_Type')


    para=('bg_n'+str(background_cell_num)+'_sdn'+str(short_range_diag)+'_sd'+str(all_range_diag_sum_weight_HiC_bound)+
          '_sr'+str(short_range_HiC_weight_bound)+'_lr'+str(long_range_HiC_weight_bound))


    temp_path = os.path.join(ImputeHiFI_dir, "ImputeHiFI_" + version + "_proximity_score_adata_" + str(para) + ".h5ad")
    ImputeHiFI_adata = sc.read(temp_path)
    # n_loci_pair=6000
    if loci_pair_type == 'leiden_diff':
        diff_genes,not_diff_genes,non_diff_genes_mask=ImputeHiFI.get_diffgene_with_leiden(ImputeHiFI_adata.copy(), n_loci_pair=n_loci_pair)

    if loci_pair_type == 'highly_variable_loci_pair':
        diff_genes, not_diff_genes, non_diff_genes_mask = ImputeHiFI.get_highly_variable_genes(ImputeHiFI_adata.copy(),
                                                                                          n_loci_pair=n_loci_pair)
    if loci_pair_type == 'no_diff':
        diff_genes = []
        not_diff_genes = []
        non_diff_genes_mask = np.zeros(ImputeHiFI_adata.shape[1], dtype=bool)
    # sc.pp.normalize_total(ImputeHiFI_adata, target_sum=1e4)
    ImputeHiFI_diff_genes_adata=ImputeHiFI_adata.copy()
    ImputeHiFI_diff_genes_adata.X[:, non_diff_genes_mask] =0

    all_FISH_name=list(ImputeHiFI_adata.obs_names)

    all_cell_dna_list=data['all_cell_dna_list']


    temp_path = os.path.join(Linear_impute_dir, 'Impute_data.pkl')
    Linear_impute_data = basic.load_variable_from_pikle_file(temp_path)
    Linear_impute_all_cell_dna_list = Linear_impute_data['Linear_Impute']

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

    import multiprocessing as mp
    num_threads = kernerl_number

    _domain_args = [( chr_bin_length_dic,chr_name_part[0],chr_name_list,all_FISH_name,
                          ImputeHiFI_adata,ImputeHiFI_diff_genes_adata,all_cell_dna_list,
                      Linear_impute_all_cell_dna_list,
                          chr_cell_type_center_value_df,chrom_ids_to_index_dict,loci_pair_type)
                    for chr_name_part in chr_name_split_list]
    with mp.Pool(num_threads) as domain_pool:
        domain_results = domain_pool.starmap(get_evaluation_result, _domain_args)
        domain_pool.close()
        domain_pool.join()
        domain_pool.terminate()


    finall_optimize_all_cell_dna_list=copy.deepcopy(all_cell_dna_list)
    for result in domain_results:
        print("Collect result")
        chr_name,one_chr_corr_dict=result
        optimize_all_cell_dna_list=one_chr_corr_dict[chr_name]['optimize_all_cell_dna_list']
        chr_idx = chr_name_list.index(chr_name)
        for cell_idx in range(len(optimize_all_cell_dna_list)):
            finall_optimize_all_cell_dna_list[cell_idx][chr_idx]=optimize_all_cell_dna_list[cell_idx][chr_idx]
    final_ImputeHiFI_data=copy.deepcopy(data)
    final_ImputeHiFI_data['all_cell_dna_list']=finall_optimize_all_cell_dna_list
    final_ImputeHiFI_data['ledien_diff_gene'] = diff_genes

    temp_path = os.path.join(ImputeHiFI_dir, "ImputeHiFI_" + version + "_proximity_score_adata_" + str(para) +infer_3D_type+"Impute_data.pkl")
    # temp_path = r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/ImputHiFI/first_seen_probe_data_Impute/ImputeHiFI_v2_proximity_score_adata_"+str(para)+".h5ad"
    basic.store_variable_from_pikle_file(temp_path, final_ImputeHiFI_data)

    def generate_spatial_h5ad():
        para=('bg_n'+str(background_cell_num)+'_sdn'+str(short_range_diag)+'_sd'+str(all_range_diag_sum_weight_HiC_bound)+
              '_sr'+str(short_range_HiC_weight_bound)+'_lr'+str(long_range_HiC_weight_bound))
        temp_path = os.path.join(ImputeHiFI_dir, "ImputeHiFI_" + version + "_proximity_score_adata_" + str(
            para) + infer_3D_type + "Impute_data.pkl")
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

        chr_name_list = list(chr_bin_length_dic.keys())
        kernerl_number = 19
        chr_name_split_list = np.array_split(chr_name_list, kernerl_number)
        # store_dir = r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/ImputHiFI/first_seen_probe_data"
        import multiprocessing as mp
        num_threads = kernerl_number

        _domain_args = [(data, chr_name_part[0], chr_bin_length_dic)
                        for chr_name_part in chr_name_split_list]
        with mp.Pool(num_threads) as domain_pool:
            domain_results = domain_pool.starmap(generate_spatial_bin_pair_df_one_chr, _domain_args)
            domain_pool.close()
            domain_pool.join()
            domain_pool.terminate()
        all_cell_df_dic = {}
        for result in domain_results:
            chr_name, temp_all_cell_df_dic=result
            all_cell_df_dic[chr_name] = temp_all_cell_df_dic[chr_name]
        all_cell_df = pd.concat(all_cell_df_dic.values())

        adata = sc.AnnData(all_cell_df.T)

        temp_path = r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/RNA/Cai_RNA_adata.h5ad"
        Cai_RNA_adata = sc.read(temp_path)

        adata.obs = Cai_RNA_adata.obs.loc[adata.obs_names]

        # temp_path=r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/ImputHiFI/first_seen_probe_data_spatial_dist_bin_pair_adata.h5ad"

        temp_path = os.path.join(ImputeHiFI_dir, "ImputeHiFI_" + version + "_proximity_score_adata_" + str(
            para) + infer_3D_type + "Impute_data_spatial_dist.h5ad")
        first_seen_probe_data_spatial_dist_bin_pair_adata = adata.copy()
        first_seen_probe_data_spatial_dist_bin_pair_adata.write(temp_path)


    generate_spatial_h5ad()
    def generate_proximity_h5ad():
        para=('bg_n'+str(background_cell_num)+'_sdn'+str(short_range_diag)+'_sd'+str(all_range_diag_sum_weight_HiC_bound)+
              '_sr'+str(short_range_HiC_weight_bound)+'_lr'+str(long_range_HiC_weight_bound))
        temp_path = os.path.join(ImputeHiFI_dir, "ImputeHiFI_" + version + "_proximity_score_adata_" + str(
            para) + infer_3D_type + "Impute_data_spatial_dist.h5ad")
        DNA_FISH_spad_adata = sc.read(temp_path)

        temp_path = '/mnt/disk1/scfan/data/Cai_21_bioRxiv/ImputHiFI/first_seen_probe_data/chr_cell_type_center_value_P50.txt'  # 替换为您的实际路径
        # 读取CSV文件
        df = pd.read_csv(temp_path, sep='\t')
        chr_cell_type_center_value_df = df
        chr_cell_type_center_value_df = chr_cell_type_center_value_df.set_index('Chr_Cell_Type')

        chr_length_file = r"/mnt/disk1/scfan/data/Genome/mm10_chr_length.txt"
        resolution_name = '1Mb'
        resolution = 1000000
        chr_length_dic = basic.get_chr_length_info(chr_length_file)
        chr_bin_length_dic = {}
        for one_chr in chr_length_dic.keys():
            if one_chr != 'chrY' and one_chr != 'chrX':
                bin_length = chr_length_dic[one_chr] // resolution + 1
                chr_bin_length_dic[one_chr] = bin_length

        chr_name_list = list(chr_bin_length_dic.keys())
        kernerl_number = 19
        chr_name_split_list = np.array_split(chr_name_list, kernerl_number)
        # store_dir = r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/ImputHiFI/first_seen_probe_data"
        import multiprocessing as mp
        num_threads = kernerl_number

        _domain_args = [
            (DNA_FISH_spad_adata, chr_name_part[0], chr_bin_length_dic, chr_cell_type_center_value_df)
            for chr_name_part in chr_name_split_list]
        with mp.Pool(num_threads) as domain_pool:
            domain_results = domain_pool.starmap(generate_proximity_bin_pair_df_one_chr, _domain_args)
            domain_pool.close()
            domain_pool.join()
            domain_pool.terminate()

        all_cell_df_dic = {}
        for result in domain_results:
            chr_name, temp_all_cell_df_dic = result
            all_cell_df_dic[chr_name] = temp_all_cell_df_dic[chr_name]
        all_cell_df = pd.concat(all_cell_df_dic.values())

        adata = sc.AnnData(all_cell_df.T)

        temp_path = r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/RNA/Cai_RNA_adata.h5ad"
        Cai_RNA_adata = sc.read(temp_path)

        adata.obs = Cai_RNA_adata.obs.loc[adata.obs_names]

        # temp_path = r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/ImputHiFI/first_seen_probe_data_bin_pair_adata.h5ad"
        temp_path = os.path.join(ImputeHiFI_dir, "ImputeHiFI_" + version + "_proximity_score_adata_" + str(
            para) + infer_3D_type + "Impute_data_proximity_score.h5ad")
        first_seen_probe_data_bin_pair_adata = adata.copy()
        first_seen_probe_data_bin_pair_adata.write(temp_path)

    generate_proximity_h5ad()
    # Record the end time
    end_time = datetime.now()

    # Print the runtime
    runtime = end_time - start_time
    print(f"Code executed in {runtime}")