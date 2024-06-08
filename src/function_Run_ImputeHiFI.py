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
from scipy.interpolate import griddata

warnings.filterwarnings('ignore')
from function_basic import basic_function as basic

import ImputeHiFI_Algorithm_v5 as ImputeHiFI

import ImputeHiFI_Algorithm_v6_no_HiC as ImputeHiFI_no_HiC

def preprocess_DNA_HiC(data_dir):
    temp_path = os.path.join(data_dir, 'DNA_FISH_HiC.pkl')
    Example_data_dict=basic.load_variable_from_pikle_file(temp_path)

    cell_name_list=Example_data_dict['cell_name_list']
    all_cell_dna_list=Example_data_dict['all_cell_dna_list']
    Linear_impute_all_cell_dna_list=Example_data_dict['Linear_impute_all_cell_dna_list']
    chr_cell_type_center_value_df=Example_data_dict['chr_cell_type_center_value_df']
    FISH_nearest_neighbors_names_dic=Example_data_dict['FISH_nearest_neighbors_names_dic']
    DNA_FISH_adata=Example_data_dict['DNA_FISH_adata']
    HiC_adata=Example_data_dict['HiC_adata']
    HiC_RNA_correlations_df=Example_data_dict['HiC_RNA_correlations_df']
    DNA_FISH_RNA_HiC_dic_name_tuple_dict=Example_data_dict['DNA_FISH_RNA_HiC_dic_name_tuple_dict']
    chr_bin_length_dic=Example_data_dict['chr_bin_length_dic']
    prxoimtiy_score_suffix = 'proximity_score'
    store_path = os.path.join(data_dir, f'first_seen_probe_data_{prxoimtiy_score_suffix}.h5ad')
    DNA_FISH_adata.write(store_path)
    input_data=(cell_name_list,all_cell_dna_list,Linear_impute_all_cell_dna_list,chr_cell_type_center_value_df,FISH_nearest_neighbors_names_dic,DNA_FISH_adata,HiC_adata,HiC_RNA_correlations_df,DNA_FISH_RNA_HiC_dic_name_tuple_dict,chr_bin_length_dic)
    return input_data


def preprocess_DNA_FISH(input_data_file,data_dir):
    # input_data_file = r"41586_2020_3126_MOESM2_ESM_modify_for_ImputeHiFI.xlsx"
    xls = pd.ExcelFile(os.path.join(data_dir, input_data_file))
    DNA_probe_25kb_df = pd.read_excel(xls, '25-kb resolution')
    start_position, end_position, mid_position, mid_position_Mb, chrom_name, gene_ID, chrom_region_name = basic.get_position(
        DNA_probe_25kb_df)
    data_path = os.path.join(DNA_seqFISH_dir, "DNAseqFISH+{}loci-E14-replicate{}.csv")
    resol = '25kb'
    rep1_data = pd.read_csv(data_path.format(resol, 1))
    rep1_data["cell_type"] = 'rep1'
    rep2_data = pd.read_csv(data_path.format(resol, 2))
    rep2_data["cell_type"] = 'rep2'

    data_path = os.path.join(DNA_seqFISH_dir, "DNAseqFISH+{}loci-E14-clonal{}hr.csv")
    clone_24_data = pd.read_csv(data_path.format(resol, 24))
    clone_24_data["cell_type"] = '24hr'
    clone_48_data = pd.read_csv(data_path.format(resol, 48))
    clone_48_data["cell_type"] = '48hr'

    DNA_pos_data = pd.concat([rep1_data, rep2_data, clone_24_data, clone_48_data], ignore_index=True, sort=False)
    DNA_pos_data = DNA_pos_data[(DNA_pos_data["labelID"] == 0) | (DNA_pos_data["labelID"] == 1)]
    DNA_pos_data['cell_name'] = DNA_pos_data['fov'].astype(str) + '_' + DNA_pos_data['channel'].astype(str) + '_' + \
                                DNA_pos_data['cellID'].astype(str) + '_' + DNA_pos_data['cell_type'].astype(str)
    temp_df = DNA_pos_data.copy()
    unique_cells = temp_df.drop_duplicates(subset='cell_name', keep='first')

    unique_cells = unique_cells[['cell_name', 'cell_type']]

    unique_cells.reset_index(inplace=True, drop=True)

    unique_cells['cell_index'] = list(unique_cells.index)
    cell_name_array = unique_cells['cell_name'].values
    DNA_pos_data['regionID'] = DNA_pos_data['regionID (hyb1-60)']
    zxys_df = DNA_pos_data[['chromID', 'labelID', 'regionID', 'cell_type', 'cell_name', 'z', 'x', 'y']]
    template_dfs = {}

    # 为每个chromID生成一个DataFrame
    for chromID in range(1, 21):  # chromID从1到20
        # 从原始DataFrame中选出对应chromID的regionID
        regions = np.arange(1, 61)

        # 创建新的DataFrame
        template_df_0 = pd.DataFrame({
            'chromID': chromID,
            'regionID': regions,
            'labelID': 0,
            'x': np.nan,
            'y': np.nan,
            'z': np.nan
        })
        template_df_0.index = template_df_0['chromID'].astype(str) + '_' + template_df_0['regionID'].astype(str) + '_' + \
                              template_df_0['labelID'].astype(str)
        template_df_1 = pd.DataFrame({
            'chromID': chromID,
            'regionID': regions,
            'labelID': 1,
            'x': np.nan,
            'y': np.nan,
            'z': np.nan
        })
        template_df_1.index = template_df_1['chromID'].astype(str) + '_' + template_df_1['regionID'].astype(str) + '_' + \
                              template_df_1['labelID'].astype(str)
        # 将新的DataFrame存储到字典中
        template_dfs[chromID] = [template_df_0, template_df_1]
    zxys_df['chromID_regionID_labelID'] = zxys_df['chromID'].astype(str) + '_' + zxys_df['regionID'].astype(str) + '_' + \
                                          zxys_df['labelID'].astype(str)
    zxys_df['chromID'] = zxys_df['chromID'].astype(int)
    zxys_df['labelID'] = zxys_df['labelID'].astype(int)
    all_cell_dna_list = []
    for one_cell_name in tqdm(cell_name_array):
        temp_one_cell_df = zxys_df[zxys_df['cell_name'] == one_cell_name]
        temp_one_cell_df.index = temp_one_cell_df['chromID_regionID_labelID']
        one_cell_list = []
        one_cell_dna_dfs = copy.deepcopy(template_dfs)
        for chromID in range(1, 21):
            one_chrom_list = []
            for labelID in [0, 1]:
                one_chromID_temp_df = temp_one_cell_df[
                    (temp_one_cell_df['chromID'] == chromID) & (temp_one_cell_df['labelID'] == labelID)]
                one_cell_dna_dfs[chromID][labelID].loc[one_chromID_temp_df.index, ['x', 'y', 'z']] = \
                one_chromID_temp_df[['x', 'y', 'z']]
                one_chrom_list.append(one_cell_dna_dfs[chromID][labelID][['z', 'x', 'y']].values)
            one_cell_list.append(one_chrom_list)
        all_cell_dna_list.append(one_cell_list)
    cluster_label = list(unique_cells['cell_type'].values)
    cluster_label_name_dic = {one_label: one_label for one_label in np.unique(cluster_label)}
    chrom_ids = list(unique_cells['cell_name'].values)
    data_dna_zxy = all_cell_dna_list
    keys_list = ['chrom_ids',
                 'region_names',
                 'start_position',
                 'end_position',
                 'mid_position',
                 'mid_position_Mb',
                 'dna_zxys',
                 'chrom_name',
                 'gene_ID',
                 'cluster_label',
                 'cluster_label_name'
                 ]
    values_list = [chrom_ids,
                   chrom_region_name,
                   start_position,
                   end_position,
                   mid_position,
                   mid_position_Mb,
                   data_dna_zxy,
                   chrom_name,
                   gene_ID,
                   cluster_label,
                   cluster_label_name_dic
                   ]
    data = {}
    for one_key, one_value in zip(keys_list, values_list):
        data[one_key] = one_value
    data_pkl_path = 'data.pkl'

    basic.store_variable_from_pikle_file(os.path.join(data_dir, data_pkl_path),
                                         data)
    RNA_seqFISH_dir=data_dir
    data_path = os.path.join(RNA_seqFISH_dir, "mRNA-copy-number-E14-replicate{}.csv")

    rep1_data = pd.read_csv(data_path.format(1))
    gene_name_list = list(rep1_data.columns[7:])
    rep1_data["cell_type"] = 'rep1'
    rep2_data = pd.read_csv(data_path.format(2))
    gene_name_list += list(rep2_data.columns[7:])
    rep2_data["cell_type"] = 'rep2'

    data_path = os.path.join(RNA_seqFISH_dir, "mRNA-copy-number-E14-clonal{}hr.csv")
    clone_24_data = pd.read_csv(data_path.format(24))
    gene_name_list += list(clone_24_data.columns[7:])
    clone_24_data["cell_type"] = '24hr'
    clone_48_data = pd.read_csv(data_path.format(48))
    gene_name_list += list(clone_48_data.columns[7:])
    clone_48_data["cell_type"] = '48hr'
    gene_name_list = list(np.unique(gene_name_list))
    RNA_origin_data = pd.concat([rep1_data, rep2_data, clone_24_data, clone_48_data], ignore_index=True, sort=False)
    RNA_origin_data['cell_name'] = RNA_origin_data['fov'].astype(str) + '_3_' + RNA_origin_data['cellID'].astype(
        str) + '_' + RNA_origin_data['cell_type'].astype(str)
    RNA_origin_data.index = RNA_origin_data['cell_name']
    RNA_name_array = RNA_origin_data['cell_name'].values
    RNA_df=RNA_origin_data[gene_name_list]

    RNA_df=RNA_df.fillna(0)
    RNA_adata = sc.AnnData(RNA_df)
    RNA_adata.obs['cell_type'] = RNA_origin_data['cell_type']
    RNA_adata.obs['cell_name'] = RNA_origin_data['cell_name']
    temp_path = os.path.join(data_dir, 'RNA_adata.h5ad')
    RNA_adata.write(temp_path)
    data_pkl_path = 'data.pkl'
    all_cell_data=basic.load_variable_from_pikle_file(os.path.join(data_dir, data_pkl_path))
    all_cell_name_list = all_cell_data['chrom_ids']
    dna_zxys = all_cell_data['dna_zxys']
    cluster_label = all_cell_data['cluster_label']
    new_cell_name_list = []
    new_dna_zxys = []
    new_cluster_label = []
    for i, one_cell_name in enumerate(all_cell_name_list):
        if one_cell_name in RNA_name_array:
            new_cell_name_list.append(one_cell_name)
            new_dna_zxys.append(dna_zxys[i])
            new_cluster_label.append(cluster_label[i])
    new_cell_data = copy.deepcopy(all_cell_data)
    new_cell_data['chrom_ids'] = new_cell_name_list
    new_cell_data['dna_zxys'] = new_dna_zxys
    new_cell_data['cluster_label'] = new_cluster_label
    data_pkl_path = 'align_name_with_RNA_data.pkl'
    basic.store_variable_from_pikle_file(os.path.join(r"/mnt/disk1/scfan/data/Cai_21_Nature/DNA_25kb", data_pkl_path),
                                         new_cell_data)

    data = new_cell_data
    first_seen_probe_data_dir = data_dir
    ImputeHiFI_data_dir = os.path.join(first_seen_probe_data_dir, 'ImputeHiFI')
    RNA_FISH_adata=RNA_adata
    chrom_ids_to_index_dict = {}
    cell_name_list = []
    for index, chrom_ids in enumerate(data['chrom_ids']):
        chrom_ids_to_index_dict[str(chrom_ids)] = index
        cell_name_list.append(str(chrom_ids))
    cell_name_array = np.array(cell_name_list)
    chrom_index_to_ids_dict = {index: chrom_id for chrom_id, index in chrom_ids_to_index_dict.items()}
    chr_name_list = data['chrom_name'][0:20]
    num_cells = len(chrom_ids_to_index_dict)
    num_chromosomes = len(chr_name_list)

    len_num = 60

    # 初始化结构，每个cell有一个列表，列表中有num_chromosomes个元素
    max_all_cell_dna_list = [[] for _ in range(num_cells)]
    min_all_cell_dna_list = [[] for _ in range(num_cells)]
    max_all_cell_dna_list_not_missing_rate = [[] for _ in range(num_cells)]
    min_all_cell_dna_list_not_missing_rate = [[] for _ in range(num_cells)]
    choice_list = [[] for _ in range(num_cells)]  # 新增列表用于记录选择的是0还是1

    # 遍历每个cell和其对应的染色体
    for cell_index, (one_cell_name, one_cell_ids) in enumerate(chrom_ids_to_index_dict.items()):
        for chr_ids, chr_name in enumerate(chr_name_list):
            # 计算非缺失率
            not_nan_sum_0 = np.sum(~np.isnan(all_cell_dna_list[one_cell_ids][chr_ids][0])[:, 0])
            not_nan_sum_1 = np.sum(~np.isnan(all_cell_dna_list[one_cell_ids][chr_ids][1])[:, 0])
            not_missing_rate_0 = not_nan_sum_0 / len_num
            not_missing_rate_1 = not_nan_sum_1 / len_num

            # 根据非缺失率将数据存储到max或min列表中，并记录选择
            if not_missing_rate_0 > not_missing_rate_1:
                max_all_cell_dna_list[cell_index].append(all_cell_dna_list[one_cell_ids][chr_ids][0])
                max_all_cell_dna_list_not_missing_rate[cell_index].append(not_missing_rate_0)
                min_all_cell_dna_list[cell_index].append(all_cell_dna_list[one_cell_ids][chr_ids][1])
                min_all_cell_dna_list_not_missing_rate[cell_index].append(not_missing_rate_1)
                choice_list[cell_index].append(0)
            else:
                max_all_cell_dna_list[cell_index].append(all_cell_dna_list[one_cell_ids][chr_ids][1])
                max_all_cell_dna_list_not_missing_rate[cell_index].append(not_missing_rate_1)
                min_all_cell_dna_list[cell_index].append(all_cell_dna_list[one_cell_ids][chr_ids][0])
                min_all_cell_dna_list_not_missing_rate[cell_index].append(not_missing_rate_0)
                choice_list[cell_index].append(1)
    all_cell_mean_not_missing_rate=[]
    for cell_index, (one_cell_name, one_cell_ids) in enumerate(chrom_ids_to_index_dict.items()):
        one_cell_mean_not_missing_rate=np.mean(min_all_cell_dna_list_not_missing_rate[cell_index]+max_all_cell_dna_list_not_missing_rate[cell_index])
        all_cell_mean_not_missing_rate.append(one_cell_mean_not_missing_rate)

    first_seen_probe_data = copy.deepcopy(data)
    first_seen_probe_data['all_cell_dna_list'] = max_all_cell_dna_list
    first_seen_probe_data['chrom_ids_to_index_dict'] = chrom_ids_to_index_dict

    first_seen_probe_data['chrom_index_to_ids_dict'] = chrom_index_to_ids_dict
    first_seen_probe_data['chr_name_list'] = chr_name_list
    first_seen_probe_data['cell_name_array'] = cell_name_array
    first_seen_probe_data['cell_name_list'] = cell_name_list
    result_list = []


    for cell_index, (one_cell_name, one_cell_ids) in enumerate(chrom_ids_to_index_dict.items()):
        for chr_index, chr_name in enumerate(chr_name_list):
            # 获取当前cell和染色体的DNA数据
            dna_data = max_all_cell_dna_list[cell_index][chr_index]

            # 计算总探针数量和NaN探针数量
            total_probe_number = len(dna_data)
            nan_probe_number = np.sum(np.isnan(dna_data)[:, 0])

            # 计算缺失率
            missing_rate = nan_probe_number / total_probe_number if total_probe_number else np.nan

            # 将结果存储到列表中
            result_list.append({
                'cell_name': one_cell_name,
                'chr_name': chr_name,
                'total_probe_number': total_probe_number,
                'nan_probe_number': nan_probe_number,
                'missing_rate': missing_rate
            })

    df_columns = ['cell_name', 'chr_name', 'total_probe_number', 'nan_probe_number', 'missing_rate']
    each_chr_missing_rate_df = pd.DataFrame(result_list, columns=df_columns)
    missing_rate_df = each_chr_missing_rate_df.set_index(['cell_name', 'chr_name'])
    first_seen_probe_data['each_chr_missing_rate_df'] = each_chr_missing_rate_df
    first_seen_probe_data['missing_rate_df'] = missing_rate_df
    chr_bin_length_dic = {}
    for chr_name in chr_name_list:
        chr_bin_length_dic[chr_name] = len_num
    first_seen_probe_data['chr_bin_length_dic'] = chr_bin_length_dic
    cell_number = len(first_seen_probe_data['chrom_ids'])
    first_seen_probe_data['cell_number'] = cell_number
    cluster_label_list = first_seen_probe_data['cluster_label']
    cluster_label_name_dict = first_seen_probe_data['cluster_label_name']
    result_list = []
    for cell_index, label_idx in enumerate(cluster_label_list):
        one_cell_name = chrom_index_to_ids_dict[cell_index]
        cell_type = cluster_label_name_dict[label_idx]
        result_list.append({
            'cell_name': one_cell_name,
            'cell_type': cell_type,
        })
    cell_type_df = pd.DataFrame(result_list)
    cell_type_df.index = cell_type_df['cell_name']
    first_seen_probe_data['cell_type_df'] = cell_type_df
    adata_obs_df = cell_type_df
    first_seen_probe_data['adata_obs_df'] = adata_obs_df
    spaital_dist_suffix = 'spatial_dist'
    prxoimtiy_score_suffix = 'proximity_score'
    first_seen_probe_data['spaital_dist_suffix'] = spaital_dist_suffix
    first_seen_probe_data['prxoimtiy_score_suffix'] = prxoimtiy_score_suffix
    store_path = os.path.join(first_seen_probe_data_dir, f'first_seen_probe_data_{spaital_dist_suffix}.h5ad')
    ImputeHiFI.generate_spatial_distance_h5ad.generate_spatial_h5ad(chr_name_list, cell_number, all_cell_dna_list,
                                                                    chr_bin_length_dic, chrom_index_to_ids_dict,
                                                                    adata_obs_df, store_path, )
    store_path = os.path.join(first_seen_probe_data_dir, f'first_seen_probe_data_{spaital_dist_suffix}.h5ad')
    DNA_FISH_spad_adata = sc.read(store_path)
    adata = DNA_FISH_spad_adata.copy()
    cell_types = adata.obs['cell_type'].unique()

    # 创建一个字典来存储每种细胞类型的子集数据
    cell_type_subsets = {}

    for cell_type in cell_types:
        # 使用 adata[adata.obs['cell_type'] == cell_type] 来选择特定类型的细胞
        cell_type_subset = adata[adata.obs['cell_type'] == cell_type]
        # 将子集数据存储在字典中
        cell_type_subsets[cell_type] = cell_type_subset
    p = 50
    chr_cell_type_center_value_dict = {}
    for chr_name in tqdm(chr_bin_length_dic.keys()):
        num_cell_types = len(cell_types)
        for index, cell_type in enumerate(cell_types):
            title_name = f"{cell_type} {chr_name}"
            bin_length = chr_bin_length_dic[chr_name]
            Example_df = ImputeHiFI.generate_Example_bin_pair_df(chr_name, bin_length)
            temp_array = cell_type_subsets[cell_type][:, Example_df.index].copy().X
            title_name += f" ({temp_array.shape[0]} chromosomes)"
            Example_df['data'] = np.nanmedian(temp_array, axis=0)
            matrix_array = ImputeHiFI.convert_bin_pair_df_to_array(bin_length, Example_df)
            matrix_array = matrix_array + matrix_array.T
            np.fill_diagonal(matrix_array, 0)
            rows, cols = np.triu_indices(matrix_array.shape[0], k=1)

            # 使用这些索引提取上三角矩阵的值
            upper_triangle_values = matrix_array[rows, cols]

            # 从上三角值中过滤出非NaN的值
            non_nan_upper_triangle_values = upper_triangle_values[~np.isnan(upper_triangle_values)]
            #         p=50
            center_value = np.percentile(non_nan_upper_triangle_values, p)
            chr_cell_type_center_value_dict[chr_name + cell_type] = center_value
    chr_cell_type_center_value_df = pd.DataFrame(list(chr_cell_type_center_value_dict.items()),
                                                 columns=['Chr_Cell_Type', 'Center_Value'])
    chr_cell_type_center_value_df.index = chr_cell_type_center_value_df['Chr_Cell_Type']
    first_seen_probe_data['chr_cell_type_center_value_df'] = chr_cell_type_center_value_df
    store_path = os.path.join(first_seen_probe_data_dir, f'first_seen_probe_data_{spaital_dist_suffix}.h5ad')
    DNA_FISH_spad_adata = sc.read(store_path)

    store_path = os.path.join(first_seen_probe_data_dir, f'first_seen_probe_data_{prxoimtiy_score_suffix}.h5ad')
    ImputeHiFI.generate_proximity_score_h5ad.generate_proximity_h5ad(DNA_FISH_spad_adata, chr_name_list,
                                                                     chr_cell_type_center_value_df,
                                                                     chr_bin_length_dic, adata_obs_df, store_path,
                                                                     kernerl_number=None)
    chr_name_list = first_seen_probe_data['chr_name_list']
    cell_number = first_seen_probe_data['cell_number']
    all_cell_dna_list = first_seen_probe_data['all_cell_dna_list']
    cell_name_array = first_seen_probe_data['cell_name_array']
    All_cell_RMSD_dict = ImputeHiFI.generate_cell_cell_RMSD.get_All_cell_RMSD_dict(chr_name_list, cell_number,
                                                                                   all_cell_dna_list,
                                                                                   cell_name_array, kernerl_number=None)
    first_seen_probe_data['All_cell_RMSD_dict'] = All_cell_RMSD_dict
    All_cell_RNA_RMSD_dict = ImputeHiFI.generate_RNA_FISH_cell_cell_RMSD.get_RNA_FISH_cell_cell_RMSD_dict(
        RNA_FISH_adata, chr_name_list, cell_name_array,
        All_cell_RMSD_dict, kernerl_number=None)
    first_seen_probe_data['All_cell_RNA_RMSD_dict'] = All_cell_RNA_RMSD_dict

    cell_number = first_seen_probe_data['cell_number']
    all_cell_dna_list = first_seen_probe_data['all_cell_dna_list']
    chr_num = len(first_seen_probe_data['chr_name_list'])
    Extriming_chr_list = []

    for chr_idx in tqdm(range(chr_num)):
        for cell_idx in range(cell_number):
            _zxys = all_cell_dna_list[cell_idx][chr_idx]
            probe_number = _zxys.shape[0]
            nan_ids = np.where(np.isnan(all_cell_dna_list[cell_idx][chr_idx][:, 0]))[0]
            (continue_number_list, continue_start_ids) = FillNANBasic.FillNANMean.get_NAN_continue_info(nan_ids)
            _zxys_fill_NAN, extreaming_condition_1 = FillNANBasic.FillNANMean.fill_NAN_v1(_zxys,
                                                                                          continue_number_list,
                                                                                          continue_start_ids,
                                                                                          probe_number)
            if not extreaming_condition_1:
                Extriming_chr_list.append(_zxys_fill_NAN)
                break
    Impute_dna_list = copy.deepcopy(all_cell_dna_list)
    for cell_idx in tqdm(range(cell_number)):
        for chr_idx in range(chr_num):
            _zxys = all_cell_dna_list[cell_idx][chr_idx]
            probe_number = _zxys.shape[0]
            nan_ids = np.where(np.isnan(all_cell_dna_list[cell_idx][chr_idx][:, 0]))[0]

            if (len(nan_ids) > 0):
                (continue_number_list, continue_start_ids) = FillNANBasic.FillNANMean.get_NAN_continue_info(nan_ids)

                _zxys_fill_NAN, extreaming_condition_1 = FillNANBasic.FillNANMean.fill_NAN_v1(_zxys,
                                                                                              continue_number_list,
                                                                                              continue_start_ids,
                                                                                              probe_number)
            else:
                _zxys_fill_NAN = _zxys
                extreaming_condition_1 = False
            if (extreaming_condition_1):
                _zxys_fill_NAN = Extriming_chr_list[chr_idx]
            Impute_dna_list[cell_idx][chr_idx] = _zxys_fill_NAN
    first_seen_probe_data['Linear_all_cell_dna_list'] = Impute_dna_list
    first_seen_probe_data['Extriming_chr_list'] = Extriming_chr_list
    first_seen_probe_data_path = os.path.join(first_seen_probe_data_dir, 'first_seen_probe_data.pkl')
    basic.store_variable_from_pikle_file(first_seen_probe_data_path, first_seen_probe_data)
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

def get_infer3D_result(chr_bin_length_dic, chr_name,chr_name_list,all_FISH_name,
                          ImputeHiFI_adata,ImputeHiFI_diff_genes_adata,all_cell_dna_list,Linear_impute_all_cell_dna_list,
                          chr_cell_type_center_value_df,chrom_ids_to_index_dict,loci_pair_type,
                          ):
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
def run_ImputeHiFI_mode_1(data_dir,input_data,HiC_weight_bound=0.4,neighbor_cell_num=100):
    (cell_name_list, all_cell_dna_list, Linear_impute_all_cell_dna_list, chr_cell_type_center_value_df,
     FISH_nearest_neighbors_names_dic, DNA_FISH_adata, HiC_adata, HiC_RNA_correlations_df,
     DNA_FISH_RNA_HiC_dic_name_tuple_dict, chr_bin_length_dic)=input_data
    kernerl_number = 32
    DNA_FISH_name_list=cell_name_list
    DNA_FISH_name_split_list = np.array_split(DNA_FISH_name_list, kernerl_number)
    chr_cell_type_center_value_df = chr_cell_type_center_value_df.set_index('Chr_Cell_Type')

    df_name_list = []
    df_chr_list = []
    df_number_list = []
    for one_cell_name in cell_name_list:
        for one_chr in chr_bin_length_dic.keys():
            probe_num = chr_bin_length_dic[one_chr]
            df_name_list.append(one_cell_name)
            df_chr_list.append(one_chr)
            df_number_list.append(probe_num)

    missing_rate_df = pd.DataFrame()
    missing_rate_df['cell_name'] = df_name_list
    missing_rate_df['chr_name'] = df_chr_list
    missing_rate_df['total_probe_number'] = df_number_list
    missing_rate_df = missing_rate_df.set_index(['cell_name', 'chr_name'])

    import multiprocessing as mp
    num_threads = kernerl_number
    base_coef = 0.001
    HiC_weight = 0.4
    FISH_weight = 0.6
    short_range_HiC_weight_bound = HiC_weight_bound
    long_range_HiC_weight_bound = 0.1
    all_range_diag_sum_weight_HiC_bound = 0.1
    short_range_diag = 20
    initial_guess = [HiC_weight, FISH_weight]
    verbose = False
    background_cell_num = neighbor_cell_num
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
    ImputeHiFI_adata=sc.AnnData(all_cell_df.T)
    ImputeHiFI_adata.obs = DNA_FISH_adata.obs.loc[ImputeHiFI_adata.obs_names]
    ImputeHiFI_adata.uns['parameter'] = parameter_dic
    n_loci_pair=6000
    loci_pair_type='leiden_diff'
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

    chr_name_list=list(chr_bin_length_dic.keys())
    kernerl_number=1
    chr_name_split_list = np.array_split(chr_name_list, kernerl_number)

    import multiprocessing as mp
    num_threads = kernerl_number

    _domain_args = [( chr_bin_length_dic,chr_name_part[0],chr_name_list,all_FISH_name,
                          ImputeHiFI_adata,ImputeHiFI_diff_genes_adata,all_cell_dna_list,
                      Linear_impute_all_cell_dna_list,
                          chr_cell_type_center_value_df,chrom_ids_to_index_dict,loci_pair_type)
                    for chr_name_part in chr_name_split_list]
    with mp.Pool(num_threads) as domain_pool:
        domain_results = domain_pool.starmap(get_infer3D_result, _domain_args)
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
    temp_path = os.path.join(data_dir, "Impute_data.pkl")

    basic.store_variable_from_pikle_file(temp_path, final_ImputeHiFI_data)


    data = final_ImputeHiFI_data

    kernerl_number = 1
    chr_name_split_list = np.array_split(chr_name_list, kernerl_number)

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
        chr_name, temp_all_cell_df_dic = result
        all_cell_df_dic[chr_name] = temp_all_cell_df_dic[chr_name]
    all_cell_df = pd.concat(all_cell_df_dic.values())

    adata = sc.AnnData(all_cell_df.T)

    temp_path = os.path.join(data_dir,"Impute_data_spatial_dist.h5ad")
    adata.write(temp_path)

    DNA_FISH_spad_adata = adata


    kernerl_number = 1
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
    temp_path = os.path.join(data_dir,"Impute_data_proximity_score.h5ad")
    adata.write(temp_path)

def run_ImputeHiFI_mode_2(first_seen_probe_data_dir,neighbor_cell_num=100):
    # current_probe_data_dir = r"/mnt/disk1/scfan/data/Cai_21_Nature/DNA_25kb"
    # first_seen_probe_data_dir = r"/mnt/disk1/scfan/data/Cai_21_Nature/DNA_25kb/Impute/"
    basic.recursive_mkdir(first_seen_probe_data_dir)

    ImputeHiFI_data_dir = os.path.join(first_seen_probe_data_dir, 'ImputeHiFI')
    basic.recursive_mkdir(ImputeHiFI_data_dir)

    ### proxi adata

    FISH_RMSD_name = 'All_cell_RNA_RMSD_dict'
    version = 'no_HiC_RNA_FISH_RMSD_not_use_mean_count'
    first_seen_probe_data_path = os.path.join(first_seen_probe_data_dir, 'first_seen_probe_data.pkl')
    first_seen_probe_data = basic.load_variable_from_pikle_file(first_seen_probe_data_path)

    chr_cell_type_center_value_df = first_seen_probe_data['chr_cell_type_center_value_df']
    chr_name_list = first_seen_probe_data['chr_name_list']
    cell_name_list=first_seen_probe_data['cell_name_list']
    cell_number = first_seen_probe_data['cell_number']
    all_cell_dna_list = first_seen_probe_data['all_cell_dna_list']
    chr_bin_length_dic = first_seen_probe_data['chr_bin_length_dic']
    chrom_index_to_ids_dict = first_seen_probe_data['chrom_index_to_ids_dict']
    adata_obs_df = first_seen_probe_data['adata_obs_df']
    spaital_dist_suffix = first_seen_probe_data['spaital_dist_suffix']
    prxoimtiy_score_suffix = first_seen_probe_data['prxoimtiy_score_suffix']
    missing_rate_df = first_seen_probe_data['missing_rate_df']
    FISH_nearest_neighbors_dic = first_seen_probe_data[FISH_RMSD_name]

    store_path = os.path.join(first_seen_probe_data_dir, f'first_seen_probe_data_{prxoimtiy_score_suffix}.h5ad')
    DNA_FISH_proxi_adata = sc.read(store_path)

    importlib.reload(ImputeHiFI_no_HiC)
    base_coef = 0.001
    HiC_weight = 0
    FISH_weight = 1
    short_range_HiC_weight_bound = 0.0
    long_range_HiC_weight_bound = 0.0
    all_range_diag_sum_weight_HiC_bound = 0.0
    short_range_diag = 0
    initial_guess = [HiC_weight, FISH_weight]
    verbose = False
    background_cell_num = neighbor_cell_num
    use_mean_count = 0
    para = ('bg_n' + str(background_cell_num) + '_sdn' + str(short_range_diag) + '_sd' + str(
        all_range_diag_sum_weight_HiC_bound) +
            '_sr' + str(short_range_HiC_weight_bound) + '_lr' + str(long_range_HiC_weight_bound))

    store_path = os.path.join(ImputeHiFI_data_dir, str(version) + "_proximity_score_adata_" + str(para) + ".h5ad")

    ImputeHiFI_no_HiC.ImputeHiFi_mode2_mode3.get_Impute_proximity_score(DNA_FISH_proxi_adata, missing_rate_df,
                                                                        cell_name_list, chr_bin_length_dic, base_coef,
                                                                        HiC_weight, FISH_weight, initial_guess,
                                                                        short_range_HiC_weight_bound,
                                                                        long_range_HiC_weight_bound,
                                                                        all_range_diag_sum_weight_HiC_bound,
                                                                        short_range_diag, background_cell_num,
                                                                        use_mean_count, FISH_nearest_neighbors_dic,
                                                                        store_path, kernerl_number=20)


    ### Impute 3D data

    first_seen_probe_data.keys()

    importlib.reload(ImputeHiFI)
    store_path = os.path.join(ImputeHiFI_data_dir, str(version) + "_proximity_score_adata_" + str(para) + ".h5ad")
    ImputeHiFI_adata = sc.read(store_path)

    chr_cell_type_center_value_df = first_seen_probe_data['chr_cell_type_center_value_df']
    chr_name_list = first_seen_probe_data['chr_name_list']
    cell_number = first_seen_probe_data['cell_number']
    all_cell_dna_list = first_seen_probe_data['all_cell_dna_list']
    chr_bin_length_dic = first_seen_probe_data['chr_bin_length_dic']
    chrom_index_to_ids_dict = first_seen_probe_data['chrom_index_to_ids_dict']
    chrom_ids_to_index_dict = first_seen_probe_data['chrom_ids_to_index_dict']
    adata_obs_df = first_seen_probe_data['adata_obs_df']
    spaital_dist_suffix = first_seen_probe_data['spaital_dist_suffix']
    prxoimtiy_score_suffix = first_seen_probe_data['prxoimtiy_score_suffix']
    Linear_impute_all_cell_dna_list = first_seen_probe_data['Linear_all_cell_dna_list']

    infer_3D_type = r"_part_loci_pair_weight_"
    loci_pair_type = 'no_diff'

    store_path = os.path.join(ImputeHiFI_data_dir,
                              version + str(para) + infer_3D_type + loci_pair_type + "_Impute_data.pkl")

    finall_optimize_all_cell_dna_list, diff_genes = ImputeHiFI.generate_impute_dna_list.get_impute_dna_list(
        chr_bin_length_dic, chr_name_list, ImputeHiFI_adata,
        all_cell_dna_list, Linear_impute_all_cell_dna_list,
        chr_cell_type_center_value_df, chrom_ids_to_index_dict,
        loci_pair_type, kernerl_number=None)

    final_ImputeHiFI_data = copy.deepcopy(first_seen_probe_data)
    final_ImputeHiFI_data['all_cell_dna_list'] = finall_optimize_all_cell_dna_list
    final_ImputeHiFI_data['ledien_diff_gene'] = diff_genes
    final_ImputeHiFI_data['cell_number']=len(finall_optimize_all_cell_dna_list)

    basic.store_variable_from_pikle_file(store_path, final_ImputeHiFI_data)

    ### fill all nan with linear
    first_seen_probe_data_path = os.path.join(first_seen_probe_data_dir, 'first_seen_probe_data.pkl')
    first_seen_probe_data = basic.load_variable_from_pikle_file(first_seen_probe_data_path)

    store_path = os.path.join(ImputeHiFI_data_dir,
                              version + str(para) + infer_3D_type + loci_pair_type + "_Impute_data.pkl")
    final_ImputeHiFI_data = basic.load_variable_from_pikle_file(store_path)
    Linear_impute_all_cell_dna_list = first_seen_probe_data['Linear_all_cell_dna_list']
    final_ImputeHiFI_data['Linear_all_cell_dna_list']= Linear_impute_all_cell_dna_list
    finall_optimize_all_cell_dna_list=final_ImputeHiFI_data['all_cell_dna_list']
    finall_optimize_all_cell_dna_list=ImputeHiFI.generate_impute_dna_list.fill_all_nan_chr_with_linear(finall_optimize_all_cell_dna_list,Linear_impute_all_cell_dna_list)
    final_ImputeHiFI_data['all_cell_dna_list']=finall_optimize_all_cell_dna_list
    store_path = os.path.join(ImputeHiFI_data_dir,
                              version + str(para) + infer_3D_type + loci_pair_type + "_Impute_data.pkl")
    basic.store_variable_from_pikle_file(store_path, final_ImputeHiFI_data)

    ### generate 3D spatial distance

    store_path = os.path.join(ImputeHiFI_data_dir,
                              version + str(para) + infer_3D_type + loci_pair_type + "_Impute_data.pkl")

    final_ImputeHiFI_data = basic.load_variable_from_pikle_file(store_path)

    chr_name_list = first_seen_probe_data['chr_name_list']
    cell_number = first_seen_probe_data['cell_number']
    all_cell_dna_list = first_seen_probe_data['all_cell_dna_list']
    chr_bin_length_dic = first_seen_probe_data['chr_bin_length_dic']
    chrom_index_to_ids_dict = first_seen_probe_data['chrom_index_to_ids_dict']
    adata_obs_df = first_seen_probe_data['adata_obs_df']
    spaital_dist_suffix = first_seen_probe_data['spaital_dist_suffix']
    prxoimtiy_score_suffix = first_seen_probe_data['prxoimtiy_score_suffix']

    Impute_all_cell_dna_list = final_ImputeHiFI_data['all_cell_dna_list']
    cell_number = final_ImputeHiFI_data['cell_number']

    store_path = os.path.join(ImputeHiFI_data_dir,
                              version + str(para) + infer_3D_type + loci_pair_type + f"_{spaital_dist_suffix}.h5ad")

    ImputeHiFI.generate_spatial_distance_h5ad.generate_spatial_h5ad(chr_name_list, cell_number, Impute_all_cell_dna_list,
                                                                    chr_bin_length_dic, chrom_index_to_ids_dict,
                                                                    adata_obs_df, store_path, )

    store_path = os.path.join(ImputeHiFI_data_dir,
                              version + str(para) + infer_3D_type + loci_pair_type + f"_{spaital_dist_suffix}.h5ad")


    ### generate proximity score

    importlib.reload(ImputeHiFI)
    store_path = os.path.join(ImputeHiFI_data_dir,
                              version + str(para) + infer_3D_type + loci_pair_type + f"_{spaital_dist_suffix}.h5ad")
    Impute_3D_spad_adata = sc.read(store_path)

    chr_cell_type_center_value_df = first_seen_probe_data['chr_cell_type_center_value_df']
    chr_name_list = first_seen_probe_data['chr_name_list']
    cell_number = first_seen_probe_data['cell_number']
    all_cell_dna_list = first_seen_probe_data['all_cell_dna_list']
    chr_bin_length_dic = first_seen_probe_data['chr_bin_length_dic']
    chrom_index_to_ids_dict = first_seen_probe_data['chrom_index_to_ids_dict']
    adata_obs_df = first_seen_probe_data['adata_obs_df']
    spaital_dist_suffix = first_seen_probe_data['spaital_dist_suffix']
    prxoimtiy_score_suffix = first_seen_probe_data['prxoimtiy_score_suffix']

    store_path = os.path.join(ImputeHiFI_data_dir,
                              version + str(para) + infer_3D_type + loci_pair_type + f"_{prxoimtiy_score_suffix}.h5ad")
    ImputeHiFI.generate_proximity_score_h5ad.generate_proximity_h5ad(Impute_3D_spad_adata, chr_name_list,
                                                                     chr_cell_type_center_value_df,
                                                                     chr_bin_length_dic, adata_obs_df, store_path,
                                                                     kernerl_number=None)

    Impute_3D_proxi_adata = sc.read(store_path)

def run_ImputeHiFI_mode_3(first_seen_probe_data_dir,neighbor_cell_num=100):
    # data_dir = r"/mnt/disk1/scfan/data/Cai_21_Nature"
    # current_probe_data_dir = r"/mnt/disk1/scfan/data/Cai_21_Nature/DNA_25kb"
    # first_seen_probe_data_dir = r"/mnt/disk1/scfan/data/Cai_21_Nature/DNA_25kb/Impute/"
    basic.recursive_mkdir(first_seen_probe_data_dir)

    ImputeHiFI_data_dir = os.path.join(first_seen_probe_data_dir, 'ImputeHiFI')
    basic.recursive_mkdir(ImputeHiFI_data_dir)

    ### proxi adata

    FISH_RMSD_name = 'All_cell_RMSD_dict'
    version = 'no_HiC_RMSD_not_use_mean_count'

    first_seen_probe_data_path = os.path.join(first_seen_probe_data_dir, 'first_seen_probe_data.pkl')
    first_seen_probe_data = basic.load_variable_from_pikle_file(first_seen_probe_data_path)

    chr_cell_type_center_value_df = first_seen_probe_data['chr_cell_type_center_value_df']
    chr_name_list = first_seen_probe_data['chr_name_list']
    cell_name_list=first_seen_probe_data['cell_name_list']
    cell_number = first_seen_probe_data['cell_number']
    all_cell_dna_list = first_seen_probe_data['all_cell_dna_list']
    chr_bin_length_dic = first_seen_probe_data['chr_bin_length_dic']
    chrom_index_to_ids_dict = first_seen_probe_data['chrom_index_to_ids_dict']
    adata_obs_df = first_seen_probe_data['adata_obs_df']
    spaital_dist_suffix = first_seen_probe_data['spaital_dist_suffix']
    prxoimtiy_score_suffix = first_seen_probe_data['prxoimtiy_score_suffix']
    missing_rate_df = first_seen_probe_data['missing_rate_df']
    FISH_nearest_neighbors_dic = first_seen_probe_data[FISH_RMSD_name]

    store_path = os.path.join(first_seen_probe_data_dir, f'first_seen_probe_data_{prxoimtiy_score_suffix}.h5ad')
    DNA_FISH_proxi_adata = sc.read(store_path)

    importlib.reload(ImputeHiFI_no_HiC)
    base_coef = 0.001
    HiC_weight = 0
    FISH_weight = 1
    short_range_HiC_weight_bound = 0.0
    long_range_HiC_weight_bound = 0.0
    all_range_diag_sum_weight_HiC_bound = 0.0
    short_range_diag = 0
    initial_guess = [HiC_weight, FISH_weight]
    verbose = False
    background_cell_num = neighbor_cell_num
    use_mean_count = 0
    para = ('bg_n' + str(background_cell_num) + '_sdn' + str(short_range_diag) + '_sd' + str(
        all_range_diag_sum_weight_HiC_bound) +
            '_sr' + str(short_range_HiC_weight_bound) + '_lr' + str(long_range_HiC_weight_bound))

    store_path = os.path.join(ImputeHiFI_data_dir, str(version) + "_proximity_score_adata_" + str(para) + ".h5ad")

    ImputeHiFI_no_HiC.ImputeHiFi_mode2_mode3.get_Impute_proximity_score(DNA_FISH_proxi_adata, missing_rate_df,
                                                                        cell_name_list, chr_bin_length_dic, base_coef,
                                                                        HiC_weight, FISH_weight, initial_guess,
                                                                        short_range_HiC_weight_bound,
                                                                        long_range_HiC_weight_bound,
                                                                        all_range_diag_sum_weight_HiC_bound,
                                                                        short_range_diag, background_cell_num,
                                                                        use_mean_count, FISH_nearest_neighbors_dic,
                                                                        store_path, kernerl_number=20)


    ### Impute 3D data

    first_seen_probe_data.keys()

    importlib.reload(ImputeHiFI)
    store_path = os.path.join(ImputeHiFI_data_dir, str(version) + "_proximity_score_adata_" + str(para) + ".h5ad")
    ImputeHiFI_adata = sc.read(store_path)

    chr_cell_type_center_value_df = first_seen_probe_data['chr_cell_type_center_value_df']
    chr_name_list = first_seen_probe_data['chr_name_list']
    cell_number = first_seen_probe_data['cell_number']
    all_cell_dna_list = first_seen_probe_data['all_cell_dna_list']
    chr_bin_length_dic = first_seen_probe_data['chr_bin_length_dic']
    chrom_index_to_ids_dict = first_seen_probe_data['chrom_index_to_ids_dict']
    chrom_ids_to_index_dict = first_seen_probe_data['chrom_ids_to_index_dict']
    adata_obs_df = first_seen_probe_data['adata_obs_df']
    spaital_dist_suffix = first_seen_probe_data['spaital_dist_suffix']
    prxoimtiy_score_suffix = first_seen_probe_data['prxoimtiy_score_suffix']
    Linear_impute_all_cell_dna_list = first_seen_probe_data['Linear_all_cell_dna_list']

    infer_3D_type = r"_part_loci_pair_weight_"
    loci_pair_type = 'no_diff'

    store_path = os.path.join(ImputeHiFI_data_dir,
                              version + str(para) + infer_3D_type + loci_pair_type + "_Impute_data.pkl")

    finall_optimize_all_cell_dna_list, diff_genes = ImputeHiFI.generate_impute_dna_list.get_impute_dna_list(
        chr_bin_length_dic, chr_name_list, ImputeHiFI_adata,
        all_cell_dna_list, Linear_impute_all_cell_dna_list,
        chr_cell_type_center_value_df, chrom_ids_to_index_dict,
        loci_pair_type, kernerl_number=None)

    final_ImputeHiFI_data = copy.deepcopy(first_seen_probe_data)
    final_ImputeHiFI_data['all_cell_dna_list'] = finall_optimize_all_cell_dna_list
    final_ImputeHiFI_data['ledien_diff_gene'] = diff_genes
    final_ImputeHiFI_data['cell_number']=len(finall_optimize_all_cell_dna_list)

    basic.store_variable_from_pikle_file(store_path, final_ImputeHiFI_data)

    ### fill all nan with linear
    first_seen_probe_data_path = os.path.join(first_seen_probe_data_dir, 'first_seen_probe_data.pkl')
    first_seen_probe_data = basic.load_variable_from_pikle_file(first_seen_probe_data_path)

    store_path = os.path.join(ImputeHiFI_data_dir,
                              version + str(para) + infer_3D_type + loci_pair_type + "_Impute_data.pkl")
    final_ImputeHiFI_data = basic.load_variable_from_pikle_file(store_path)
    Linear_impute_all_cell_dna_list = first_seen_probe_data['Linear_all_cell_dna_list']
    final_ImputeHiFI_data['Linear_all_cell_dna_list']= Linear_impute_all_cell_dna_list
    finall_optimize_all_cell_dna_list=final_ImputeHiFI_data['all_cell_dna_list']
    finall_optimize_all_cell_dna_list=ImputeHiFI.generate_impute_dna_list.fill_all_nan_chr_with_linear(finall_optimize_all_cell_dna_list,Linear_impute_all_cell_dna_list)
    final_ImputeHiFI_data['all_cell_dna_list']=finall_optimize_all_cell_dna_list
    store_path = os.path.join(ImputeHiFI_data_dir,
                              version + str(para) + infer_3D_type + loci_pair_type + "_Impute_data.pkl")
    basic.store_variable_from_pikle_file(store_path, final_ImputeHiFI_data)

    ### generate 3D spatial distance

    store_path = os.path.join(ImputeHiFI_data_dir,
                              version + str(para) + infer_3D_type + loci_pair_type + "_Impute_data.pkl")

    final_ImputeHiFI_data = basic.load_variable_from_pikle_file(store_path)

    chr_name_list = first_seen_probe_data['chr_name_list']
    cell_number = first_seen_probe_data['cell_number']
    all_cell_dna_list = first_seen_probe_data['all_cell_dna_list']
    chr_bin_length_dic = first_seen_probe_data['chr_bin_length_dic']
    chrom_index_to_ids_dict = first_seen_probe_data['chrom_index_to_ids_dict']
    adata_obs_df = first_seen_probe_data['adata_obs_df']
    spaital_dist_suffix = first_seen_probe_data['spaital_dist_suffix']
    prxoimtiy_score_suffix = first_seen_probe_data['prxoimtiy_score_suffix']

    Impute_all_cell_dna_list = final_ImputeHiFI_data['all_cell_dna_list']
    cell_number = final_ImputeHiFI_data['cell_number']

    store_path = os.path.join(ImputeHiFI_data_dir,
                              version + str(para) + infer_3D_type + loci_pair_type + f"_{spaital_dist_suffix}.h5ad")

    ImputeHiFI.generate_spatial_distance_h5ad.generate_spatial_h5ad(chr_name_list, cell_number, Impute_all_cell_dna_list,
                                                                    chr_bin_length_dic, chrom_index_to_ids_dict,
                                                                    adata_obs_df, store_path, )

    store_path = os.path.join(ImputeHiFI_data_dir,
                              version + str(para) + infer_3D_type + loci_pair_type + f"_{spaital_dist_suffix}.h5ad")


    ### generate proximity score

    importlib.reload(ImputeHiFI)
    store_path = os.path.join(ImputeHiFI_data_dir,
                              version + str(para) + infer_3D_type + loci_pair_type + f"_{spaital_dist_suffix}.h5ad")
    Impute_3D_spad_adata = sc.read(store_path)

    chr_cell_type_center_value_df = first_seen_probe_data['chr_cell_type_center_value_df']
    chr_name_list = first_seen_probe_data['chr_name_list']
    cell_number = first_seen_probe_data['cell_number']
    all_cell_dna_list = first_seen_probe_data['all_cell_dna_list']
    chr_bin_length_dic = first_seen_probe_data['chr_bin_length_dic']
    chrom_index_to_ids_dict = first_seen_probe_data['chrom_index_to_ids_dict']
    adata_obs_df = first_seen_probe_data['adata_obs_df']
    spaital_dist_suffix = first_seen_probe_data['spaital_dist_suffix']
    prxoimtiy_score_suffix = first_seen_probe_data['prxoimtiy_score_suffix']

    store_path = os.path.join(ImputeHiFI_data_dir,
                              version + str(para) + infer_3D_type + loci_pair_type + f"_{prxoimtiy_score_suffix}.h5ad")
    ImputeHiFI.generate_proximity_score_h5ad.generate_proximity_h5ad(Impute_3D_spad_adata, chr_name_list,
                                                                     chr_cell_type_center_value_df,
                                                                     chr_bin_length_dic, adata_obs_df, store_path,
                                                                     kernerl_number=None)

    Impute_3D_proxi_adata = sc.read(store_path)

def display_raw_data(first_seen_probe_data_dir,one_FISH_name,chr_name):
    prxoimtiy_score_suffix='proximity_score'
    store_path = os.path.join(first_seen_probe_data_dir, f'first_seen_probe_data_{prxoimtiy_score_suffix}.h5ad')
    DNA_FISH_proxi_adata = sc.read(store_path)
    # one_FISH_name = '0_3_1_rep1'
    # chr_name = 'chr1'
    first_seen_probe_data_path = os.path.join(first_seen_probe_data_dir, 'first_seen_probe_data.pkl')
    first_seen_probe_data = basic.load_variable_from_pikle_file(first_seen_probe_data_path)
    chr_bin_length_dic = first_seen_probe_data['chr_bin_length_dic']

    one_array = ImputeHiFI.get_proximity_array(DNA_FISH_proxi_adata, one_FISH_name, chr_name, chr_bin_length_dic,
                                               set_min_limit=True, min_limit=0.001, set_max_limit=False, max_limit=1.0)
    ImputeHiFI.plot_proximity(one_array, title=one_FISH_name + ' ' + chr_name)

def display_imputed_data(first_seen_probe_data_dir,one_FISH_name,chr_name,file_name=None,version = 'no_HiC_RNA_FISH_RMSD_not_use_mean_count',):
    prxoimtiy_score_suffix='proximity_score'
    first_seen_probe_data_path = os.path.join(first_seen_probe_data_dir, 'first_seen_probe_data.pkl')
    first_seen_probe_data = basic.load_variable_from_pikle_file(first_seen_probe_data_path)
    chr_bin_length_dic = first_seen_probe_data['chr_bin_length_dic']

    HiC_weight = 0
    FISH_weight = 1
    short_range_HiC_weight_bound = 0.0
    long_range_HiC_weight_bound = 0.0
    all_range_diag_sum_weight_HiC_bound = 0.0
    short_range_diag = 0
    initial_guess = [HiC_weight, FISH_weight]
    verbose = False
    background_cell_num = 100
    use_mean_count = 0
    para = ('bg_n' + str(background_cell_num) + '_sdn' + str(short_range_diag) + '_sd' + str(
        all_range_diag_sum_weight_HiC_bound) +
            '_sr' + str(short_range_HiC_weight_bound) + '_lr' + str(long_range_HiC_weight_bound))
    infer_3D_type = r"_part_loci_pair_weight_"
    loci_pair_type = 'no_diff'
    if file_name is None:
        ImputeHiFI_data_dir = os.path.join(first_seen_probe_data_dir, 'ImputeHiFI')
        store_path = os.path.join(ImputeHiFI_data_dir,
                                  version + str(para) + infer_3D_type + loci_pair_type + f"_{prxoimtiy_score_suffix}.h5ad")
    else:
        store_path = os.path.join(first_seen_probe_data_dir, file_name)
    DNA_FISH_proxi_adata = sc.read(store_path)


    one_array = ImputeHiFI.get_proximity_array(DNA_FISH_proxi_adata, one_FISH_name, chr_name, chr_bin_length_dic,
                                               set_min_limit=True, min_limit=0.001, set_max_limit=False, max_limit=1.0)
    ImputeHiFI.plot_proximity(one_array, title=one_FISH_name + ' ' + chr_name)