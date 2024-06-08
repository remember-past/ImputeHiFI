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


data_dir=r"/mnt/disk1/scfan/data/Cai_21_bioRxiv"
if __name__ == '__main__':
    data_dir = r"/mnt/disk1/scfan/data/Cai_21_bioRxiv"
    first_seen_probe_data_dir = os.path.join(data_dir, 'ImputeHiFI/first_seen_probe_data')

    first_seen_probe_data_path = os.path.join(first_seen_probe_data_dir, 'first_seen_probe_data.pkl')
    first_seen_probe_data = basic.load_variable_from_pikle_file(first_seen_probe_data_path)

    chr_name_list=first_seen_probe_data['chr_name_list']
    cell_number=first_seen_probe_data['cell_number']
    all_cell_dna_list=first_seen_probe_data['all_cell_dna_list']
    chr_bin_length_dic=first_seen_probe_data['chr_bin_length_dic']
    chrom_index_to_ids_dict=first_seen_probe_data['chrom_index_to_ids_dict']
    adata_obs_df=first_seen_probe_data['adata_obs_df']
    spaital_dist_suffix=first_seen_probe_data['spaital_dist_suffix']
    prxoimtiy_score_suffix=first_seen_probe_data['prxoimtiy_score_suffix']

    store_path = os.path.join(first_seen_probe_data_dir, f'first_seen_probe_data_{spaital_dist_suffix}.h5ad')
    ImputeHiFI.generate_spatial_distance_h5ad.generate_spatial_h5ad(chr_name_list, cell_number, all_cell_dna_list,
                                                                    chr_bin_length_dic, chrom_index_to_ids_dict,
                                                                    adata_obs_df, store_path, )



    store_path=os.path.join(first_seen_probe_data_dir,f'first_seen_probe_data_{spaital_dist_suffix}.h5ad')
    DNA_FISH_spad_adata=sc.read(store_path)

    chr_cell_type_center_value_df=first_seen_probe_data['chr_cell_type_center_value_df']
    chr_name_list=first_seen_probe_data['chr_name_list']
    cell_number=first_seen_probe_data['cell_number']
    all_cell_dna_list=first_seen_probe_data['all_cell_dna_list']
    chr_bin_length_dic=first_seen_probe_data['chr_bin_length_dic']
    chrom_index_to_ids_dict=first_seen_probe_data['chrom_index_to_ids_dict']
    adata_obs_df=first_seen_probe_data['adata_obs_df']
    spaital_dist_suffix=first_seen_probe_data['spaital_dist_suffix']
    prxoimtiy_score_suffix=first_seen_probe_data['prxoimtiy_score_suffix']


    store_path=os.path.join(first_seen_probe_data_dir,f'first_seen_probe_data_{prxoimtiy_score_suffix}.h5ad')
    ImputeHiFI.generate_proximity_score_h5ad.generate_proximity_h5ad(DNA_FISH_spad_adata,chr_name_list,chr_cell_type_center_value_df,
                                  chr_bin_length_dic, adata_obs_df, store_path,
                                  kernerl_number=None)