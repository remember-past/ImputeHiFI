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
print('v6 similar FISH no HiC')
#we can use FISH_nearest_neighbors_names_dic to specify what are the nearest neighbors of each cell under each chromosome.
#According to observations, whether using mean_count or bg_all_count will have a greater impact on the clustering effect.

#In this version, this becomes optional through the use_mean_count parameter.


def infer_distance_matrix(matrix_array, sigma):
    """
    Infers the distance matrix from a given matrix computed using the Gaussian function.

    Parameters:
        matrix_array (np.array): The matrix computed by the Gaussian function.
        sigma (float): The standard deviation of the Gaussian function.

    Returns:
        dist_matrix (np.array): The inferred distance matrix.
    """
    # Ensure all values in matrix_array are positive
    if np.any(matrix_array <= 0):
        raise ValueError("All elements in matrix_array must be positive.")

    # Compute the square of the distances
    d_squared = -np.log2(matrix_array) * sigma**2

    # Compute the distance matrix by taking the square root of the squared distances
    dist_matrix = np.sqrt(d_squared)

    return dist_matrix




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

def generate_DNA_FISH_df(one_FISH_name,chr_name,missing_rate_df,DNA_FISH_adata):
    one_cell_chr_info_df = missing_rate_df.loc[[(one_FISH_name, chr_name)]]
    bin_length = one_cell_chr_info_df['total_probe_number'].values[0]
    Example_df = generate_Example_bin_pair_df(chr_name, bin_length)
    Common_Example_df = copy.deepcopy(Example_df)
    Common_Example_df['diag'] = Common_Example_df['col'].astype(int) - Common_Example_df['row'].astype(int)
    diag_array = np.unique(Common_Example_df['diag'].values)

    Example_df['data'] = DNA_FISH_adata[one_FISH_name, Example_df.index].copy().X[0]
    all_count = np.nansum(Example_df['data'].values)
    Example_df['prop'] = Example_df['data'] / all_count
    # norm diag prop
    # one_diag=1
    diag_sum_prop = {}
    for one_diag in diag_array:
        one_diag_index = Common_Example_df[Common_Example_df['diag'] == one_diag].index
        temp_sum = np.nansum(Example_df.loc[one_diag_index]['prop'].values)
        diag_sum_prop[one_diag] = temp_sum
        if (temp_sum == 0):
            Example_df.loc[one_diag_index, 'diag_prop'] = 0.0
        else:
            Example_df.loc[one_diag_index, 'diag_prop'] = Example_df.loc[one_diag_index]['prop'] / temp_sum

    DNA_FISH_df = Example_df.copy()
    return DNA_FISH_df,diag_sum_prop,all_count,Common_Example_df,diag_array,bin_length
def generate_HiC_df(diag_array,Common_Example_df,multi_HiC_name_list,chr_name,bin_length,weight_multi_HiC_RNAcoef_array,HiC_adata):
    temp_HiC_Example_df_dict = {}
    # one_HiC_ids=0
    # one_HiC_name=multi_HiC_name_list[one_HiC_ids]
    for one_HiC_ids, one_HiC_name in enumerate(multi_HiC_name_list):
        Example_df = generate_Example_bin_pair_df(chr_name, bin_length)

        one_HiC_weight = weight_multi_HiC_RNAcoef_array[one_HiC_ids]
        Example_df['data'] = HiC_adata[one_HiC_name, Example_df.index].copy().X[0]
        all_count = np.nansum(Example_df['data'].values)
        Example_df['prop'] = Example_df['data'] / all_count
        Example_df['weight_prop'] = Example_df['prop'] * one_HiC_weight
        temp_HiC_Example_df_dict[one_HiC_name] = Example_df
    Example_df = generate_Example_bin_pair_df(chr_name, bin_length)
    Example_df['prop'] = 0.0
    for one_HiC_name, one_HiC_df in temp_HiC_Example_df_dict.items():
        Example_df['prop'] = Example_df['prop'] + one_HiC_df['weight_prop']

    # norm diag prop
    # one_diag=1
    diag_sum_prop = {}
    for one_diag in diag_array:
        one_diag_index = Common_Example_df[Common_Example_df['diag'] == one_diag].index
        temp_sum = np.nansum(Example_df.loc[one_diag_index]['prop'].values)
        diag_sum_prop[one_diag] = temp_sum
        if (temp_sum == 0):
            Example_df.loc[one_diag_index, 'diag_prop'] = 0.0
        else:
            Example_df.loc[one_diag_index, 'diag_prop'] = Example_df.loc[one_diag_index]['prop'] / temp_sum

    HiC_df = copy.deepcopy(Example_df)
    HiC_diag_sum_prop_dict = copy.deepcopy(diag_sum_prop)
    return HiC_df, HiC_diag_sum_prop_dict
def generate_background_df(diag_array,Common_Example_df,one_FISH_name,chr_name,bin_length,DNA_FISH_adata,FISH_nearest_neighbors_names_dic,background_cell_num):
    # background_cell_num=50
    Example_df = generate_Example_bin_pair_df(chr_name, bin_length)
    temp_array = DNA_FISH_adata[
        FISH_nearest_neighbors_names_dic[chr_name][one_FISH_name][0:background_cell_num], Example_df.index].copy().X

    Example_df['data'] = np.nanmedian(temp_array, axis=0)

    all_count = np.nansum(Example_df['data'].values)

    Example_df['prop'] = Example_df['data'] / all_count

    # norm diag prop
    # one_diag=1
    diag_sum_prop = {}
    for one_diag in diag_array:
        one_diag_index = Common_Example_df[Common_Example_df['diag'] == one_diag].index
        temp_sum = np.nansum(Example_df.loc[one_diag_index]['prop'].values)
        diag_sum_prop[one_diag] = temp_sum
        if (temp_sum == 0):
            Example_df.loc[one_diag_index, 'diag_prop'] = 0.0
        else:
            Example_df.loc[one_diag_index, 'diag_prop'] = Example_df.loc[one_diag_index]['prop'] / temp_sum

    bg_df = copy.deepcopy(Example_df)
    bg_diag_sum_prop_dict = copy.deepcopy(diag_sum_prop)
    return bg_df, bg_diag_sum_prop_dict,all_count
def generate_Impute_DNA_FISH_df(DNA_FISH_df,HiC_df,bg_df,Common_Example_df,diag_array,HiC_diag_sum_prop_dict,bg_diag_sum_prop_dict,DNA_FISH_weight,HiC_weight,background_weight,HiC_diag_weight,background_diag_weight):
    DNA_FISH_df['prop'].fillna(0, inplace=True)
    DNA_FISH_df['diag_prop'].fillna(0, inplace=True)

    HiC_df['prop'].fillna(0, inplace=True)
    HiC_df['diag_prop'].fillna(0, inplace=True)

    bg_df['prop'].fillna(0, inplace=True)
    bg_df['diag_prop'].fillna(0, inplace=True)

    merged_diag_sum_prop_dict = {}
    for one_diag in diag_array:
        temp_sum_prop = HiC_diag_sum_prop_dict[one_diag] * HiC_diag_weight + bg_diag_sum_prop_dict[
            one_diag] * background_diag_weight
        merged_diag_sum_prop_dict[one_diag] = temp_sum_prop

    Impute_DNA_FISH_df = copy.deepcopy(DNA_FISH_df)

    Impute_DNA_FISH_df['Impute_diag_prop'] = 0
    Impute_DNA_FISH_df['Impute_prop'] = 0

    for one_diag in diag_array:
        one_diag_index = Common_Example_df[Common_Example_df['diag'] == one_diag].index
        temp_DNA_FISH_diag = (DNA_FISH_df.loc[one_diag_index, 'diag_prop']) * DNA_FISH_weight
        temp_HiC_diag = (HiC_df.loc[one_diag_index, 'diag_prop']) * HiC_weight
        temp_bg_diag = (bg_df.loc[one_diag_index, 'diag_prop']) * background_weight

        Impute_DNA_FISH_df.loc[one_diag_index, 'Impute_diag_prop'] = temp_DNA_FISH_diag + temp_HiC_diag + temp_bg_diag
        #     print(one_diag,np.sum(Impute_DNA_FISH_df.loc[one_diag_index,'Impute_diag_prop']))
        merged_one_diag_sum_prop = merged_diag_sum_prop_dict[one_diag]
        Impute_DNA_FISH_df.loc[one_diag_index, 'Impute_prop'] = Impute_DNA_FISH_df.loc[
                                                                    one_diag_index, 'Impute_diag_prop'] * merged_one_diag_sum_prop

    prop_count = np.sum(Impute_DNA_FISH_df['Impute_prop'])
    Impute_DNA_FISH_df['Impute_prop_norm'] = Impute_DNA_FISH_df['Impute_prop'] / prop_count
    return Impute_DNA_FISH_df
def generate_sample_df(chr_name,bin_length,sample_range,sample_time,sample_prop):
    sample_result = np.random.choice(sample_range, size=sample_range * sample_time, replace=True, p=sample_prop)

    Example_df = generate_Example_bin_pair_df(chr_name, bin_length, data=1)
    temp_Example_df = Example_df.iloc[sample_result]
    temp_Example_df.reset_index(drop=True, inplace=True)

    sum_by_chr_bin_bin = temp_Example_df.groupby('chr_bin_bin')['data'].sum().reset_index()

    sum_by_chr_bin_bin.index = sum_by_chr_bin_bin['chr_bin_bin']

    Example_df.loc[sum_by_chr_bin_bin.index, 'data'] = sum_by_chr_bin_bin['data']
    return Example_df

def fill_nan_with_interpolation(distmap_list_median):

    nan_locs = np.isnan(distmap_list_median)
    not_nan_locs = ~nan_locs
    not_nan_values = distmap_list_median[not_nan_locs]
    not_nan_positions = np.array(np.nonzero(not_nan_locs)).T


    nan_positions = np.array(np.nonzero(nan_locs)).T


    interpolated_values = griddata(not_nan_positions, not_nan_values, nan_positions, method='nearest')


    filled_distmap_list_median = distmap_list_median.copy()
    filled_distmap_list_median[nan_locs] = interpolated_values

    return filled_distmap_list_median

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import minimize

# 定义目标函数
def objective(weights, P_FISH_observed, P_HiC_observed, P_neighbor_FISH_observed):
    a, b = weights
    a=0
    b=weights[1]
    prediction = a * P_HiC_observed + b * P_neighbor_FISH_observed
    return np.sum((prediction - P_FISH_observed) ** 2)
def get_predicted(weights, P_FISH_observed, P_HiC_observed, P_neighbor_FISH_observed):
    a, b = weights
    a=0
    b=weights[1]
    prediction = a * P_HiC_observed + b * P_neighbor_FISH_observed
    return prediction
# 使用最佳权重填补缺失数据
def impute_missing_data(weights, P_HiC_missing, P_neighbor_FISH_missing):
    a, b = weights
    a=0
    b=weights[1]
    return a * P_HiC_missing + b * P_neighbor_FISH_missing
def safe_normalize(array):
    array_sum = array.sum()
    if array_sum == 0:
        return np.zeros_like(array)
    else:
        return array / array_sum
def extract_data(one_diag_index, DNA_FISH_df, HiC_df, bg_df):
    P_FISH = DNA_FISH_df.loc[one_diag_index, 'diag_prop'].values
    P_HiC = HiC_df.loc[one_diag_index, 'diag_prop'].values
    P_neighbor_FISH = bg_df.loc[one_diag_index, 'diag_prop'].values
    P_neighbor_FISH = np.nan_to_num(P_neighbor_FISH, 0)
    # 区分观测到的数据和缺失的数据
    observed_mask = ~DNA_FISH_df.loc[one_diag_index, 'data'].isna()
    missing_mask = DNA_FISH_df.loc[one_diag_index, 'data'].isna()

    # 创建观测到的数据和缺失的数据的数组
    P_FISH_observed = DNA_FISH_df.loc[one_diag_index[observed_mask], 'diag_prop'].values
    # P_FISH_missing = DNA_FISH_df.loc[one_diag_index[missing_mask], 'diag_prop'].values

    # 如果需要，也可以创建包含其他列的DataFrame
    # P_FISH_observed_df = DNA_FISH_df.loc[one_diag_index[observed_mask]]
    # P_FISH_missing_df = DNA_FISH_df.loc[one_diag_index[missing_mask]]

    # 现在你有了观测到的数据和缺失的数据的索引，可以用于后续的插补步骤
    # 提取 P_HiC 和 P_neighbor_FISH 中观测到的数据和缺失的数据
    P_HiC_observed = HiC_df.loc[one_diag_index[observed_mask], 'diag_prop'].values
    P_HiC_missing = HiC_df.loc[one_diag_index[missing_mask], 'diag_prop'].values

    P_neighbor_FISH_observed = bg_df.loc[one_diag_index[observed_mask], 'diag_prop'].values
    P_neighbor_FISH_missing = bg_df.loc[one_diag_index[missing_mask], 'diag_prop'].values
    P_neighbor_FISH_observed = np.nan_to_num(P_neighbor_FISH_observed, 0)
    P_neighbor_FISH_missing = np.nan_to_num(P_neighbor_FISH_missing, 0)

    # P_FISH_observed_norm = safe_normalize(P_FISH_observed)
    # P_HiC_observed_norm = safe_normalize(P_HiC_observed)
    # P_HiC_missing_norm = safe_normalize(P_HiC_missing)
    # P_neighbor_FISH_observed_norm = safe_normalize(P_neighbor_FISH_observed)
    # P_neighbor_FISH_missing_norm = safe_normalize(P_neighbor_FISH_missing)
    #
    # # 如果需要，也可以创建包含其他列的DataFrame
    # P_HiC_observed_df = HiC_df.loc[one_diag_index[observed_mask]]
    # P_HiC_missing_df = HiC_df.loc[one_diag_index[missing_mask]]
    #
    # P_neighbor_FISH_observed_df = bg_df.loc[one_diag_index[observed_mask]]
    # P_neighbor_FISH_missing_df = bg_df.loc[one_diag_index[missing_mask]]
    return P_FISH_observed, P_HiC_observed, P_HiC_missing,P_neighbor_FISH_observed, P_neighbor_FISH_missing,observed_mask, missing_mask

# 执行优化
def perform_optimization(objective, initial_guess, P_FISH_observed_norm, P_HiC_observed_norm,
                P_neighbor_FISH_observed_norm, constraints, diag_sum_constraints):
    result = minimize(
        objective,
        initial_guess,
        args=(P_FISH_observed_norm, P_HiC_observed_norm, P_neighbor_FISH_observed_norm),
        constraints=constraints,
        method='SLSQP'
    )

    fitted_weights = result.x
    result = minimize(
        objective,
        initial_guess,
        args=(P_FISH_observed_norm, P_HiC_observed_norm, P_neighbor_FISH_observed_norm),
        constraints=diag_sum_constraints,
        method='SLSQP'
    )
    diag_fitted_weights = result.x
    return fitted_weights, diag_fitted_weights


def optimize_and_impute_data(
    diag_array, Common_Example_df, DNA_FISH_df, HiC_df, bg_df,
    initial_guess, short_range_diag, short_range_HiC_weight_bound,
    long_range_HiC_weight_bound, all_range_diag_sum_weight_HiC_bound,
    HiC_diag_sum_prop_dict, bg_diag_sum_prop_dict, bg_all_count,use_mean_count,
):

    # Set up the constraints for the optimization problem
    diag_sum_constraints = (
        {'type': 'eq', 'fun': lambda weights: weights[0]},  # a =0 no Hi-C weight
        {'type': 'eq', 'fun': lambda weights: weights[1]-1},  # b = 1
        # {'type': 'ineq', 'fun': lambda weights: weights[1]},  # b >= 0
        # {'type': 'ineq', 'fun': lambda weights: 1 - weights[1]},  # b <= 1
        # {'type': 'eq', 'fun': lambda weights: weights[0] + weights[1] - 1}  # a + b = 1
    )

    # Initialize dictionaries to store the weights, losses, and updated data
    diag_learning_weight_dict = {}
    diag_learning_weight_for_diag_sum_dict = {}
    diag_init_loss = {}
    diag_learning_loss = {}
    DNA_Impute_FISH_df = DNA_FISH_df.copy()
    DNA_Impute_diag_sum_prop_dict = {}

    # Iterate over each diagonal in the diag_array
    for one_diag in diag_array:
        # Set the constraints based on the diagonal range
        if one_diag <= short_range_diag:
            min_HiC_weight = short_range_HiC_weight_bound
        else:
            min_HiC_weight = long_range_HiC_weight_bound
        constraints = (
            {'type': 'eq', 'fun': lambda weights: weights[0] },  # no Hi-C weight
            {'type': 'ineq', 'fun': lambda weights: weights[1]},  # b >= 0
            # {'type': 'ineq', 'fun': lambda weights: 1-weights[1]},  # b <= 1
            # {'type': 'eq', 'fun': lambda weights: weights[0] + weights[1] - 1}  # a + b = 1
        )

        # Get the indices for the current diagonal
        one_diag_index = Common_Example_df[Common_Example_df['diag'] == one_diag].index

        # Extract the observed and missing data for FISH, HiC, and neighboring FISH
        P_FISH_observed, P_HiC_observed, P_HiC_missing,P_neighbor_FISH_observed, P_neighbor_FISH_missing,observed_mask, missing_mask = extract_data(
            one_diag_index, DNA_FISH_df, HiC_df, bg_df
        )

        # Normalize the observed and missing data
        P_FISH_observed_norm = safe_normalize(P_FISH_observed)
        P_HiC_observed_norm = safe_normalize(P_HiC_observed)
        P_HiC_missing_norm = safe_normalize(P_HiC_missing)
        P_neighbor_FISH_observed_norm = safe_normalize(P_neighbor_FISH_observed)
        P_neighbor_FISH_missing_norm = safe_normalize(P_neighbor_FISH_missing)

        # Optimize the weights if there are observed data points
        if len(P_FISH_observed) == 0 or len(one_diag_index) == 1 or P_FISH_observed.sum() == 0:
            fitted_weights = initial_guess
            diag_fitted_weights = initial_guess
        else:
            fitted_weights, diag_fitted_weights = perform_optimization(
                objective, initial_guess, P_FISH_observed_norm, P_HiC_observed_norm,
                P_neighbor_FISH_observed_norm, constraints, diag_sum_constraints
            )

        # Impute the missing data using the optimized weights
        P_FISH_missing_filled = impute_missing_data(fitted_weights, P_HiC_missing_norm, P_neighbor_FISH_missing_norm)

        # Calculate the initial and learned losses for diagnostics
        diag_learning_weight_dict[one_diag] = fitted_weights
        diag_learning_weight_for_diag_sum_dict[one_diag] = diag_fitted_weights
        diag_init_loss[one_diag] = objective(initial_guess, P_FISH_observed_norm, P_HiC_observed_norm, P_neighbor_FISH_observed_norm)
        diag_learning_loss[one_diag] = objective(fitted_weights, P_FISH_observed_norm, P_HiC_observed_norm, P_neighbor_FISH_observed_norm)

        # Update the original data frame with the imputed data
        a, b =fitted_weights
        P_observed_impute_sum=a*P_HiC_observed.sum()+b*P_neighbor_FISH_observed.sum()
        if P_FISH_observed.sum()>0 and P_observed_impute_sum==0:
            observed_len=one_diag_index[observed_mask].shape[0]
            missing_len=one_diag_index[missing_mask].shape[0]
            P_observed_impute_sum=observed_len/(observed_len+missing_len)
            P_missing_impute_sum=1-P_observed_impute_sum
        else:
            P_missing_impute_sum=1-P_observed_impute_sum
        P_FISH_missing_filled=P_FISH_missing_filled*P_missing_impute_sum

        P_FISH_observed_filled=P_FISH_observed_norm*P_observed_impute_sum

        # 更新原始数据框中缺失数据的估计值

        DNA_Impute_FISH_df.loc[one_diag_index[missing_mask], 'diag_prop'] = P_FISH_missing_filled
        DNA_Impute_FISH_df.loc[one_diag_index[observed_mask], 'diag_prop'] = P_FISH_observed_filled
        temp_sum=DNA_Impute_FISH_df.loc[one_diag_index]['diag_prop'].sum()
        if temp_sum>0:
            DNA_Impute_FISH_df.loc[one_diag_index, 'diag_prop']=DNA_Impute_FISH_df.loc[one_diag_index, 'diag_prop']/temp_sum
        diag_sum_a,diag_sum_b=diag_fitted_weights
    #     DNA_Impute_diag_sum_prop_dict[one_diag]=HiC_diag_sum_prop_dict[one_diag]*a+bg_diag_sum_prop_dict[one_diag]*b
        DNA_Impute_diag_sum_prop_dict[one_diag]=HiC_diag_sum_prop_dict[one_diag]*diag_sum_a+bg_diag_sum_prop_dict[one_diag]*diag_sum_b

    DNA_Impute_diag_sum_prop_norm_dict = {}

    DNA_Impute_diag_sum_prop_array = np.array(
        [DNA_Impute_diag_sum_prop_dict[key] for key in DNA_Impute_diag_sum_prop_dict.keys()])

    temp_sum = DNA_Impute_diag_sum_prop_array.sum()
    for key, values in DNA_Impute_diag_sum_prop_dict.items():
        DNA_Impute_diag_sum_prop_norm_dict[key] = values / temp_sum

    DNA_Impute_FISH_df['Impute_data'] = DNA_Impute_FISH_df['data']
    for one_diag in diag_array:
        #     one_diag=1
        one_diag_index = Common_Example_df[Common_Example_df['diag'] == one_diag].index
        DNA_Impute_FISH_df.loc[one_diag_index, 'prop'] = DNA_Impute_diag_sum_prop_norm_dict[one_diag] * \
                                                         DNA_Impute_FISH_df.loc[one_diag_index, 'diag_prop']
        observed_mask = ~DNA_FISH_df.loc[one_diag_index, 'data'].isna()
        # missing_mask = DNA_FISH_df.loc[one_diag_index, 'data'].isna()
        temp_DNA_Impute_FISH_df = DNA_Impute_FISH_df.loc[one_diag_index[observed_mask]]
        count_series = temp_DNA_Impute_FISH_df['data'] / temp_DNA_Impute_FISH_df['prop']
        mean_count = count_series.mean()
        # diag_series_dict[one_diag] = mean_count

        if use_mean_count:
            if (np.isnan(mean_count)):
                DNA_Impute_FISH_df.loc[one_diag_index, 'Impute_data'] = DNA_Impute_FISH_df.loc[
                                                                            one_diag_index, 'prop'] * bg_all_count
            #         print(one_diag)
            else:
                DNA_Impute_FISH_df.loc[one_diag_index, 'Impute_data'] = DNA_Impute_FISH_df.loc[
                                                                               one_diag_index, 'prop'] * mean_count
        else:
            DNA_Impute_FISH_df.loc[one_diag_index, 'Impute_data'] = DNA_Impute_FISH_df.loc[one_diag_index, 'prop'] * bg_all_count

    DNA_Impute_FISH_df['data'] = DNA_Impute_FISH_df['Impute_data']
    # DNA_Impute_FISH_df['data'] = DNA_Impute_FISH_df['prop'] * bg_all_count
    # Impute_df_dict[chr_name] = DNA_Impute_FISH_df

    return DNA_Impute_FISH_df, diag_learning_weight_dict, diag_learning_weight_for_diag_sum_dict, diag_init_loss, diag_learning_loss
class ImputeHiFi_mode2_mode3(object):
    @staticmethod
    def generate_Impute_proximity_score_result_part(chr_bin_length_dic, initial_guess,short_range_HiC_weight_bound,
                               long_range_HiC_weight_bound, all_range_diag_sum_weight_HiC_bound,
                               short_range_diag, background_cell_num,
                               DNA_FISH_name_list, missing_rate_df, DNA_FISH_adata,
                               FISH_nearest_neighbors_names_dic, use_mean_count):
        Impute_DNA_FISH_df_dict = {}
        for one_FISH_name in tqdm(DNA_FISH_name_list):
            # chr_name='chr1'
            Impute_df_dict = {}
            for chr_name in chr_bin_length_dic.keys():
                DNA_FISH_df, diag_sum_prop, DNA_FISH_all_count, Common_Example_df, diag_array, bin_length = generate_DNA_FISH_df(
                    one_FISH_name, chr_name, missing_rate_df, DNA_FISH_adata)

                # HiC_df, HiC_diag_sum_prop_dict = ImputeHiFI.generate_HiC_df(diag_array, Common_Example_df,
                #                                                             multi_HiC_name_list, chr_name, bin_length,
                #                                                             weight_multi_HiC_RNAcoef_array, HiC_adata)

                bg_df, bg_diag_sum_prop_dict, bg_all_count = generate_background_df(diag_array,
                                                                                               Common_Example_df,
                                                                                               one_FISH_name, chr_name,
                                                                                               bin_length,
                                                                                               DNA_FISH_adata,
                                                                                               FISH_nearest_neighbors_names_dic,
                                                                                               background_cell_num)
                HiC_df = copy.deepcopy(bg_df)
                HiC_diag_sum_prop_dict = copy.deepcopy(bg_diag_sum_prop_dict)

                DNA_Impute_FISH_df, diag_learning_weight_dict, diag_learning_weight_for_diag_sum_dict, diag_init_loss, diag_learning_loss = optimize_and_impute_data(
                    diag_array, Common_Example_df, DNA_FISH_df, HiC_df, bg_df,
                    initial_guess, short_range_diag, short_range_HiC_weight_bound,
                    long_range_HiC_weight_bound, all_range_diag_sum_weight_HiC_bound,
                    HiC_diag_sum_prop_dict, bg_diag_sum_prop_dict, bg_all_count, use_mean_count)
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

    @staticmethod
    def get_Impute_proximity_score(DNA_FISH_adata,missing_rate_df,cell_name_list,chr_bin_length_dic,base_coef,HiC_weight,FISH_weight,initial_guess,short_range_HiC_weight_bound,
                                 long_range_HiC_weight_bound, all_range_diag_sum_weight_HiC_bound,
                                    short_range_diag, background_cell_num,use_mean_count,FISH_nearest_neighbors_dic,store_path,kernerl_number=32):
        FISH_nearest_neighbors_names_dic = FISH_nearest_neighbors_dic['nearest_neighbors_names_dic']

        parameter_dic = {}
        parameter_dic['base_coef'] = base_coef
        parameter_dic['HiC_weight'] = HiC_weight
        parameter_dic['FISH_weight'] = FISH_weight
        parameter_dic['short_range_HiC_weight_bound'] = short_range_HiC_weight_bound
        parameter_dic['long_range_HiC_weight_bound'] = long_range_HiC_weight_bound
        parameter_dic['all_range_diag_sum_weight_HiC_bound'] = all_range_diag_sum_weight_HiC_bound
        parameter_dic['short_range_diag'] = short_range_diag
        parameter_dic['background_cell_num'] = background_cell_num
        parameter_dic['use_mean_count'] = use_mean_count

        DNA_FISH_name_split_list = np.array_split(cell_name_list, kernerl_number)

        _domain_args = [(chr_bin_length_dic, initial_guess,short_range_HiC_weight_bound,
                               long_range_HiC_weight_bound, all_range_diag_sum_weight_HiC_bound,
                               short_range_diag, background_cell_num,
                               DNA_FISH_name_list, missing_rate_df, DNA_FISH_adata,
                               FISH_nearest_neighbors_names_dic, use_mean_count)
                        for DNA_FISH_name_list in DNA_FISH_name_split_list]
        import multiprocessing as mp

        with mp.Pool(kernerl_number) as domain_pool:
            domain_results = domain_pool.starmap(ImputeHiFi_mode2_mode3.generate_Impute_proximity_score_result_part, _domain_args)
            domain_pool.close()
            domain_pool.join()
            domain_pool.terminate()

        result_dic = {}
        for result in domain_results:
            print("Collect result")
            for cell_name, df in result.items():
                result_dic[cell_name] = df
        Example_dict = {}
        for chr_name, bin_length in chr_bin_length_dic.items():
            Example_dict[chr_name] = generate_Example_bin_pair_df(chr_name, bin_length)

        all_index_df = pd.concat(Example_dict.values(), axis=0)
        all_cell_df = pd.DataFrame(index=all_index_df.index)
        for cell_name, df in result_dic.items():
            all_cell_df[cell_name] = df.loc[all_cell_df.index, 'data'].values
        Impute_adata = sc.AnnData(all_cell_df.T)
        Impute_adata.obs = DNA_FISH_adata.obs.loc[Impute_adata.obs_names]
        Impute_adata.uns['parameter'] = parameter_dic

        Impute_adata.write(store_path)
def get_RNA_FISH_nearset_neighbors(data_dir):
    from scipy.spatial import cKDTree
    temp_path = os.path.join(data_dir, 'RNA', 'Cai_RNA_adata.h5ad')
    Cai_adata = sc.read(temp_path)
    adata = Cai_adata.copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    # sc.pp.highly_variable_genes(adata, n_top_genes=15000,min_mean=0.0125, max_mean=3, min_disp=0.5)
    # adata = adata[:, adata.var['highly_variable']]

    sc.pp.pca(adata, n_comps=30)
    # sc.pl.pca(adata,color='cell_type')
    sc.pp.neighbors(adata, n_pcs=30, n_neighbors=100, use_rep='X_pca')
    sc.tl.umap(adata)
    adata_tree = cKDTree(adata.obsm['X_umap'])

    # 对于 Transcriptom 数据集中的每个细胞，找到在 3D Genome 数据集中的最近邻

    neighbor_num = adata.shape[0]
    nearest_neighbors_distances, nearest_neighbors_indices = adata_tree.query(adata.obsm['X_umap'], k=neighbor_num)

    RNA_FISH_adata = adata.copy()
    RNA_FISH_nearest_neighbors_distances = nearest_neighbors_distances
    RNA_FISH_nearest_neighbors_indices = nearest_neighbors_indices
    RNA_FISH_nearest_neighbors_indices_dic = {}
    RNA_FISH_nearest_neighbors_names_dic = {}
    for idx, one_name in enumerate(adata.obs_names):
        RNA_FISH_nearest_neighbors_indices_dic[one_name] = RNA_FISH_nearest_neighbors_indices[idx][1:]
        RNA_FISH_nearest_neighbors_names_dic[one_name] = adata.obs_names[RNA_FISH_nearest_neighbors_indices[idx][1:]]
    RNA_FISH_nearest_neighbors_distances_dic = {}
    for idx, one_name in enumerate(adata.obs_names):
        RNA_FISH_nearest_neighbors_distances_dic[one_name] = RNA_FISH_nearest_neighbors_distances[idx][1:]
    return RNA_FISH_nearest_neighbors_names_dic