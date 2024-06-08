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
print('v5 similar FISH')
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



def calculate_max_corr_stratum(hic_matrix, fish_matrix, bin_length):
    """
    Calculate the end_stratum corresponding to the maximum Spearman correlation coefficient
    for given Hi-C and FISH matrices.

    Parameters:
    hic_matrix : np.array
        Matrix of Hi-C data.
    fish_matrix : np.array
        Matrix of FISH data.
    bin_length : int
        Bin length of the chromosome.

    Returns:
    int
        The end_stratum corresponding to the maximum Spearman correlation coefficient.
    """
    from scipy.stats import spearmanr
    spearman_correlation_list = []
    end_stratum_list = []

    # Iterate over all possible end_stratum values
    for end_stratum in range(2, bin_length + 1):
        # Calculate the Spearman correlation coefficient for each end_stratum
        spearman_corr, _ = spearmanr(hic_matrix[:end_stratum, :end_stratum], fish_matrix[:end_stratum, :end_stratum])
        spearman_correlation_list.append(spearman_corr)
        end_stratum_list.append(end_stratum)

    # Create a DataFrame to store correlation coefficients and end_stratum values
    correlation_data = pd.DataFrame({
        'End Stratum': end_stratum_list,
        'Spearman Correlation': spearman_correlation_list
    })

    # Find the index of the maximum correlation coefficient
    max_corr_index = correlation_data['Spearman Correlation'].idxmax()
    max_corr_stratum = correlation_data.loc[max_corr_index, 'End Stratum']

    return max_corr_stratum


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
    prediction = a * P_HiC_observed + b * P_neighbor_FISH_observed
    return np.sum((prediction - P_FISH_observed) ** 2)
def get_predicted(weights, P_FISH_observed, P_HiC_observed, P_neighbor_FISH_observed):
    a, b = weights
    prediction = a * P_HiC_observed + b * P_neighbor_FISH_observed
    return prediction
# 使用最佳权重填补缺失数据
def impute_missing_data(weights, P_HiC_missing, P_neighbor_FISH_missing):
    a, b = weights
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

    # 检查优化是否成功
    if result.success:
        fitted_weights = result.x
        # 使用最佳权重填补缺失数据
    #             P_FISH_missing_filled = impute_missing_data(fitted_weights, P_HiC_missing_norm, P_neighbor_FISH_missing_norm)
    else:
        raise ValueError(result.message)
    result = minimize(
        objective,
        initial_guess,
        args=(P_FISH_observed_norm, P_HiC_observed_norm, P_neighbor_FISH_observed_norm),
        constraints=diag_sum_constraints,
        method='SLSQP'
    )
    # 检查优化是否成功
    if result.success:
        diag_fitted_weights = result.x
    else:
        raise ValueError(result.message)
    return fitted_weights, diag_fitted_weights


def optimize_and_impute_data(
    diag_array, Common_Example_df, DNA_FISH_df, HiC_df, bg_df,
    initial_guess, short_range_diag, short_range_HiC_weight_bound,
    long_range_HiC_weight_bound, all_range_diag_sum_weight_HiC_bound,
    HiC_diag_sum_prop_dict, bg_diag_sum_prop_dict, bg_all_count,use_mean_count,
):
    # Set up the constraints for the optimization problem
    diag_sum_constraints = (
        {'type': 'ineq', 'fun': lambda weights: weights[0] - all_range_diag_sum_weight_HiC_bound},  # a >= all_range_diag_sum_weight_HiC_bound
        {'type': 'ineq', 'fun': lambda weights: weights[1]},  # b >= 0
        {'type': 'eq', 'fun': lambda weights: weights[0] + weights[1] - 1}  # a + b = 1
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
            {'type': 'ineq', 'fun': lambda weights: weights[0] - min_HiC_weight},  # a >= min_HiC_weight
            {'type': 'ineq', 'fun': lambda weights: weights[1]},  # b >= 0
            {'type': 'eq', 'fun': lambda weights: weights[0] + weights[1] - 1}  # a + b = 1
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
            # print('use mean_count')
            if (np.isnan(mean_count)):
                DNA_Impute_FISH_df.loc[one_diag_index, 'Impute_data'] = DNA_Impute_FISH_df.loc[
                                                                            one_diag_index, 'prop'] * bg_all_count
            #         print(one_diag)
            else:
                DNA_Impute_FISH_df.loc[one_diag_index, 'Impute_data'] = DNA_Impute_FISH_df.loc[
                                                                               one_diag_index, 'prop'] * mean_count
        else:
            # print('use bg_all_count')
            DNA_Impute_FISH_df.loc[one_diag_index, 'Impute_data'] = DNA_Impute_FISH_df.loc[one_diag_index, 'prop'] * bg_all_count

    DNA_Impute_FISH_df['data'] = DNA_Impute_FISH_df['Impute_data']
    # DNA_Impute_FISH_df['data'] = DNA_Impute_FISH_df['prop'] * bg_all_count
    # Impute_df_dict[chr_name] = DNA_Impute_FISH_df

    return DNA_Impute_FISH_df, diag_learning_weight_dict, diag_learning_weight_for_diag_sum_dict, diag_init_loss, diag_learning_loss
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

def get_RNA_FISH_leiden(data_dir):
    temp_path = os.path.join(data_dir, 'RNA', 'Cai_RNA_adata.h5ad')
    Cai_adata = sc.read(temp_path)
    adata = Cai_adata.copy()

    # Normalization, logarithmization, and scaling
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)

    # Principal component analysis
    sc.pp.pca(adata, n_comps=30)

    # Computing the neighborhood graph
    sc.pp.neighbors(adata, n_pcs=30, n_neighbors=100, use_rep='X_pca')

    # Run UMAP dimensionality reduction
    sc.tl.umap(adata)

    # Clustering the data using the Leiden algorithm
    sc.tl.leiden(adata, resolution=1)
    return adata

def get_proximity_array(adata, one_FISH_name, chr_name,chr_bin_length_dic,set_min_limit=True,min_limit=0.001,set_max_limit=False,max_limit=1.0):
    bin_length = chr_bin_length_dic[chr_name]
    Example_df = generate_Example_bin_pair_df(chr_name, bin_length)
    Example_df['data'] = adata[one_FISH_name, Example_df.index].copy().X[0]
    temp_array = convert_bin_pair_df_to_array(bin_length, Example_df)
    if set_min_limit:
        temp_array[temp_array <=0] = min_limit
    if set_max_limit:
        temp_array[temp_array > 1] = max_limit
    proximity_array = temp_array + temp_array.T
    np.fill_diagonal(proximity_array, 1)
    return proximity_array

def get_proximity_array_for_list(adata, one_FISH_name, chr_name, chr_bin_length_dic, aggregation_method='median',
                                 set_min_limit=True, min_limit=0.001, set_max_limit=False, max_limit=1.0):
    bin_length = chr_bin_length_dic[chr_name]
    Example_df = generate_Example_bin_pair_df(chr_name, bin_length)
    temp_array = adata[one_FISH_name, Example_df.index].copy().X

    # 根据聚合方式计算聚合值
    if aggregation_method == 'median':
        Example_df['data'] = np.nanmedian(temp_array, axis=0)
    elif aggregation_method == 'mean':
        Example_df['data'] = np.nanmean(temp_array, axis=0)
    elif aggregation_method == 'sum':
        Example_df['data'] = np.nansum(temp_array, axis=0)
    else:
        raise ValueError("Invalid aggregation method. Choose 'median', 'mean', or 'sum'.")

    temp_array = convert_bin_pair_df_to_array(bin_length, Example_df)

    if set_min_limit:
        temp_array[temp_array <= 0] = min_limit
    if set_max_limit:
        temp_array[temp_array > max_limit] = max_limit

    proximity_array = temp_array + temp_array.T

    for i in range(proximity_array.shape[0] - 1):  # 减1是为了避免在最后一个元素时越界访问
        if not np.isnan(proximity_array[i, i + 1]):
            proximity_array[i, i] = 1
        else:
            proximity_array[i, i] = np.nan

    # 单独处理最后一个对角线元素，根据左侧元素来决定
    i = proximity_array.shape[0] - 1
    if not np.isnan(proximity_array[i, i - 1]):
        proximity_array[i, i] = 1
    else:
        proximity_array[i, i] = np.nan

    return proximity_array



def convert_proximity_to_distance(adata, chr_cell_type_center_value_df, one_FISH_name, chr_name,chr_bin_length_dic,set_min_limit=True,min_limit=0.001,set_max_limit=False,max_limit=1.0):
    chr_cell_type_center_value_df = chr_cell_type_center_value_df.set_index('Chr_Cell_Type')
    one_cell_type = adata.obs.loc[one_FISH_name]['cell_type']
    sigma = chr_cell_type_center_value_df.loc[chr_name + one_cell_type]['Center_Value']

    bin_length = chr_bin_length_dic[chr_name]
    Example_df = generate_Example_bin_pair_df(chr_name, bin_length)
    Example_df['data'] = adata[one_FISH_name, Example_df.index].copy().X[0]
    temp_array = convert_bin_pair_df_to_array(bin_length, Example_df)
    if set_min_limit:
        temp_array[temp_array <=0] = min_limit
    if set_max_limit:
        temp_array[temp_array > 1] = max_limit
    proximity_array = temp_array + temp_array.T
    np.fill_diagonal(proximity_array, 1)


    distance_array = infer_distance_matrix(proximity_array, sigma)
    np.fill_diagonal(distance_array, 0)
    return distance_array
def convert_proximity_to_distance(adata, chr_cell_type_center_value_df, one_FISH_name, chr_name,chr_bin_length_dic,set_min_limit=True,min_limit=0.001,set_max_limit=False,max_limit=1.0):
    chr_cell_type_center_value_df = chr_cell_type_center_value_df.set_index('Chr_Cell_Type')
    one_cell_type = adata.obs.loc[one_FISH_name]['cell_type']
    sigma = chr_cell_type_center_value_df.loc[chr_name + one_cell_type]['Center_Value']

    bin_length = chr_bin_length_dic[chr_name]
    Example_df = generate_Example_bin_pair_df(chr_name, bin_length)
    Example_df['data'] = adata[one_FISH_name, Example_df.index].copy().X[0]
    temp_array = convert_bin_pair_df_to_array(bin_length, Example_df)
    if set_min_limit:
        temp_array[temp_array <=0] = min_limit
    if set_max_limit:
        temp_array[temp_array > 1] = max_limit
    proximity_array = temp_array + temp_array.T
    np.fill_diagonal(proximity_array, 1)


    distance_array = infer_distance_matrix(proximity_array, sigma)
    np.fill_diagonal(distance_array, 0)
    return distance_array
def convert_dna_zxys_to_distance(_zxy):
    from scipy.spatial.distance import pdist, squareform
    distance_array=np.array(squareform(pdist(_zxy)))
    return distance_array
def convert_distance_to_proximity(dist_matrix,sigma):
    matrix_array = np.exp2(-dist_matrix * dist_matrix / (sigma * sigma))
    return matrix_array

def plot_distance(data_array, title,sigma,cmap=plt.cm.seismic_r,fontsize=16):
#     cmap = plt.cm.seismic_r
    cmap.set_bad('#8a898a')
    sns.heatmap(data_array, cmap=cmap, center=sigma,vmax=sigma*2)
    plt.title(title, fontsize=fontsize)
    plt.show()


def plot_multi_distance_old(data_array_list, title_list, sigma, unit_fig_row, unit_fig_col, cmap=plt.cm.seismic_r,
                        fontsize=16):
    # 设置子图的数量为data_array_list的长度
    num_plots = len(data_array_list)

    # 创建足够的子图，一行中所有子图
    fig, axes = plt.subplots(1, num_plots, figsize=(unit_fig_col * num_plots, unit_fig_row))

    # 如果只有一个子图，axes不是列表，这里我们统一它的格式
    if num_plots == 1:
        axes = [axes]

    # 遍历所有的data_array和titles，绘制每个子图
    for i, (data_array, title) in enumerate(zip(data_array_list, title_list)):
        # 设置坏值的颜色
        cmap.set_bad('#8a898a')
        # 绘制热力图
        sns.heatmap(data_array, ax=axes[i], cmap=cmap, center=sigma, vmax=sigma * 2, cbar=i == num_plots - 1)
        # 设置标题
        axes[i].set_title(title, fontsize=fontsize)

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图形
    plt.show()


def plot_multi_distance(data_array_list, title_list, sigma, unit_fig_row, unit_fig_col, cmap=plt.cm.seismic_r,
                        save=False, save_path=None, show_cbar=True, show_title=True, show_ticks=True,
                        show_spines=True,wspace=0.5,dpi=600):
    import seaborn as sns
    import matplotlib.pyplot as plt
    display_font_factor = 1
    tick_label_width = 0.5
    ticklabel_size = 2
    ticklabel_width = 0.5,
    tick_label_length = 2

    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    import matplotlib.pyplot as plt
    # plt.rc('font', family='serif')
    # plt.rc('font', serif='Arial')

    SMALL_SIZE = 7 * display_font_factor
    MEDIUM_SIZE = 7.5 * display_font_factor
    BIGGER_SIZE = 7.5 * display_font_factor

    import matplotlib.pyplot as plt
    plt.rc('font', family='sans-serif')
    plt.rc('font', serif='Arial')

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('xtick.major', width=tick_label_width, size=ticklabel_size,
           pad=1)  # fontsize of the tick labels
    plt.rc('xtick.minor', width=tick_label_width, size=ticklabel_size,
           pad=1)  # fontsize of the tick labels

    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick.major', width=tick_label_width, size=ticklabel_size,
           pad=1)  # fontsize of the tick labels
    plt.rc('ytick.minor', width=tick_label_width, size=ticklabel_size,
           pad=1)  # fontsize of the tick labels

    plt.rc('legend', fontsize=SMALL_SIZE, title_fontsize=SMALL_SIZE,
           handlelength=ticklabel_size,
           handletextpad=1)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # 设置字体为Arial
    plt.rcParams['font.sans-serif'] = 'Arial'

    # 确保图形中的负号正常显示
    plt.rcParams['axes.unicode_minus'] = False
    # 设置子图的数量为data_array_list的长度
    num_plots = len(data_array_list)

    # 创建足够的子图，一行中所有子图
    fig, axes = plt.subplots(1, num_plots, figsize=(unit_fig_col * num_plots, unit_fig_row),dpi=dpi)

    # 如果只有一个子图，axes不是列表，这里我们统一它的格式
    if num_plots == 1:
        axes = [axes]

    # 遍历所有的data_array和titles，绘制每个子图
    for i, (data_array, title) in enumerate(zip(data_array_list, title_list)):
        # 设置坏值的颜色
        cmap.set_bad('#8a898a')
        # 绘制热力图，根据show_cbar参数决定是否显示颜色条
        sns.heatmap(data_array, ax=axes[i], cmap=cmap, center=sigma, vmax=sigma * 2,square=True, cbar=show_cbar and i == num_plots - 1)
        # 根据show_title参数决定是否显示标题
        if show_title:
            axes[i].set_title(title)
        # 根据show_ticks参数决定是否显示刻度
        if not show_ticks:
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        if show_spines:
            axes[i].spines['top'].set_visible(True)
            axes[i].spines['right'].set_visible(True)
            axes[i].spines['bottom'].set_visible(True)
            axes[i].spines['left'].set_visible(True)
            for spine in axes[i].spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1)

    # 调整子图之间的间距
    # plt.tight_layout()
    plt.subplots_adjust(wspace=wspace)
    # 如果save为True，则保存图形到save_path指定的路径
    if save and save_path:
        plt.savefig(save_path, transparent=True)


    # 根据save参数决定是否显示图形
    if not save:
        plt.show()
    else:
        plt.close()
def plot_multi_proximity(data_array_list, title_list, vmin,vmax, unit_fig_row, unit_fig_col, cmap=plt.cm.seismic_r,
                        save=False, save_path=None, show_cbar=True, show_title=True, show_ticks=True,
                        show_spines=True,wspace=0.5,dpi=600):
    import seaborn as sns
    import matplotlib.pyplot as plt
    display_font_factor = 1
    tick_label_width = 0.5
    ticklabel_size = 2
    ticklabel_width = 0.5,
    tick_label_length = 2

    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    import matplotlib.pyplot as plt
    # plt.rc('font', family='serif')
    # plt.rc('font', serif='Arial')

    SMALL_SIZE = 7 * display_font_factor
    MEDIUM_SIZE = 7.5 * display_font_factor
    BIGGER_SIZE = 7.5 * display_font_factor

    import matplotlib.pyplot as plt
    plt.rc('font', family='sans-serif')
    plt.rc('font', serif='Arial')

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('xtick.major', width=tick_label_width, size=ticklabel_size,
           pad=1)  # fontsize of the tick labels
    plt.rc('xtick.minor', width=tick_label_width, size=ticklabel_size,
           pad=1)  # fontsize of the tick labels

    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick.major', width=tick_label_width, size=ticklabel_size,
           pad=1)  # fontsize of the tick labels
    plt.rc('ytick.minor', width=tick_label_width, size=ticklabel_size,
           pad=1)  # fontsize of the tick labels

    plt.rc('legend', fontsize=SMALL_SIZE, title_fontsize=SMALL_SIZE,
           handlelength=ticklabel_size,
           handletextpad=1)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # 设置字体为Arial
    plt.rcParams['font.sans-serif'] = 'Arial'

    # 确保图形中的负号正常显示
    plt.rcParams['axes.unicode_minus'] = False
    # 设置子图的数量为data_array_list的长度
    num_plots = len(data_array_list)

    # 创建足够的子图，一行中所有子图
    fig, axes = plt.subplots(1, num_plots, figsize=(unit_fig_col * num_plots, unit_fig_row),dpi=dpi)

    # 如果只有一个子图，axes不是列表，这里我们统一它的格式
    if num_plots == 1:
        axes = [axes]

    # 遍历所有的data_array和titles，绘制每个子图
    for i, (data_array, title) in enumerate(zip(data_array_list, title_list)):
        # 设置坏值的颜色
        cmap.set_bad('#8a898a')
        # 绘制热力图，根据show_cbar参数决定是否显示颜色条
        sns.heatmap(data_array, ax=axes[i], cmap=cmap, vmin=vmin, vmax=vmax,square=True, cbar=show_cbar and i == num_plots - 1)
        # 根据show_title参数决定是否显示标题
        if show_title:
            axes[i].set_title(title)
        # 根据show_ticks参数决定是否显示刻度
        if not show_ticks:
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        if show_spines:
            axes[i].spines['top'].set_visible(True)
            axes[i].spines['right'].set_visible(True)
            axes[i].spines['bottom'].set_visible(True)
            axes[i].spines['left'].set_visible(True)
            for spine in axes[i].spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1)

    # 调整子图之间的间距
    # plt.tight_layout()
    plt.subplots_adjust(wspace=wspace)
    # 如果save为True，则保存图形到save_path指定的路径
    if save and save_path:
        plt.savefig(save_path, transparent=True)


    # 根据save参数决定是否显示图形
    if not save:
        plt.show()
    else:
        plt.close()

def plot_proximity(data_array, title,cmap=plt.cm.seismic,vmax=1,fontsize=16):
#     cmap = plt.cm.seismic_r
    cmap.set_bad('#8a898a')
    sns.heatmap(data_array, cmap=cmap, vmax=vmax)
    plt.title(title, fontsize=fontsize)
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns


def plot_multi_proximity_old(data_array_list, title_list, unit_fig_row, unit_fig_col, cmap=plt.cm.seismic, fontsize=16):
    # 设置子图的数量为data_array_list的长度
    num_plots = len(data_array_list)

    # 创建足够的子图，一行中所有子图
    fig, axes = plt.subplots(1, num_plots, figsize=(unit_fig_col * num_plots, unit_fig_row))

    # 如果只有一个子图，axes不是列表，这里我们统一它的格式
    if num_plots == 1:
        axes = [axes]

    # 遍历所有的data_array和titles，绘制每个子图
    for i, (data_array, title) in enumerate(zip(data_array_list, title_list)):
        # 设置坏值的颜色
        cmap.set_bad('#8a898a')
        # 绘制热力图
        sns.heatmap(data_array, ax=axes[i], cmap=cmap, vmax=1, cbar=i == num_plots - 1)
        # 设置标题
        axes[i].set_title(title, fontsize=fontsize)

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图形
    plt.show()


#def function to plot



class infer_3D_position_old(object):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.spatial.distance import pdist, squareform
    from scipy.special import factorial

    # 计算两点之间的欧氏距离
    def f_D(X_i, X_j):
        return np.sqrt(np.sum((X_i - X_j)**2))

    # 计算亲近度评分
    def f_K(D_ij, sigma):
        return np.exp2(-D_ij**2 / (2 * sigma**2))

    # 对数似然函数
    def log_likelihood(X, K_imputed, sigma,n):
        # print(X.shape)
        # print(X)
        # print(K_imputed.shape)
        # print(sigma)
        # n = X.shape[0]
        X = X.reshape((n, 3))  # 确保X是n×3的矩阵
        log_likelihood = 0
        for i in range(n):
            for j in range(i+1, n):  # 避免重复计算
                D_ij = f_D(X[i], X[j])
                K_ij = f_K(D_ij, sigma)
                # log_likelihood += K_imputed[i, j] * np.log(K_ij) - K_ij - np.log(factorial(K_imputed[i, j]))
                log_likelihood += K_imputed[i, j] * np.log(K_ij) - K_ij
        return -log_likelihood  # 由于我们通常最小化，因此取负号

    # 优化函数
    def optimize_coordinates(K_imputed, sigma, initial_guess,n):
        # print(initial_guess.shape)
        result = minimize(log_likelihood, initial_guess, args=(K_imputed, sigma,n), method='L-BFGS-B')
        if result.success:
            fitted_params = result.x
            optimized_X = fitted_params.reshape((-1, 3))
            return optimized_X
        else:
            raise ValueError(result.message)

    def get_bounds(not_nan_array,initial_guess):
        if not_nan_array is not None and initial_guess is not None:
            bounds = [(None, None)] * len(initial_guess.flatten())  # Start with all None
            bounds = np.array(bounds)
            for i in not_nan_array:
                for j in range(3):
                    bounds[i * 3 + j] = (initial_guess[i, j], initial_guess[i, j])  # Set bounds for i
        else:
            bounds = None
        return bounds

    def get_specific_dan_zxys(chrom_ids_to_index_dict, one_FISH_name, chr_name_list, chr_name, all_cell_dna_list):
        """
        Get the positions of non-NaN entries in the ZXY coordinates for a given FISH name and chromosome name.

        :param chrom_ids_to_index_dict: Dictionary mapping FISH names to indices
        :param one_FISH_name: The FISH name of interest
        :param chr_name_list: List of chromosome names
        :param chr_name: The chromosome name of interest
        :param all_cell_dna_list: List containing ZXY coordinates for all cells and chromosomes
        :return: Array of positions where the ZXY coordinates are not NaN
        """
        cell_idx = chrom_ids_to_index_dict[one_FISH_name]
        chr_idx = chr_name_list.index(chr_name)
        dan_zxys = all_cell_dna_list[cell_idx][chr_idx]

        return dan_zxys

    def get_not_nan_positions(chrom_ids_to_index_dict, one_FISH_name, chr_name_list, chr_name, all_cell_dna_list):
        """
        Get the positions of non-NaN entries in the ZXY coordinates for a given FISH name and chromosome name.

        :param chrom_ids_to_index_dict: Dictionary mapping FISH names to indices
        :param one_FISH_name: The FISH name of interest
        :param chr_name_list: List of chromosome names
        :param chr_name: The chromosome name of interest
        :param all_cell_dna_list: List containing ZXY coordinates for all cells and chromosomes
        :return: Array of positions where the ZXY coordinates are not NaN
        """
        cell_idx = chrom_ids_to_index_dict[one_FISH_name]
        chr_idx = chr_name_list.index(chr_name)
        zxys = all_cell_dna_list[cell_idx][chr_idx]
        not_nan_mask = ~np.isnan(zxys[:, 0])
        not_nan_pos_array = np.where(not_nan_mask)[0]
        return not_nan_pos_array

    def get_nan_positions(chrom_ids_to_index_dict, one_FISH_name, chr_name_list, chr_name, all_cell_dna_list):
        """
        Get the positions of non-NaN entries in the ZXY coordinates for a given FISH name and chromosome name.

        :param chrom_ids_to_index_dict: Dictionary mapping FISH names to indices
        :param one_FISH_name: The FISH name of interest
        :param chr_name_list: List of chromosome names
        :param chr_name: The chromosome name of interest
        :param all_cell_dna_list: List containing ZXY coordinates for all cells and chromosomes
        :return: Array of positions where the ZXY coordinates are not NaN
        """
        cell_idx = chrom_ids_to_index_dict[one_FISH_name]
        chr_idx = chr_name_list.index(chr_name)
        zxys = all_cell_dna_list[cell_idx][chr_idx]
        nan_mask = np.isnan(zxys[:, 0])
        nan_pos_array = np.where(nan_mask)[0]
        return nan_pos_array
class infer_3D_position_all_old(object):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.spatial.distance import pdist, squareform
    from scipy.special import factorial
    @staticmethod
    # 计算目标函数
    def poisson_obj(X, K_imputed, sigma, bias=None):
        if bias is None:
            bias = np.ones((K_imputed.shape[0], 1))

        if sparse.issparse(K_imputed):
            return infer_3D_position._poisson_obj_sparse(X, K_imputed, sigma, bias=bias)
        else:
            raise NotImplementedError("This function is implemented for sparse matrices only.")

    @staticmethod
    # 计算稀疏矩阵情况下的目标函数
    def _poisson_obj_sparse(X, K_imputed, sigma, bias=None):
        if bias is None:
            bias = np.ones((K_imputed.shape[0], 1))
        bias = bias.flatten()

        # 计算距离和核函数
        dis = np.sqrt(((X[K_imputed.row] - X[K_imputed.col]) ** 2).sum(axis=1))
        K = np.exp2(-dis ** 2 / (sigma ** 2))

        # log_2_e = np.log2(np.e)

        # 计算目标函数值
        # L_array = -((K_imputed.data * dis) / (log_2_e * (sigma ** 2)) + (K * bias[K_imputed.row] * bias[K_imputed.col]))
        obj = (K_imputed.data*np.log(K)).sum()-(K* bias[K_imputed.row] * bias[K_imputed.col]).sum()
        # obj = L_array
        #     obj -= np.sum(np.log(np.arange(1, K_imputed.data.max() + 1)))  # 计算阶乘的对数
        return -obj  # 返回负的目标函数值以进行最小化
    @staticmethod
    def eval_f(x, user_data=None):
        n, K_imputed, sigma, bias = user_data
        x = x.reshape((n, 3))
        obj = infer_3D_position.poisson_obj(x, K_imputed, sigma, bias=bias)
        x = x.flatten()
        # print("obj", obj)
        return obj

    @staticmethod
    # 优化函数
    def optimize_coordinates(K_imputed, sigma, initial_guess,maxfun=5000,maxiter=5000,bounds=None,bias=None,verbose=True):
        n = K_imputed.shape[0]
        if not sparse.isspmatrix_coo(K_imputed):
            K_imputed = sparse.coo_matrix(K_imputed)
        # print(initial_guess.shape)
        data = (n, K_imputed, sigma,bias)

        result = minimize(infer_3D_position.eval_f, initial_guess.flatten(),
                          bounds=bounds,args=(data,), method='L-BFGS-B',
                          options={'disp': verbose, 'maxfun': maxfun, 'maxiter': maxiter},)

        fitted_params = result.x
        optimized_X = fitted_params.reshape((-1, 3))
        return optimized_X

        # if result.success:
        #     fitted_params = result.x
        #     optimized_X = fitted_params.reshape((-1, 3))
        #     return optimized_X
        # else:
        #     raise ValueError(result.message)

    @staticmethod
    def get_bounds(not_nan_array,initial_guess):
        if not_nan_array is not None and initial_guess is not None:
            bounds = [(None, None)] * len(initial_guess.flatten())  # Start with all None
            bounds = np.array(bounds)
            for i in not_nan_array:
                for j in range(3):
                    bounds[i * 3 + j] = (initial_guess[i, j], initial_guess[i, j])  # Set bounds for i
        else:
            bounds = None
        return bounds

    @staticmethod
    def get_specific_dan_zxys(chrom_ids_to_index_dict, one_FISH_name, chr_name_list, chr_name, all_cell_dna_list):
        """
        Get the positions of non-NaN entries in the ZXY coordinates for a given FISH name and chromosome name.

        :param chrom_ids_to_index_dict: Dictionary mapping FISH names to indices
        :param one_FISH_name: The FISH name of interest
        :param chr_name_list: List of chromosome names
        :param chr_name: The chromosome name of interest
        :param all_cell_dna_list: List containing ZXY coordinates for all cells and chromosomes
        :return: Array of positions where the ZXY coordinates are not NaN
        """
        cell_idx = chrom_ids_to_index_dict[one_FISH_name]
        chr_idx = chr_name_list.index(chr_name)
        dan_zxys = all_cell_dna_list[cell_idx][chr_idx]

        return dan_zxys

    @staticmethod
    def get_not_nan_positions(chrom_ids_to_index_dict, one_FISH_name, chr_name_list, chr_name, all_cell_dna_list):
        """
        Get the positions of non-NaN entries in the ZXY coordinates for a given FISH name and chromosome name.

        :param chrom_ids_to_index_dict: Dictionary mapping FISH names to indices
        :param one_FISH_name: The FISH name of interest
        :param chr_name_list: List of chromosome names
        :param chr_name: The chromosome name of interest
        :param all_cell_dna_list: List containing ZXY coordinates for all cells and chromosomes
        :return: Array of positions where the ZXY coordinates are not NaN
        """
        cell_idx = chrom_ids_to_index_dict[one_FISH_name]
        chr_idx = chr_name_list.index(chr_name)
        zxys = all_cell_dna_list[cell_idx][chr_idx]
        not_nan_mask = ~np.isnan(zxys[:, 0])
        not_nan_pos_array = np.where(not_nan_mask)[0]
        return not_nan_pos_array

    @staticmethod
    def get_nan_positions(chrom_ids_to_index_dict, one_FISH_name, chr_name_list, chr_name, all_cell_dna_list):
        """
        Get the positions of non-NaN entries in the ZXY coordinates for a given FISH name and chromosome name.

        :param chrom_ids_to_index_dict: Dictionary mapping FISH names to indices
        :param one_FISH_name: The FISH name of interest
        :param chr_name_list: List of chromosome names
        :param chr_name: The chromosome name of interest
        :param all_cell_dna_list: List containing ZXY coordinates for all cells and chromosomes
        :return: Array of positions where the ZXY coordinates are not NaN
        """
        cell_idx = chrom_ids_to_index_dict[one_FISH_name]
        chr_idx = chr_name_list.index(chr_name)
        zxys = all_cell_dna_list[cell_idx][chr_idx]
        nan_mask = np.isnan(zxys[:, 0])
        nan_pos_array = np.where(nan_mask)[0]
        return nan_pos_array



class infer_3D_position_all(object):
    from scipy import sparse
    import numpy as np
    from scipy.optimize import minimize
    from scipy.spatial.distance import pdist, squareform
    from scipy.special import factorial

    @classmethod
    def poisson_obj(cls, X, K_imputed, sigma, bias=None):
        if bias is None:
            bias = np.ones((K_imputed.shape[0], 1))

        if sparse.issparse(K_imputed):
            return cls._poisson_obj_sparse(X, K_imputed, sigma, bias=bias)
        else:
            raise NotImplementedError("This function is implemented for sparse matrices only.")

    @classmethod
    def _poisson_obj_sparse(cls, X, K_imputed, sigma, bias=None):
        if bias is None:
            bias = np.ones((K_imputed.shape[0], 1))
        bias = bias.flatten()

        dis = np.sqrt(((X[K_imputed.row] - X[K_imputed.col]) ** 2).sum(axis=1))
        K = np.exp2(-dis ** 2 / (sigma ** 2))

        obj = (K_imputed.data * np.log(K)).sum() - (K * bias[K_imputed.row] * bias[K_imputed.col]).sum()
        return -obj

    @classmethod
    def eval_f(cls, x, user_data=None):
        n, K_imputed, sigma, bias = user_data
        x = x.reshape((n, 3))
        obj = cls.poisson_obj(x, K_imputed, sigma, bias=bias)
        x = x.flatten()
        return obj

    @classmethod
    def optimize_coordinates(cls, K_imputed, sigma, initial_guess, maxfun=5000, maxiter=5000, bounds=None, bias=None, verbose=True):
        n = K_imputed.shape[0]
        if not sparse.isspmatrix_coo(K_imputed):
            K_imputed[np.isnan(K_imputed)] = 0
            K_imputed = sparse.coo_matrix(K_imputed)
        else:
            K_imputed = K_imputed.tocsr()
            K_imputed.eliminate_zeros()
            K_imputed = K_imputed.tocoo()
        K_imputed.setdiag(0)
        K_imputed.eliminate_zeros()

        data = (n, K_imputed, sigma, bias)

        result = minimize(cls.eval_f, initial_guess.flatten(),
                          bounds=bounds, args=(data,), method='L-BFGS-B',
                          options={'disp': verbose, 'maxfun': maxfun, 'maxiter': maxiter})

        fitted_params = result.x
        optimized_X = fitted_params.reshape((-1, 3))
        return optimized_X

    @classmethod
    def get_bounds(cls, not_nan_array, initial_guess):
        if not_nan_array is not None and initial_guess is not None:
            bounds = [(None, None)] * len(initial_guess.flatten())
            bounds = np.array(bounds)
            for i in not_nan_array:
                for j in range(3):
                    bounds[i * 3 + j] = (initial_guess[i, j], initial_guess[i, j])
        else:
            bounds = None
        return bounds

    @classmethod
    def get_specific_dan_zxys(cls, chrom_ids_to_index_dict, one_FISH_name, chr_name_list, chr_name, all_cell_dna_list):
        cell_idx = chrom_ids_to_index_dict[one_FISH_name]
        chr_idx = chr_name_list.index(chr_name)
        dan_zxys = all_cell_dna_list[cell_idx][chr_idx]
        return dan_zxys

    @classmethod
    def get_not_nan_positions(cls, chrom_ids_to_index_dict, one_FISH_name, chr_name_list, chr_name, all_cell_dna_list):
        cell_idx = chrom_ids_to_index_dict[one_FISH_name]
        chr_idx = chr_name_list.index(chr_name)
        zxys = all_cell_dna_list[cell_idx][chr_idx]
        not_nan_mask = ~np.isnan(zxys[:, 0])
        not_nan_pos_array = np.where(not_nan_mask)[0]
        return not_nan_pos_array

    @classmethod
    def get_nan_positions(cls, chrom_ids_to_index_dict, one_FISH_name, chr_name_list, chr_name, all_cell_dna_list):
        cell_idx = chrom_ids_to_index_dict[one_FISH_name]
        chr_idx = chr_name_list.index(chr_name)
        zxys = all_cell_dna_list[cell_idx][chr_idx]
        nan_mask = np.isnan(zxys[:, 0])
        nan_pos_array = np.where(nan_mask)[0]
        return nan_pos_array



class infer_3D_position(object):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.spatial.distance import pdist, squareform
    from scipy.special import factorial
    @staticmethod
    # 计算目标函数
    def poisson_obj(X, K_imputed, not_imputed_all_x,nan_position_array,sigma, bias=None):
        if bias is None:
            bias = np.ones((K_imputed.shape[0], 1))

        if sparse.issparse(K_imputed):
            return infer_3D_position._poisson_obj_sparse(X, K_imputed,not_imputed_all_x,nan_position_array, sigma, bias=bias)
        else:
            raise NotImplementedError("This function is implemented for sparse matrices only.")

    @staticmethod
    # 计算稀疏矩阵情况下的目标函数
    def _poisson_obj_sparse(X, K_imputed, not_imputed_all_x,nan_position_array,sigma, bias=None):
        if bias is None:
            bias = np.ones((K_imputed.shape[0], 1))
        bias = bias.flatten()
        All_X=not_imputed_all_x.copy()
        All_X[nan_position_array]=X
        #All_X 包含固定的不需要优化的位置nan_position和需要优化的X
        # 计算距离和核函数
        dis = np.sqrt(((All_X[K_imputed.row] - All_X[K_imputed.col]) ** 2).sum(axis=1))
        K = np.exp2(-dis ** 2 / (sigma ** 2))

        # log_2_e = np.log2(np.e)

        # 计算目标函数值
        # L_array = -((K_imputed.data * dis) / (log_2_e * (sigma ** 2)) + (K * bias[K_imputed.row] * bias[K_imputed.col]))
        obj = (K_imputed.data*np.log(K)).sum()-(K* bias[K_imputed.row] * bias[K_imputed.col]).sum()
        # obj = L_array
        #     obj -= np.sum(np.log(np.arange(1, K_imputed.data.max() + 1)))  # 计算阶乘的对数
        return -obj  # 返回负的目标函数值以进行最小化
    @staticmethod
    def eval_f(x, user_data=None):
        n, need_imputed_n,K_imputed, not_imputed_all_x,nan_position_array,sigma,bias = user_data
        x = x.reshape((need_imputed_n, 3))
        obj = infer_3D_position.poisson_obj(x, K_imputed, not_imputed_all_x,nan_position_array,sigma, bias=bias)
        x = x.flatten()
        # print("obj", obj)
        return obj

    @staticmethod
    # 优化函数
    def optimize_coordinates(K_imputed, sigma, need_imputed_x_initial_guess,not_imputed_all_x,nan_position_array,maxfun=5000,maxiter=5000,bounds=None,bias=None,verbose=True):
        n = K_imputed.shape[0]
        need_imputed_n=nan_position_array.shape[0]
        if not sparse.isspmatrix_coo(K_imputed):
            K_imputed[np.isnan(K_imputed)] = 0
            K_imputed = sparse.coo_matrix(K_imputed)
        else:
            K_imputed = K_imputed.tocsr()
            K_imputed.eliminate_zeros()
            K_imputed = K_imputed.tocoo()
        K_imputed.setdiag(0)
        K_imputed.eliminate_zeros()
        # print(initial_guess.shape)
        data = (n, need_imputed_n,K_imputed, not_imputed_all_x,nan_position_array,sigma,bias)

        result = minimize(infer_3D_position.eval_f, need_imputed_x_initial_guess.flatten(),
                          bounds=bounds,args=(data,), method='L-BFGS-B',
                          options={'disp': verbose, 'maxfun': maxfun, 'maxiter': maxiter},)

        fitted_params = result.x
        optimized_X = fitted_params.reshape((-1, 3))
        All_X = not_imputed_all_x.copy()
        All_X[nan_position_array] = optimized_X
        return All_X

        # if result.success:
        #     fitted_params = result.x
        #     optimized_X = fitted_params.reshape((-1, 3))
        #     return optimized_X
        # else:
        #     raise ValueError(result.message)

    @staticmethod
    def get_bounds(not_nan_array,initial_guess):
        if not_nan_array is not None and initial_guess is not None:
            bounds = [(None, None)] * len(initial_guess.flatten())  # Start with all None
            bounds = np.array(bounds)
            for i in not_nan_array:
                for j in range(3):
                    bounds[i * 3 + j] = (initial_guess[i, j], initial_guess[i, j])  # Set bounds for i
        else:
            bounds = None
        return bounds

    @staticmethod
    def get_specific_dan_zxys(chrom_ids_to_index_dict, one_FISH_name, chr_name_list, chr_name, all_cell_dna_list):
        """
        Get the positions of non-NaN entries in the ZXY coordinates for a given FISH name and chromosome name.

        :param chrom_ids_to_index_dict: Dictionary mapping FISH names to indices
        :param one_FISH_name: The FISH name of interest
        :param chr_name_list: List of chromosome names
        :param chr_name: The chromosome name of interest
        :param all_cell_dna_list: List containing ZXY coordinates for all cells and chromosomes
        :return: Array of positions where the ZXY coordinates are not NaN
        """
        cell_idx = chrom_ids_to_index_dict[one_FISH_name]
        chr_idx = chr_name_list.index(chr_name)
        dan_zxys = all_cell_dna_list[cell_idx][chr_idx]

        return dan_zxys

    @staticmethod
    def get_not_nan_positions(chrom_ids_to_index_dict, one_FISH_name, chr_name_list, chr_name, all_cell_dna_list):
        """
        Get the positions of non-NaN entries in the ZXY coordinates for a given FISH name and chromosome name.

        :param chrom_ids_to_index_dict: Dictionary mapping FISH names to indices
        :param one_FISH_name: The FISH name of interest
        :param chr_name_list: List of chromosome names
        :param chr_name: The chromosome name of interest
        :param all_cell_dna_list: List containing ZXY coordinates for all cells and chromosomes
        :return: Array of positions where the ZXY coordinates are not NaN
        """
        cell_idx = chrom_ids_to_index_dict[one_FISH_name]
        chr_idx = chr_name_list.index(chr_name)
        zxys = all_cell_dna_list[cell_idx][chr_idx]
        not_nan_mask = ~np.isnan(zxys[:, 0])
        not_nan_pos_array = np.where(not_nan_mask)[0]
        return not_nan_pos_array

    @staticmethod
    def get_nan_positions(chrom_ids_to_index_dict, one_FISH_name, chr_name_list, chr_name, all_cell_dna_list):
        """
        Get the positions of non-NaN entries in the ZXY coordinates for a given FISH name and chromosome name.

        :param chrom_ids_to_index_dict: Dictionary mapping FISH names to indices
        :param one_FISH_name: The FISH name of interest
        :param chr_name_list: List of chromosome names
        :param chr_name: The chromosome name of interest
        :param all_cell_dna_list: List containing ZXY coordinates for all cells and chromosomes
        :return: Array of positions where the ZXY coordinates are not NaN
        """
        cell_idx = chrom_ids_to_index_dict[one_FISH_name]
        chr_idx = chr_name_list.index(chr_name)
        zxys = all_cell_dna_list[cell_idx][chr_idx]
        nan_mask = np.isnan(zxys[:, 0])
        nan_pos_array = np.where(nan_mask)[0]
        return nan_pos_array
def get_diffgene_with_leiden(temp_adata,n_loci_pair=200):
    temp_adata.X=np.nan_to_num(temp_adata.X)
    sc.pp.normalize_total(temp_adata, target_sum=1e4)
    sc.pp.log1p(temp_adata)

    sc.pp.pca(temp_adata,n_comps=30)
    sc.pp.neighbors(temp_adata,n_neighbors=30,n_pcs=30,use_rep='X_pca')
    sc.tl.umap(temp_adata,min_dist=0.5,)
    sc.tl.leiden(temp_adata,resolution=1)
    leiden_type_num=len(np.unique(temp_adata.obs['leiden']))
    n_genes=n_loci_pair//leiden_type_num
    # sc.pl.umap(temp_adata, color=['leiden','cell_type'])

    groupby_name='leiden'
    # 差异表达分析
    sc.tl.rank_genes_groups(temp_adata, groupby=groupby_name, method='t-test', n_genes=n_genes)

    # 获取差异基因的名字
    # 创建一个空的列表来保存差异基因
    diff_genes_per_group = {}

    # 对于每个细胞类型，获取排名前100的差异基因
    for group in temp_adata.uns['rank_genes_groups']['names'].dtype.names:
        diff_genes_per_group[group] = temp_adata.uns['rank_genes_groups']['names'][group][:n_genes]

    # 如果想要一个包含所有细胞类型的前100个差异基因的统一列表，我们可以将它们合并
    # 使用集合来自动去除重复的基因
    diff_genes = list({gene for genes in diff_genes_per_group.values() for gene in genes})
    non_diff_genes_mask=~temp_adata.var.index.isin(diff_genes)
    not_diff_genes = list(temp_adata.var_names[~temp_adata.var.index.isin(diff_genes)])
    return diff_genes,not_diff_genes,non_diff_genes_mask


import scanpy as sc


def get_highly_variable_genes(temp_adata, n_loci_pair=20):
    # 数据规范化
    temp_adata.X = np.nan_to_num(temp_adata.X)
    sc.pp.normalize_total(temp_adata, target_sum=1e4)
    # 对数变换
    sc.pp.log1p(temp_adata)
    # 找到高可变基因
    sc.pp.highly_variable_genes(temp_adata, n_top_genes=n_loci_pair)

    # 从adata中提取高可变基因的信息
    highly_variable_genes = temp_adata.var[temp_adata.var['highly_variable']].index.tolist()

    # 如果想要找到非高可变基因，可以取反这个条件
    non_highly_variable_genes = temp_adata.var[~temp_adata.var['highly_variable']].index.tolist()
    non_highly_variable_genes_mask = ~temp_adata.var['highly_variable']

    return highly_variable_genes, non_highly_variable_genes, non_highly_variable_genes_mask




def fill_nan_with_interpolation(distmap_list_median):
    from scipy.interpolate import griddata
    # 获取nan和非nan的位置和值
    nan_locs = np.isnan(distmap_list_median)
    not_nan_locs = ~nan_locs
    not_nan_values = distmap_list_median[not_nan_locs]
    not_nan_positions = np.array(np.nonzero(not_nan_locs)).T

    # 获取nan的位置
    nan_positions = np.array(np.nonzero(nan_locs)).T

    # 使用插值填充nan值
    interpolated_values = griddata(not_nan_positions, not_nan_values, nan_positions, method='nearest')

    # 将插值放入原始数组中的nan位置
    filled_distmap_list_median = distmap_list_median.copy()
    filled_distmap_list_median[nan_locs] = interpolated_values

    return filled_distmap_list_median
def get_distmap_list_from_dna_zxys(dna_zxys):
    distmap_list = np.array([squareform(pdist(_zxy)) for _zxy in tqdm(dna_zxys)])
    return distmap_list
class generate_spatial_distance_h5ad(object):
    @staticmethod
    def generate_spatial_bin_pair_df_one_chr(chr_name_list,cell_number,all_cell_dna_list,
                                             chr_name,chr_bin_length_dic,chrom_index_to_ids_dict):

        # store_dir = r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/ImputHiFI/first_seen_probe_data"
        # used_chr_name_list = data['chrom_name'][0:19]
        chr_index = chr_name_list.index(chr_name)
        # cell_number = len(data['chrom_ids'])
        # all_cell_dna_list = data['all_cell_dna_list']
        print(chr_name)
        dna_zxys=[]
        distmap_list_dic = {}
        for cell_index in tqdm(range(cell_number)):
            dna_zxys.append(all_cell_dna_list[cell_index][chr_index])
        distmap_list = get_distmap_list_from_dna_zxys(dna_zxys)
        distmap_list_dic[chr_name]=distmap_list

        all_cell_df_dic = {}
        for chr_name in distmap_list_dic.keys():
            print(chr_name)
            # chr_name='chr1'
            bin_length = chr_bin_length_dic[chr_name]
            Example_bin_pair_df = generate_Example_bin_pair_df(chr_name, bin_length)
            all_cell_one_chr_df = pd.DataFrame(index=Example_bin_pair_df.index)
            for cell_index in tqdm(range(cell_number)):
                one_cell_chr_array = distmap_list_dic[chr_name][cell_index].copy()
                one_cell_chr_df = convert_array_to_bin_pair_df(chr_name, bin_length, one_cell_chr_array)
                cell_name = chrom_index_to_ids_dict[cell_index]
                all_cell_one_chr_df.loc[one_cell_chr_df.index, cell_name] = one_cell_chr_df['data'].values
            all_cell_df_dic[chr_name] = all_cell_one_chr_df

        return chr_name,all_cell_df_dic
    @staticmethod
    def generate_spatial_h5ad(chr_name_list,cell_number,all_cell_dna_list,
                              chr_bin_length_dic,chrom_index_to_ids_dict,adata_obs_df,store_path,kernerl_number=None):
        if kernerl_number is None:
            kernerl_number=len(chr_name_list)

        chr_name_split_list = np.array_split(chr_name_list, kernerl_number)
        # store_dir = r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/ImputHiFI/first_seen_probe_data"
        import multiprocessing as mp
        num_threads = kernerl_number


        _domain_args = [(chr_name_list, cell_number, all_cell_dna_list, chr_name_part[0],
                         chr_bin_length_dic, chrom_index_to_ids_dict)
                        for chr_name_part in chr_name_split_list]
        with mp.Pool(num_threads) as domain_pool:
            domain_results = domain_pool.starmap(generate_spatial_distance_h5ad.generate_spatial_bin_pair_df_one_chr, _domain_args)
            domain_pool.close()
            domain_pool.join()
            domain_pool.terminate()
        all_cell_df_dic = {}
        for result in domain_results:
            chr_name, temp_all_cell_df_dic=result
            all_cell_df_dic[chr_name] = temp_all_cell_df_dic[chr_name]
        all_cell_df = pd.concat(all_cell_df_dic.values())

        adata = sc.AnnData(all_cell_df.T)

        adata.obs = adata_obs_df.loc[adata.obs_names]
        adata.write(store_path)
    @staticmethod
    def use_spatial_h5ad(first_seen_probe_data):

        chr_name_list=first_seen_probe_data['chr_name_list']
        cell_number=first_seen_probe_data['cell_number']
        all_cell_dna_list=first_seen_probe_data['all_cell_dna_list']
        chr_bin_length_dic=first_seen_probe_data['chr_bin_length_dic']
        chrom_index_to_ids_dict=first_seen_probe_data['chrom_index_to_ids_dict']
        adata_obs_df=first_seen_probe_data['adata_obs_df']


class generate_proximity_score_h5ad(object):
    @staticmethod
    def generate_proximity_bin_pair_df_one_chr(DNA_FISH_spad_adata,chr_cell_type_center_value_df,
                                             chr_name, chr_bin_length_dic,):

        all_cell_df_dic = {}

        print(chr_name)
        # chr_name='chr1'
        bin_length = chr_bin_length_dic[chr_name]
        Example_bin_pair_df = generate_Example_bin_pair_df(chr_name, bin_length)
        all_cell_one_chr_df = pd.DataFrame(index=Example_bin_pair_df.index)
        for cell_index, one_cell_name in tqdm(enumerate(DNA_FISH_spad_adata.obs_names)):
            one_cell_type = DNA_FISH_spad_adata.obs.loc[one_cell_name]['cell_type']
            center_value = chr_cell_type_center_value_df.loc[chr_name + one_cell_type]['Center_Value']
            Example_df =generate_Example_bin_pair_df(chr_name, bin_length)
            temp_array = DNA_FISH_spad_adata[one_cell_name, Example_df.index].copy().X[0]
            Example_df['data'] = temp_array
            matrix_array = convert_bin_pair_df_to_array(bin_length, Example_df)
            matrix_array = matrix_array + matrix_array.T
            np.fill_diagonal(matrix_array, 0)
            dist_matrix = copy.deepcopy(matrix_array)
            sigma = center_value
            matrix_array = np.exp2(-dist_matrix * dist_matrix / (sigma * sigma))
            proxi_array = copy.deepcopy(matrix_array)

            one_cell_chr_array = proxi_array
            one_cell_chr_df = convert_array_to_bin_pair_df(chr_name, bin_length, one_cell_chr_array)
            cell_name = one_cell_name
            all_cell_one_chr_df.loc[one_cell_chr_df.index, cell_name] = one_cell_chr_df['data'].values
        all_cell_df_dic[chr_name] = all_cell_one_chr_df

        return chr_name, all_cell_df_dic

    @staticmethod
    def generate_proximity_h5ad(DNA_FISH_spad_adata,chr_name_list,chr_cell_type_center_value_df,
                              chr_bin_length_dic, adata_obs_df, store_path,
                              kernerl_number=None):
        if kernerl_number is None:
            kernerl_number = len(chr_name_list)

        chr_name_split_list = np.array_split(chr_name_list, kernerl_number)
        # store_dir = r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/ImputHiFI/first_seen_probe_data"
        import multiprocessing as mp
        num_threads = kernerl_number


        _domain_args = [(DNA_FISH_spad_adata,  chr_cell_type_center_value_df,chr_name_part[0], chr_bin_length_dic)
                        for chr_name_part in chr_name_split_list]
        with mp.Pool(num_threads) as domain_pool:
            domain_results = domain_pool.starmap(generate_proximity_score_h5ad.generate_proximity_bin_pair_df_one_chr,
                                                 _domain_args)
            domain_pool.close()
            domain_pool.join()
            domain_pool.terminate()
        all_cell_df_dic = {}
        for result in domain_results:
            chr_name, temp_all_cell_df_dic = result
            all_cell_df_dic[chr_name] = temp_all_cell_df_dic[chr_name]
        all_cell_df = pd.concat(all_cell_df_dic.values())

        adata = sc.AnnData(all_cell_df.T)

        adata.obs = adata_obs_df.loc[adata.obs_names]
        adata.write(store_path)

    @staticmethod
    def use_spatial_h5ad(first_seen_probe_data):

        chr_name_list = first_seen_probe_data['chr_name_list']
        cell_number = first_seen_probe_data['cell_number']
        all_cell_dna_list = first_seen_probe_data['all_cell_dna_list']
        chr_bin_length_dic = first_seen_probe_data['chr_bin_length_dic']
        chrom_index_to_ids_dict = first_seen_probe_data['chrom_index_to_ids_dict']
        adata_obs_df = first_seen_probe_data['adata_obs_df']
def Get_RMSD_rotated_A(A, B):
    '''
    Get RMSD and rotated A. That means B is the target and the A is rotated to B.
    Parameters
    ----------
    A
        dna zxys list
    B
        dna zxys list
    Returns
    -------
    dist
        RMSD dist
    rotated_A
        rotated A dna zxys list
    '''
    import scipy
    R, scale = scipy.linalg.orthogonal_procrustes(A, B)
    N=len(A)
    dist = np.linalg.norm(np.dot(A, R) - B) / np.sqrt(N)
    rotated_A = copy.deepcopy(A) @ R

    return dist,rotated_A
class generate_cell_cell_RMSD(object):
    @staticmethod
    def generate_cell_cell_RMSD_one_chr(chr_name_list,cell_number,all_cell_dna_list,
                                             chr_name, cell_name_array):
        print(chr_name)

        chr_index = chr_name_list.index(chr_name)

        print(chr_name)
        dna_zxys = []

        for cell_index in tqdm(range(cell_number)):
            dna_zxys.append(all_cell_dna_list[cell_index][chr_index])
        RMSD_array = np.zeros((len(dna_zxys), len(dna_zxys)))
        max_RMSD = 1000000

        nearest_neighbors_indices_dic = {}
        nearest_neighbors_RMSD_dic = {}
        nearest_neighbors_names_dic = {}
        for i in tqdm(range(len(dna_zxys))):
            for j in range(i + 1, len(dna_zxys)):
                zxy_A = dna_zxys[i]
                zxy_B = dna_zxys[j]
                valid_indices = ~np.isnan(zxy_A).any(axis=1) & ~np.isnan(zxy_B).any(axis=1)
                valid_points_A = zxy_A[valid_indices]
                valid_points_B = zxy_B[valid_indices]
                mean_A = np.mean(valid_points_A, axis=0)
                mean_B = np.mean(valid_points_B, axis=0)
                valid_points_A = valid_points_A - mean_A
                valid_points_B = valid_points_B - mean_B
                RMSD, rotated_A = Get_RMSD_rotated_A(valid_points_A, valid_points_B)
                if np.isnan(RMSD):
                    RMSD = max_RMSD
                RMSD_array[i, j] = RMSD
                RMSD_array[j, i] = RMSD

            RMSD_one_cell = copy.deepcopy(RMSD_array[i])
            sorted_indices = np.argsort(RMSD_one_cell)
            sorted_name_array = cell_name_array[sorted_indices[1:]]
            one_FISH_name = cell_name_array[i]
            nearest_neighbors_indices_dic[one_FISH_name] = sorted_indices[1:]
            nearest_neighbors_RMSD_dic[one_FISH_name] = RMSD_one_cell[sorted_indices[1:]]
            nearest_neighbors_names_dic[one_FISH_name] = sorted_name_array
        all_result = {}
        all_result['nearest_neighbors_indices_dic'] = nearest_neighbors_indices_dic
        all_result['nearest_neighbors_RMSD_dic'] = nearest_neighbors_RMSD_dic
        all_result['nearest_neighbors_names_dic'] = nearest_neighbors_names_dic
        chr_all_result = {}
        chr_all_result[chr_name] = all_result
        return chr_name, chr_all_result
    @staticmethod
    def get_All_cell_RMSD_dict(chr_name_list,cell_number,all_cell_dna_list,
                               cell_name_array,kernerl_number=None):
        if kernerl_number is None:
            kernerl_number = len(chr_name_list)

        chr_name_split_list = np.array_split(chr_name_list, kernerl_number)
        # store_dir = r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/ImputHiFI/first_seen_probe_data"
        import multiprocessing as mp
        num_threads = kernerl_number

        _domain_args = [(chr_name_list, cell_number, all_cell_dna_list, chr_name_part[0],
                          cell_name_array)
                        for chr_name_part in chr_name_split_list]
        with mp.Pool(num_threads) as domain_pool:
            domain_results = domain_pool.starmap(generate_cell_cell_RMSD.generate_cell_cell_RMSD_one_chr, _domain_args)
            domain_pool.close()
            domain_pool.join()
            domain_pool.terminate()

        result_dic = {}
        result_dic['nearest_neighbors_indices_dic'] = {}
        result_dic['nearest_neighbors_RMSD_dic'] = {}
        result_dic['nearest_neighbors_names_dic'] = {}

        for result in domain_results:
            print("Collect result")
            chr_name, one_chr_corr_dict = result
            corr_dict = one_chr_corr_dict[chr_name]
            for key, item in corr_dict.items():
                result_dic[key][chr_name] = item
        All_cell_RMSD_dict=result_dic
        return All_cell_RMSD_dict

class generate_RNA_FISH_cell_cell_RMSD(object):
    @staticmethod
    def generate_RNA_FISH_cell_cell_RMSD_one_chr(chr_name, cell_name_array, one_cell_name_same_leiden_type_dict,
                                     FISH_nearest_neighbors_names_dic):
        print(chr_name)

        FISH_RMSD_nearest_neighbors_names_dic = {}
        FISH_RMSD_nearest_neighbors_names_dic[chr_name] = {}
        for one_cell_name in tqdm(cell_name_array):
            cells_with_same_leiden_type = one_cell_name_same_leiden_type_dict[one_cell_name]
            candidate_array = FISH_nearest_neighbors_names_dic[chr_name][one_cell_name]
            used_array = candidate_array[np.isin(candidate_array, cells_with_same_leiden_type)]
            FISH_RMSD_nearest_neighbors_names_dic[chr_name][one_cell_name] = used_array

        all_result = {}
        all_result['nearest_neighbors_names_dic'] = FISH_RMSD_nearest_neighbors_names_dic
        chr_all_result = {}
        chr_all_result[chr_name] = all_result
        return chr_name, chr_all_result
    @staticmethod
    def get_RNA_FISH_leiden_frome_adata(adata,n_comps=30,n_pcs=30):
        # temp_path = os.path.join(data_dir, 'RNA', 'Cai_RNA_adata.h5ad')
        # Cai_adata = sc.read(temp_path)
        # adata = Cai_adata.copy()

        # Normalization, logarithmization, and scaling
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.scale(adata)

        # Principal component analysis
        sc.pp.pca(adata, n_comps=n_comps)

        # Computing the neighborhood graph
        sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=100, use_rep='X_pca')

        # Run UMAP dimensionality reduction
        sc.tl.umap(adata)

        # Clustering the data using the Leiden algorithm
        sc.tl.leiden(adata, resolution=1)
        return adata
    @staticmethod
    def get_RNA_FISH_cell_cell_RMSD_dict(RNA_FISH_adata,chr_name_list,cell_name_array,
                                         All_cell_RMSD_dict,n_comps=30,n_pcs=30,kernerl_number=None):

        FISH_nearest_neighbors_names_dic = All_cell_RMSD_dict['nearest_neighbors_names_dic']
        adata = generate_RNA_FISH_cell_cell_RMSD.get_RNA_FISH_leiden_frome_adata(RNA_FISH_adata.copy(),n_comps=n_comps,n_pcs=n_pcs)
        one_cell_name_same_leiden_type_dict = {}
        for one_cell_name in cell_name_array:
            leiden_type_of_one_cell = adata.obs.loc[adata.obs['cell_name'] == one_cell_name, 'leiden'].values[0]
            # 找到所有具有相同 leiden 类型的细胞
            cells_with_same_leiden_type = adata.obs[adata.obs['leiden'] == leiden_type_of_one_cell]['cell_name'].values
            one_cell_name_same_leiden_type_dict[one_cell_name] = cells_with_same_leiden_type

        if kernerl_number is None:
            kernerl_number = len(chr_name_list)

        chr_name_split_list = np.array_split(chr_name_list, kernerl_number)
        # store_dir = r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/ImputHiFI/first_seen_probe_data"
        import multiprocessing as mp
        num_threads = kernerl_number

        _domain_args = [(chr_name_part[0], cell_name_array, one_cell_name_same_leiden_type_dict,
                          FISH_nearest_neighbors_names_dic)
                        for chr_name_part in chr_name_split_list]
        with mp.Pool(num_threads) as domain_pool:
            domain_results = domain_pool.starmap(generate_RNA_FISH_cell_cell_RMSD.generate_RNA_FISH_cell_cell_RMSD_one_chr, _domain_args)
            domain_pool.close()
            domain_pool.join()
            domain_pool.terminate()

        result_dic = {}

        result_dic['nearest_neighbors_names_dic'] = {}

        for result in domain_results:
            print("Collect result")
            chr_name, one_chr_corr_dict = result
            corr_dict = one_chr_corr_dict[chr_name]
            for key, item in corr_dict.items():
                result_dic[key][chr_name] = item[chr_name]
        All_cell_RNA_RMSD_dict = result_dic
        return All_cell_RNA_RMSD_dict

class generate_impute_dna_list(object):
    @staticmethod
    def get_impute_dna_list_one_chr_result(chr_bin_length_dic, chr_name, chr_name_list, all_FISH_name,
                              ImputeHiFI_adata, ImputeHiFI_diff_genes_adata, all_cell_dna_list,
                              Linear_impute_all_cell_dna_list,
                              chr_cell_type_center_value_df, chrom_ids_to_index_dict, loci_pair_type,
                              ):
        print(chr_name)
        if loci_pair_type != 'no_diff':
            print("Fine tune with leiden find import loci pair")
        optimize_all_cell_dna_list = copy.deepcopy(Linear_impute_all_cell_dna_list)
        for one_FISH_name in tqdm(all_FISH_name):
            one_cell_type = ImputeHiFI_adata.obs.loc[one_FISH_name]['cell_type']
            sigma = chr_cell_type_center_value_df.loc[chr_name + one_cell_type]['Center_Value']

            proximity_array = get_proximity_array(ImputeHiFI_adata, one_FISH_name, chr_name,
                                                             chr_bin_length_dic,
                                                             set_min_limit=False, min_limit=0.001, set_max_limit=True,
                                                             max_limit=1.0)
            linear_zxys = infer_3D_position.get_specific_dan_zxys(chrom_ids_to_index_dict, one_FISH_name,
                                                                             chr_name_list, chr_name,
                                                                             Linear_impute_all_cell_dna_list)
            not_imputed_zxys = infer_3D_position.get_specific_dan_zxys(chrom_ids_to_index_dict,
                                                                                  one_FISH_name, chr_name_list,
                                                                                  chr_name,
                                                                                  all_cell_dna_list)

            K_imputed = proximity_array.copy()
            initial_guess = linear_zxys.copy()

            nan_position_array = infer_3D_position.get_nan_positions(chrom_ids_to_index_dict,
                                                                                one_FISH_name, chr_name_list,
                                                                                chr_name, all_cell_dna_list)

            not_imputed_all_x = not_imputed_zxys.copy()
            need_imputed_x_initial_guess = initial_guess[nan_position_array].copy()
            # bounds = ImputeHiFI.infer_3D_position.get_bounds(not_nan_array, initial_guess)
            bounds = None
            # optimze_zxys = ImputeHiFI.infer_3D_position_all.optimize_coordinates(K_imputed, sigma, initial_guess,
            #                                                                      maxfun=5000, maxiter=5000, bounds=bounds,
            #                                                                      bias=None, verbose=False)
            if len(nan_position_array) > 0:
                optimze_zxys = infer_3D_position.optimize_coordinates(K_imputed, sigma,
                                                                                 need_imputed_x_initial_guess,
                                                                                 not_imputed_all_x, nan_position_array,
                                                                                 maxfun=5000,
                                                                                 maxiter=5000, bounds=bounds, bias=None,
                                                                                 verbose=False)
            else:
                optimze_zxys = initial_guess.copy()
            if loci_pair_type != 'no_diff':
                # print("Fine tune with leiden find import loci pair")
                # fine tune with leiden find import loci pair
                proximity_array = get_proximity_array(ImputeHiFI_diff_genes_adata, one_FISH_name, chr_name,
                                                                 chr_bin_length_dic,
                                                                 set_min_limit=False, min_limit=0.001,
                                                                 set_max_limit=True,
                                                                 max_limit=1.0)
                K_imputed = proximity_array.copy()
                initial_guess = optimze_zxys.copy()
                not_imputed_all_x = not_imputed_zxys.copy()
                need_imputed_x_initial_guess = initial_guess[nan_position_array].copy()
                if len(nan_position_array) > 0:
                    optimze_zxys = infer_3D_position.optimize_coordinates(K_imputed, sigma,
                                                                                     need_imputed_x_initial_guess,
                                                                                     not_imputed_all_x,
                                                                                     nan_position_array,
                                                                                     maxfun=5000,
                                                                                     maxiter=5000, bounds=bounds,
                                                                                     bias=None,
                                                                                     verbose=False)
                else:
                    optimze_zxys = initial_guess.copy()

            cell_idx = chrom_ids_to_index_dict[one_FISH_name]
            chr_idx = chr_name_list.index(chr_name)
            optimize_all_cell_dna_list[cell_idx][chr_idx] = optimze_zxys

        corr_dict = {}
        corr_dict['optimize_all_cell_dna_list'] = optimize_all_cell_dna_list
        one_chr_corr_dict = {}
        one_chr_corr_dict[chr_name] = corr_dict
        return chr_name, one_chr_corr_dict
    @staticmethod
    def get_impute_dna_list(chr_bin_length_dic, chr_name_list,
                              ImputeHiFI_adata, all_cell_dna_list,
                              Linear_impute_all_cell_dna_list,
                              chr_cell_type_center_value_df, chrom_ids_to_index_dict, loci_pair_type,
                              n_loci_pair=6000,kernerl_number=None):
        if kernerl_number is None:
            kernerl_number = len(chr_name_list)

        # n_loci_pair=6000
        if loci_pair_type == 'leiden_diff':
            diff_genes, not_diff_genes, non_diff_genes_mask = get_diffgene_with_leiden(
                ImputeHiFI_adata.copy(), n_loci_pair=n_loci_pair)

        if loci_pair_type == 'highly_variable_loci_pair':
            diff_genes, not_diff_genes, non_diff_genes_mask = get_highly_variable_genes(
                ImputeHiFI_adata.copy(),
                n_loci_pair=n_loci_pair)
        if loci_pair_type == 'no_diff':
            diff_genes = []
            not_diff_genes = []
            non_diff_genes_mask = np.zeros(ImputeHiFI_adata.shape[1], dtype=bool)
        # sc.pp.normalize_total(ImputeHiFI_adata, target_sum=1e4)
        ImputeHiFI_diff_genes_adata = ImputeHiFI_adata.copy()
        ImputeHiFI_diff_genes_adata.X[:, non_diff_genes_mask] = 0

        all_FISH_name = list(ImputeHiFI_adata.obs_names)

        chr_name_split_list = np.array_split(chr_name_list, kernerl_number)

        import multiprocessing as mp
        num_threads = kernerl_number

        _domain_args = [(chr_bin_length_dic, chr_name_part[0], chr_name_list, all_FISH_name,
                         ImputeHiFI_adata, ImputeHiFI_diff_genes_adata, all_cell_dna_list,
                         Linear_impute_all_cell_dna_list,
                         chr_cell_type_center_value_df, chrom_ids_to_index_dict, loci_pair_type)
                        for chr_name_part in chr_name_split_list]
        with mp.Pool(num_threads) as domain_pool:
            domain_results = domain_pool.starmap(generate_impute_dna_list.get_impute_dna_list_one_chr_result, _domain_args)
            domain_pool.close()
            domain_pool.join()
            domain_pool.terminate()

        finall_optimize_all_cell_dna_list = copy.deepcopy(all_cell_dna_list)
        for result in domain_results:
            print("Collect result")
            chr_name, one_chr_corr_dict = result
            optimize_all_cell_dna_list = one_chr_corr_dict[chr_name]['optimize_all_cell_dna_list']
            chr_idx = chr_name_list.index(chr_name)
            for cell_idx in range(len(optimize_all_cell_dna_list)):
                finall_optimize_all_cell_dna_list[cell_idx][chr_idx] = optimize_all_cell_dna_list[cell_idx][chr_idx]

        return finall_optimize_all_cell_dna_list,diff_genes
    @staticmethod
    def fill_all_nan_chr_with_linear(all_cell_dna_list, Linear_impute_all_cell_dna_list):
        for cell_idx in range(len(all_cell_dna_list)):
            for chr_idx in range(len(all_cell_dna_list[cell_idx])):
                if np.isnan(all_cell_dna_list[cell_idx][chr_idx]).all():
                    if ~np.isnan(Linear_impute_all_cell_dna_list[cell_idx][chr_idx]).all():
                        all_cell_dna_list[cell_idx][chr_idx] = Linear_impute_all_cell_dna_list[cell_idx][chr_idx]
                        print("Fill linear impute value", cell_idx, chr_idx)
                    else:
                        print("No linear impute value", cell_idx, chr_idx)
        return all_cell_dna_list


class file_convert(object):
    @staticmethod
    def convert_coord_df_to_dna_list(coord_df,cell_name_list,chr_name_list,chr_bin_length_dic):
        # 预先分组数据，以减少循环中的重复筛选
        grouped_df = coord_df.groupby(['cell_name', 'chr'])

        # 初始化模板
        template_dna_array = [np.full((length, 3), np.nan) for length in chr_bin_length_dic.values()]

        # 生成所有细胞的 DNA 列表
        generate_all_cell_dna_list = []
        for cell_name in tqdm(cell_name_list):
            one_cell_dna_list = [np.copy(arr) for arr in template_dna_array]  # 使用 np.copy 来替代 deepcopy
            for chr_index, chr_name in enumerate(chr_name_list):
                # 尝试从分组中获取数据
                try:
                    temp_df = grouped_df.get_group((cell_name, chr_name))
                except KeyError:
                    print(f"{cell_name}, {chr_name} not found in grouped data")
                    continue

                bin_length = chr_bin_length_dic[chr_name]
                if len(temp_df) != bin_length:
                    print(cell_name, chr_index, 'not equal')

                # 确保索引是连续的，从1开始
                temp_df = temp_df.set_index('pos')
                pos_index = np.arange(1, bin_length + 1)
                temp_array = temp_df.reindex(pos_index)[['z', 'x', 'y']].to_numpy()

                one_cell_dna_list[chr_index] = temp_array  # 更新对应染色体的数据

            generate_all_cell_dna_list.append(one_cell_dna_list)
        return generate_all_cell_dna_list
    @staticmethod
    def convert_dna_list_to_coord_df_ann_df(data,cell_type,chr_name):
        chr_name_list = data['chr_name_list']
        start_position = data['start_position']
        end_position = data['end_position']
        chr_bin_length_dic = data['chr_bin_length_dic']
        adata_obs_df = data['adata_obs_df']
        chrom_ids_to_index_dict = data['chrom_ids_to_index_dict']
        all_cell_dna_list = data['all_cell_dna_list']
        used_cell_name_array = adata_obs_df[adata_obs_df['cell_type'] == cell_type]['cell_name'].values
        chr_index = chr_name_list.index(chr_name)
        bin_length = chr_bin_length_dic[chr_name]
        pos = list(np.arange(1, bin_length + 1))

        loop_ann_df = pd.DataFrame()

        loop_ann_df['start'] = start_position[chr_index]
        loop_ann_df['end'] = end_position[chr_index]
        loop_ann_df['pos'] = pos
        loop_ann_df['chr'] = chr_name

        loop_ann_df = loop_ann_df[['chr', 'start', 'end', 'pos']]
        loop_data_df = pd.DataFrame()
        chr_list = []
        cell_id_list = []
        pos_list = []
        z_list = []
        x_list = []
        y_list = []
        for one_cell_name in used_cell_name_array:
            one_cell_index = chrom_ids_to_index_dict[one_cell_name]
            z_list += list(all_cell_dna_list[one_cell_index][chr_index][:, 0])
            x_list += list(all_cell_dna_list[one_cell_index][chr_index][:, 1])
            y_list += list(all_cell_dna_list[one_cell_index][chr_index][:, 2])
            chr_list += [chr_name] * bin_length
            cell_id_list += [one_cell_name] * bin_length
            pos_list += pos

        loop_data_df['chr'] = chr_list
        loop_data_df['cell_id'] = cell_id_list
        loop_data_df['pos'] = pos_list
        loop_data_df['x'] = x_list
        loop_data_df['y'] = y_list
        loop_data_df['z'] = z_list
        cleaned_data_df = loop_data_df.dropna(subset=['x', 'y', 'z'])
        cleaned_data_df = cleaned_data_df.reset_index(drop=True)
        return cleaned_data_df,loop_ann_df
