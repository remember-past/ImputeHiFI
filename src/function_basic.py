import os
import time
import datetime
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
from scipy.stats import zscore
import copy
import pickle
from tqdm import tqdm
import importlib
import portalocker
import threading

class basic_function(object):
    @classmethod
    def test(cls):
        print("0")
    @classmethod
    def store_variable_from_pikle_file(cls,file_name,variable):
        with open(file_name,'wb') as f:
            pickle.dump(variable,f,protocol=-1)

    @classmethod
    def store_variable_from_pikle_file_with_lock(cls, file_name, variable,timeout=20):
        with portalocker.Lock(file_name, mode='wb',flags=portalocker.LOCK_EX, timeout=timeout) as f:
            pickle.dump(variable, f, protocol=-1)
    @classmethod
    def load_variable_from_pikle_file(cls,file_name):
        with open(file_name,'rb') as f:
            variable=pickle.load(f)
        return variable

    @classmethod
    def load_variable_from_pikle_file_with_lock(cls, file_name,timeout=20):
        with portalocker.Lock(file_name, mode='rb',flags=portalocker.LOCK_EX, timeout=timeout) as f:
            variable = pickle.load(f)
        return variable

    @classmethod
    def recursive_mkdir(cls,dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    @classmethod
    def write_out_chr_pos_df(cls,sc_chr_pos_df, raw_data_dir, cell_name, file_name):
        cls.recursive_mkdir(os.path.join(raw_data_dir, cell_name))
        chr_pos_file = os.path.join(raw_data_dir, cell_name, file_name)
        sc_chr_pos_df.to_csv(chr_pos_file, sep='\t')

    @classmethod
    def read_chr_pos(cls,raw_data_dir, cell_name, file_name):
        chr_pos_file = os.path.join(raw_data_dir, cell_name, file_name)
        sc_chr_pos_df = pd.read_table(chr_pos_file, index_col=0)
        return sc_chr_pos_df

    @staticmethod
    def write_out_array_matrix(out_file_name,array_matrix):
        np.savetxt(out_file_name,array_matrix,delimiter='\t')
    @staticmethod
    def read_array_matrix(out_file_name):
        array_matrix=np.genfromtxt(out_file_name,delimiter='\t')
        return array_matrix

    @staticmethod
    def read_df(out_file_name):
        read_df = pd.read_csv(out_file_name, delimiter='\t',index_col=0,header=0)
        return read_df

    @staticmethod
    def write_out_df(out_file_name,one_df):
        one_df.to_csv(out_file_name, sep='\t')

    @staticmethod
    def log_using_time(output_file,using_time):
        using_time = "Using time {}".format(using_time)
        a = np.array([using_time])
        np.savetxt(output_file,a,fmt="%s")

    @staticmethod
    def get_array_with_specifice_index(array,row_index,col_index):
        array=array[np.ix_(row_index, col_index)]
        return array

    @staticmethod
    def read_list(out_file_name):
        import csv
        reader_result = []
        with open(out_file_name, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                reader_result.append(row)
        return out_file_name

    @staticmethod
    def write_out_list(out_file_name,out_list):
        import csv
        with open(out_file_name, 'w', newline='') as f:
            write = csv.writer(f, delimiter='\t')
            write.writerows(out_list)
    @staticmethod
    def get_chr_length_info( chr_length_file):
        chr_length_dic = {}
        # chr1	197195432
        # chr2	181748087
        file_handle = open(chr_length_file, 'r')
        for line in file_handle.readlines():
            temp = line.strip('\n').split('\t')
            chr_length_dic[temp[0]] = int(temp[1])
        return chr_length_dic

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
    def get_position(DNA_probe_df):
        ##############################################
        ################start position################
        ##############################################
        '''

        :param DNA_probe_df:
        :return: axis=0, chromID
                  axis=1, regionID_chrom start
        '''
        # chromID_set = np.unique(DNA_probe_df['chromID'])
        temp_chromID = DNA_probe_df['chromID'].values
        chromID_set=np.unique(temp_chromID[np.where(~np.isnan(temp_chromID))])

        start_position = []
        end_position = []
        mid_position = []
        mid_position_Mb = []
        chrom_name = []
        gene_ID = []
        chrom_region_name=[]
        for one_chromID in chromID_set:
            temp = DNA_probe_df[DNA_probe_df['chromID'] == one_chromID]
            one_chrom_name = temp['Chrom'].values[0]
            chrom_name.append(one_chrom_name)

            one_gene_ID = temp['geneID'].values
            one_gene_ID = one_gene_ID.tolist()
            gene_ID.append(one_gene_ID)

            one_chromID_start_position = temp['Start'].values
            one_chromID_end_position = temp['End'].values
            one_chromID_mid_position = one_chromID_start_position + (
                    one_chromID_end_position - one_chromID_start_position) / 2
            start_position.append(one_chromID_start_position)
            end_position.append(one_chromID_end_position)
            mid_position.append(one_chromID_mid_position)
            one_chromID_mid_position_Mb = one_chromID_mid_position / 1000000
            mid_position_Mb.append(one_chromID_mid_position_Mb)

            one_chrom_region_name = []
            one_chr_name = one_chrom_name
            for one_start, one_end in zip(one_chromID_start_position, one_chromID_end_position):
                one_region_name = one_chr_name + ':' + str(one_start) + '-' + str(one_end)
                one_chrom_region_name.append(one_region_name)
            chrom_region_name.append(one_chrom_region_name)
        return start_position,end_position,mid_position,mid_position_Mb,chrom_name,gene_ID,chrom_region_name
