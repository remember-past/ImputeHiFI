from multiprocessing import Pool
import os,time,random
import datetime
import numpy as np
python_file=r"/mnt/disk1/scfan/software/anaconda3/envs/ImputeHiFI/bin/python"



def run_with_one_function(title,args_list,p_num=1):
    from datetime import datetime
    # print('start run step one prepare data spatial_dist_bin_pair_df_script.py')
    print("start "+title)
    start_time = datetime.now()
    print('Parent process %s.' % os.getpid())
    p = Pool(p_num)

    all_parameter_list_list=args_list
    # print(all_parameter_list_list)

    p.starmap(run_script, all_parameter_list_list)
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    end_time=datetime.now()

    print('All subprocesses done.Using',end_time-start_time)
    print("End " + title)

if __name__ =="__main__":

    p_num=1

    def ImputeHiFI_mode_1_RUN():

        short_range_HiC_weight_bound_list = [0.4] * 1
        long_range_HiC_weight_bound_list = [0.1] * 1
        all_range_diag_sum_weight_HiC_bound_list = [0.1] * 1
        short_range_diag_list = [20] * 1
        background_cell_num_list = [100] * 1

        n_loci_pair_list=[6000]

        version = 'ImputeHiFI mode 1'
        use_mean_count = 0
        infer_3D_type = r"_part_loci_pair_weight_"
        loci_pair_type = 'leiden_diff'
        title = "step three Impute"
        script_dir = r"./ImputeHiFI"
        script_file = os.path.join(script_dir,
                                   "ImputeHiFi_mode_1_step_three_Impute_multiprocess_script.py")

        args_list = []

        similar_FISH_cell_path = os.path.join(
            r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/ImputHiFI/first_seen_probe_data",
            "All_cell_RNA_FISH_RMSD.pkl")
        for short_range_HiC_weight_bound, long_range_HiC_weight_bound, all_range_diag_sum_weight_HiC_bound, short_range_diag, background_cell_num in zip(
                short_range_HiC_weight_bound_list, long_range_HiC_weight_bound_list,
                all_range_diag_sum_weight_HiC_bound_list, short_range_diag_list, background_cell_num_list):
            args_list.append(
                [[script_file, str(short_range_HiC_weight_bound),str(long_range_HiC_weight_bound),
              str(all_range_diag_sum_weight_HiC_bound),str(short_range_diag),
              str(background_cell_num),similar_FISH_cell_path,version,str(use_mean_count)]])
        run_with_one_function(title, args_list, p_num=p_num)

        title = "step four infer3D"
        script_dir = r"./ImputeHiFI"
        script_file = os.path.join(script_dir,
                                   "ImputeHiFi_mode_1_step_four_infer3D_part_loci_pair_weight_multiprocess_script")

        args_list = []

        temp_dir = r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/ImputHiFI/first_seen_probe_data/"
        DNA_FISH_path = os.path.join(temp_dir, "ImputeHiFI_v2_proximity_score_bin_pair_adata.h5ad")

        data_dir = r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/ImputHiFI"

        ImputeHiFI_dir = r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/ImputHiFI/first_seen_probe_data_Impute"
        Linear_impute_dir =  r"/mnt/disk1/scfan/data/Cai_21_bioRxiv/ImputHiFI/first_seen_probe_data_Linear/"

        for short_range_HiC_weight_bound, long_range_HiC_weight_bound, all_range_diag_sum_weight_HiC_bound, short_range_diag, background_cell_num ,n_loci_pair in zip(
                short_range_HiC_weight_bound_list, long_range_HiC_weight_bound_list,
                all_range_diag_sum_weight_HiC_bound_list, short_range_diag_list, background_cell_num_list,n_loci_pair_list):
            args_list.append(
                [[script_file, str(short_range_HiC_weight_bound), str(long_range_HiC_weight_bound),
                  str(all_range_diag_sum_weight_HiC_bound), str(short_range_diag),
                  str(background_cell_num), DNA_FISH_path, data_dir, ImputeHiFI_dir, Linear_impute_dir, version,
                  infer_3D_type+f"{loci_pair_type}_{n_loci_pair}_",str(n_loci_pair),loci_pair_type]])
        run_with_one_function(title, args_list, p_num=p_num)

    ImputeHiFI_mode_1_RUN()
