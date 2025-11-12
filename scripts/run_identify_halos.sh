#!/bin/bash -l

# module load NiaEnv/2022a gcc/11.3.0 openssl/1.1.1k sqlite/3.35.5 hdf5/1.12.3

source /scratch/aspadawe/igrm-turbulent-diffusion/pyenvs/main/bin/activate

caesar_dir=/scratch/aspadawe/snapshots/HyenasC/L1/SimbaC_L1_Calibration/halo_3224_good-BH_QUENCH_JET_4000/caesar_snap
echo $caesar_dir

snap_nums=151
caesar_base=caesar_
caesar_suffix=''

output_file=$caesar_dir/halo_info_snap_${snap_nums}
clear_output_file=--clear_output_file
echo $output_file

target_property=m500c
domain='inside'
target_value_min=1e12
target_value_max=1e17
target_units=Msun

use_contamination=--use_contamination
contamination_min=0
contamination_max=0


python identify_halos.py --caesar_dir=$caesar_dir --snap_nums=$snap_nums --caesar_base=$caesar_base --caesar_suffix=$caesar_suffix --output_file=$output_file $clear_output_file --target_property=$target_property --domain=$domain --target_value_min=$target_value_min --target_value_max=$target_value_max --target_units=$target_units $use_contamination --contamination_min=$contamination_min --contamination_max=$contamination_max



########################################################

# main_dir=/scratch/b/babul/aspadawe/snapshots/HyenasC/L0/SimbaC_L1_Calibration/
# sub_dir=halo_3224_og_good/
# #halo_*/
# caesar_dir=caesar_fof/

# # DIRS=$(find $main_dir -type d)

# # --caesar_base=Caesar_$(basename $dir)_

# for dir in $main_dir$sub_dir; do

#         echo $dir
#         echo $(basename $dir)
#         echo

#         /scratch/b/babul/rennehan/for_aviv/python-3.13.5/bin/python identify_halos.py --caesar_dir=$dir$caesar_dir --snap_nums=151 --caesar_base=caesar_ --output_file=/project/b/babul/aspadawe/snapshots/Hyenas/L1/halo_3224/caesar_fof/halo_info --no-clear_output_file --target_property=m500c --target_value_min=1e12 --target_value_max=1e15 --target_units=Msun --use_contamination --contamination_min=0 --contamination_max=0

#         echo
#         echo
# done

# echo 'done'

########################################################
