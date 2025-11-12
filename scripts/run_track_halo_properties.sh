#!/bin/bash -l

# module load NiaEnv/2022a python/3.11.5
# source /project/b/babul/aspadawe/pyenvs/gen_profiles/bin/activate

source /scratch/aspadawe/igrm-turbulent-diffusion/pyenvs/main/bin/activate


snap_dir=/scratch/aspadawe/snapshots/HyenasC/L1/SimbaC_L1_Calibration/halo_3224_grad/
#/scratch/b/babul/wcui/HYENAS/Level1/halo_3224/
#/scratch/b/babul/aspadawe/snapshots/HyenasC/L1/SimbaC_L1_Calibration/halo_3224-BHSeedMass_1e-7/
snap_base=snapshot_
#snap_halo_3224_
caesar_dir=/scratch/aspadawe/snapshots/HyenasC/L1/SimbaC_L1_Calibration/halo_3224_grad/caesar_snap/
#/scratch/b/babul/wcui/HYENAS/Level1/halo_3224/Groups/
#/scratch/b/babul/aspadawe/snapshots/Hyenas/L1/Simba_L0_Calibration/halo_3224_weiguang/Groups/
#/scratch/b/babul/aspadawe/snapshots/HyenasC/L1/SimbaC_L1_Calibration/halo_3224-BHSeedMass_1e-7/caesar_snap/
#/scratch/b/babul/wcui/HYENAS/Level0/halo_3224/Groups/
#/project/b/babul/aspadawe/snapshots/HyenasC/L1/SimbaC_L1_Calibration/halo_3224_grad/caesar_fof/
caesar_base=caesar_
#Caesar_halo_3224_
caesar_suffix=''
#_haloid-fof_lowres-[2]
source_snap_num=151
# target_snap_nums={0..151}
target_snap_nums=$(seq 8 1 151)
source_halo_id=0
nproc=1
output_file=/scratch/aspadawe/snapshots/HyenasC/L1/SimbaC_L1_Calibration/halo_3224_grad/caesar_snap/halo_"$source_halo_id"_props-v2
# /scratch/b/babul/aspadawe/snapshots/Hyenas/L1/Simba_L0_Calibration/halo_3224_weiguang/Groups/halo_"$source_halo_id"_props
#/scratch/b/babul/aspadawe/snapshots/HyenasC/L1/SimbaC_L1_Calibration/halo_3224-BHSeedMass_1e-7/caesar_snap/halo_"$source_halo_id"_props
#/project/b/babul/aspadawe/snapshots/Hyenas/L1/halo_3224/Groups/halo_"$source_halo_id"_props
#/project/b/babul/aspadawe/snapshots/HyenasC/L1/SimbaC_L1_Calibration/halo_3224_grad/caesar_fof/halo_"$source_halo_id"_props


echo target_snap_nums:
echo $target_snap_nums


echo
echo 'CALCULATING HALO PROPERTIES'
echo

python track_halo_properties-hdf5.py --snap_dir=$snap_dir --snap_base=$snap_base --caesar_dir=$caesar_dir --caesar_base=$caesar_base --caesar_suffix=$caesar_suffix --source_snap_num=$source_snap_num --target_snap_nums $target_snap_nums --source_halo_id=$source_halo_id --nproc=$nproc --output_file=$output_file.hdf5 --clear_output_file #> $output_file.out

echo
echo 'done'

########################################################
