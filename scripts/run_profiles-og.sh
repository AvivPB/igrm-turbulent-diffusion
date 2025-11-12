#!/bin/bash -l

# module load NiaEnv/2022a python/3.11.5
# source /project/b/babul/aspadawe/pyenvs/gen_profiles/bin/activate

source /scratch/aspadawe/igrm-turbulent-diffusion/pyenvs/main/bin/activate

sim=hyenasc
code=Simba-C
redshift=0

xscale=R500
ndim=3
filter=Sphere
profile_type=Profile
weight_by=mass

temp_cut='5e5 K'
nh_cut='0.13 cm**-3'

halo_ids=0
#1
#3

snap_dir=/scratch/aspadawe/hyenas-entropy-profiles-calibration/snapshots/HyenasC/L1/SimbaC_L1_Calibration/halo_3224-BHSeedMass_1e-7
#/scratch/aspadawe/igrm-turbulent-diffusion/snapshots/HyenasC/L1/SimbaC_L1_Calibration/halo_3224
#/scratch/b/babul/aspadawe/snapshots/HyenasC/L0/SimbaC_L1_Calibration/halo_3224_og_good
# /scratch/b/babul/wcui/HYENAS/Level0/halo_3224
#/scratch/b/babul/aspadawe/snapshots/HyenasC/L1/SimbaC_L1_Calibration/halo_3224-BHSeedMass_1e-7
snap_file=snapshot_151
#snap_halo_3224_151
caesar_dir=/scratch/aspadawe/hyenas-entropy-profiles-calibration/snapshots/HyenasC/L1/SimbaC_L1_Calibration/halo_3224-BHSeedMass_1e-7/caesar_snap
#/scratch/aspadawe/igrm-turbulent-diffusion/snapshots/HyenasC/L1/SimbaC_L1_Calibration/halo_3224/caesar_snap
#/scratch/b/babul/aspadawe/snapshots/HyenasC/L0/SimbaC_L1_Calibration/halo_3224_og_good/caesar_fof
#/scratch/b/babul/aspadawe/snapshots/Hyenas/L0/Simba_L0_Calibration/halo_3224_weiguang/caesar_snap/aperture_30ckpc
# /scratch/b/babul/aspadawe/snapshots/HyenasC/L1/SimbaC_L1_Calibration/halo_3224-BHSeedMass_1e-7/caesar_fof
#/scratch/b/babul/wcui/HYENAS/Level0/halo_3224/Groups
caesar_file=caesar_151
#Caesar_halo_3224_151
#caesar_151_haloid-fof_lowres-[2]
profiles_dir=/scratch/aspadawe/hyenas-entropy-profiles-calibration/snapshots/HyenasC/L1/SimbaC_L1_Calibration/halo_3224-BHSeedMass_1e-7/caesar_snap/profiles
#/scratch/aspadawe/igrm-turbulent-diffusion/snapshots/HyenasC/L1/SimbaC_L1_Calibration/halo_3224/caesar_snap/profiles
#/scratch/b/babul/aspadawe/snapshots/HyenasC/L0/SimbaC_L1_Calibration/halo_3224_og_good/caesar_fof/profiles
#/scratch/b/babul/aspadawe/snapshots/Hyenas/L0/Simba_L0_Calibration/halo_3224_weiguang/caesar_snap/aperture_30ckpc/profiles
#/scratch/b/babul/aspadawe/snapshots/HyenasC/L1/SimbaC_L1_Calibration/halo_3224-BHSeedMass_1e-7/profiles
#/project/b/babul/aspadawe/snapshots/Hyenas/L0/halo_3224/profiles
save_file="$sim"-"$snap_file"-z"$redshift"-halo_"$halo_ids"-"$ndim"d_"${filter,,}"_profiles-xscale_"${xscale,,}"-temp_cut"=$temp_cut"-nh_cut"=$nh_cut"
suffix=all_props

echo $snap_file
echo $caesar_file
echo $save_file

# profiles_file="$save_file"-temp_cut"=$temp_cut"-nh_cut"=$nh_cut"
# echo $profiles_file

echo
echo 'CALCULATING PROFILES'
echo

python gen_nd_profile_by_halo_id_v2.py --code=$code --snap_file=$snap_dir/$snap_file.hdf5 --caesar_file=$caesar_dir/$caesar_file.hdf5 --halo_ids=$halo_ids --save_file=$profiles_dir/"$save_file" --filter=$filter --profile_type=$profile_type --ndim=$ndim --weight_by=$weight_by --xscale=$xscale --temp_cut="$temp_cut" --nh_cut="$nh_cut"

echo
echo 'CALCULATING EXTRA PROPERTIES OF PROFILES'
echo

python calc_profile_properties.py --dir=$profiles_dir --name="$save_file" --caesar_file=$caesar_dir/$caesar_file.hdf5 --suffix=$suffix --code=$code --ndim=$ndim --calc_thermo_props --calc_metal_props --calc_gradients --calc_log_props

echo
echo 'done'

########################################################
