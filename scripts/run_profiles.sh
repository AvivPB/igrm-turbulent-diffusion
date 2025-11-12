#!/bin/bash -l

module load StdEnv/2023 python/3.13
source /scratch/aspadawe/igrm-turbulent-diffusion/pyenvs/main/bin/activate

sim=simbac
code=Simba-C
redshift=0

xscale=R500
ndim=3
filter=Sphere
profile_type=Profile
weight_by=mass

temp_cut='5e5 K'
nh_cut='0.13 cm**-3'

snap_dir=/scratch/aspadawe/snapshots/HyenasC/L1/SimbaC_L0_Calibration/halo_3224_og_good
snap_file=snapshot_151
caesar_dir=$snap_dir/caesar_snap
caesar_file=caesar_151
profiles_dir=$caesar_dir/profiles
save_file="$sim"-"$snap_file"-"$ndim"d_"${filter,,}"_profiles-xscale_"${xscale,,}"-temp_cut"=$temp_cut"-nh_cut"=$nh_cut"
suffix=all_props

halo_particles=--halo_particles
dm_particles=--dm_particles
bh_particles=--bh_particles
gas_particles=--gas_particles
igrm_particles=--igrm_particles

calc_thermo_props=--calc_thermo_props
calc_metal_props=--calc_metal_props
calc_gradients=--no-calc_gradients
calc_log_props=--calc_log_props

halo_ids=0
# Read halo_ids from file
# halo_file=$caesar_dir/halo_progen_info_snap_145
# while IFS=$'\t' read -r -a line; do
#     # source_snap_nums+="${line[1]} "
#     halo_ids+="${line[8]} "
# done < $halo_file

echo halo_ids:
echo $halo_ids
echo

echo $snap_file
echo $caesar_file
echo $save_file


# echo
# echo 'CALCULATING PROFILES'
# echo

# python gen_nd_profile_by_halo_id_v2.py --code=$code --snap_file=$snap_dir/$snap_file.hdf5 --caesar_file=$caesar_dir/$caesar_file.hdf5 --halo_ids $halo_ids --save_file=$profiles_dir/"$save_file" --filter=$filter --profile_type=$profile_type --ndim=$ndim --weight_by=$weight_by --xscale=$xscale --temp_cut="$temp_cut" --nh_cut="$nh_cut" $halo_particles $dm_particles $bh_particles $gas_particles $igrm_particles

# echo
# echo

echo
echo 'CALCULATING EXTRA PROPERTIES OF PROFILES'
echo

python calc_profile_properties.py --dir=$profiles_dir --name="$save_file" --caesar_file=$caesar_dir/$caesar_file.hdf5 --suffix=$suffix --code=$code --ndim=$ndim $calc_thermo_props $calc_metal_props $calc_gradients $calc_log_props

echo
echo 'done'

########################################################
