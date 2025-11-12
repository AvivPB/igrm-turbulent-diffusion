import sys
import os
import subprocess
import numpy as np

# Directory containing gen_nd_profile_by_halo_id_v2.py
GEN_SCRIPT = os.path.join('.', 'gen_nd_profile_by_halo_id_v2.py')
CALC_SCRIPT = os.path.join('.', 'calc_profile_properties.py')

# List of snapshot numbers and halo ids to process
snap_nums = range(151, -1, -1)

# Arguments of script
snap_dir = '/scratch/aspadawe/snapshots/HyenasC/L1/SimbaC_L0_Calibration/halo_3224_og_good/'
#'/scratch/aspadawe/hyenas-entropy-profiles-calibration/snapshots/HyenasC/L1/SimbaC_L1_Calibration/halo_3224-BHSeedMass_1e-7/'
# '/scratch/aspadawe/igrm-turbulent-diffusion/snapshots/HyenasC/L0/SimbaC_L0_Calibration/halo_3224_good/'
caesar_dir = os.path.join(snap_dir, 'caesar_snap')
profiles_dir = os.path.join(caesar_dir, 'profiles')

halo_ids = np.loadtxt(os.path.join(caesar_dir, 'halo_progen_info'), usecols=9, dtype=str)
print(halo_ids)
print(type(halo_ids))
print(type(halo_ids[0]))

snap_base = 'snapshot_'
caesar_base = 'caesar_'
caesar_suffix = ''

sim = 'hyenasc'
code = 'Simba-C'

xscale = 'R500'
ndim = '3'
filt = 'Sphere'
profile_type = 'Profile'
weight_by = 'mass'

temp_cut = '5e5 K'
nh_cut = '0.13 cm**-3'

suffix = 'all_props'

# Example additional arguments
# HALO_ID = 12345
# OUTPUT_DIR = "/path/to/output"
# CONFIG_FILE = "/path/to/config.yaml"

def run_gen_script(snap_num, halo_id):
    # halo_id = int(halo_id)
    # print(halo_id)
    snap_file = os.path.join(snap_dir, f"{snap_base}{snap_num:03d}.hdf5")
    if not os.path.exists(snap_file):
        print(f"Snapshot file {snap_file} not found, skipping.")
        return

    caesar_file = os.path.join(caesar_dir, f"{caesar_base}{snap_num:03d}{caesar_suffix}.hdf5")
    if not os.path.exists(caesar_file):
        print(f"Caesar file {caesar_file} not found, skipping.")
        return
    
    print(f"\nProcessing snapshot {snap_num}...\n")

    print('\nCALCULATING PROFILES\n')
    
    # Arguments of gen script
    # save_file = f'{sim}-{snap_base}{snap_num:03d}-halo_{halo_id}-{ndim}d_{filt}_profiles-xscale_{xscale}-temp_cut={temp_cut}-nh_cut={nh_cut}'
    save_file = f'{sim}-{snap_base}{snap_num:03d}-{ndim}d_{filt}_profiles-xscale_{xscale}-temp_cut={temp_cut}-nh_cut={nh_cut}'
    args = [
        sys.executable, GEN_SCRIPT,
        '--code', code,
        '--snap_file', snap_file,
        '--caesar_file', caesar_file,
        '--halo_ids', halo_id,
        '--save_file', os.path.join(profiles_dir, save_file),
        '--filter', filt,
        '--profile_type', profile_type,
        '--ndim', ndim,
        '--weight_by', weight_by,
        '--xscale', xscale,
        '--temp_cut', temp_cut,
        '--nh_cut', nh_cut
    ]
    
    # Run gen script as a subprocess with additional arguments
    result = subprocess.run(args, capture_output=True, text=True)#, shell=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error processing snapshot {snap_num}:")
        print(result.stderr)
        print()
        return
    else:
        print(f"Snapshot {snap_num} processed successfully.\n")


    print('\nCALCULATING EXTRA PROPERTIES OF PROFILES\n')
    
    # Arguments of calc script
    args = [
        sys.executable, CALC_SCRIPT,
        '--dir', profiles_dir,
        '--name', save_file,
        '--caesar_file', caesar_file,
        '--suffix', suffix,
        '--code', code,
        '--ndim', ndim,
        '--calc_thermo_props',
        '--calc_metal_props',
        '--calc_gradients',
        '--calc_log_props'
    ]
    
    # Run calc script as a subprocess with additional arguments
    result = subprocess.run(args, capture_output=True, text=True)#, shell=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error processing snapshot {snap_num}:")
        print(result.stderr)
        print()
        return
    else:
        print(f"Snapshot {snap_num} processed successfully.\n")


def main():
    for snap_num, halo_id in zip(snap_nums, halo_ids):
        run_gen_script(snap_num, halo_id)

if __name__ == "__main__":
    # subprocess.run(['source', '/scratch/aspadawe/igrm-turbulent-diffusion/pyenvs/main/bin/activate'], shell=True) # Activate the virtual environment
    main()
    # subprocess.run(['deactivate'], shell=True)  # Deactivate the virtual environment