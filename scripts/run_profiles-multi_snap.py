import sys
import os
import subprocess
import numpy as np

# Directory containing gen_nd_profile_by_halo_id_v2.py
GEN_SCRIPT = os.path.join('.', 'gen_nd_profile_by_halo_id_v2.py')
CALC_SCRIPT = os.path.join('.', 'calc_profile_properties.py')

gen_profiles = True
calc_profiles = True

# List of snapshot numbers and halo ids to process
# snap_nums = range(151, -1, -1)
snap_nums = [151]

# Arguments of script
snap_dir = '/scratch/aspadawe/snapshots/HyenasC/L0/SimbaC_L0_Calibration/halo_3224_og_good_rennehan/'
caesar_dir = os.path.join(snap_dir, 'caesar_fof')
profiles_dir = os.path.join(caesar_dir, 'profiles')

## List of halo ids to process for each snapshot, where each element is a list of halo ids for the corresponding snapshot
# halo_ids = np.loadtxt(os.path.join(caesar_dir, 'halo_info_snap_150'), usecols=2, dtype=str)
# print(list(halo_ids[:]))
# halo_ids = [list(halo_ids[:])]
halo_ids = [[1]]

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

suffix = 'igrm_props'

# Example additional arguments
# HALO_ID = 12345
# OUTPUT_DIR = "/path/to/output"
# CONFIG_FILE = "/path/to/config.yaml"

def run_gen_script(snap_num, halo_id):
    # halo_id = str(halo_id)
    # halo_id = ','.join(halo_id)
    halo_ids_str = ','.join(str(x) for x in halo_id)

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
    # save_file = f'{sim}-{snap_base}{snap_num:03d}-halo_{halo_id}-{ndim}d_{filt}_profiles-xscale_{xscale}-temp_cut={temp_cut}-nh_cut={nh_cut}'
    save_file = f'{sim}-{snap_base}{snap_num:03d}-{ndim}d_{filt}_profiles-xscale_{xscale}-temp_cut={temp_cut}-nh_cut={nh_cut}'

    if gen_profiles:
        print('\nCALCULATING PROFILES\n')
        
        # Arguments of gen script
        args = [
            sys.executable, '-u', GEN_SCRIPT,
            '--code', code,
            '--snap_file', snap_file,
            '--caesar_file', caesar_file,
            # '--halo_ids', halo_id,
            '--halo_ids', *halo_ids_str.split(','),  # Expand the comma-separated string into separate arguments
            '--save_file', os.path.join(profiles_dir, save_file),
            '--filter', filt,
            '--profile_type', profile_type,
            '--ndim', ndim,
            '--weight_by', weight_by,
            '--xscale', xscale,
            '--temp_cut', temp_cut,
            '--nh_cut', nh_cut,
            # '--halo_particles',
            # '--dm_particles',
            # '--bh_particles',
            # '--gas_particles',
            '--igrm_particles',
        ]
        
        # Run gen script as a subprocess with additional arguments
        # process = subprocess.run(args, capture_output=True, text=True)#, shell=True)
        # print(process.stdout)

        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            # text=True,
            # bufsize=1,
            universal_newlines=True
        )
        # Read output in real-time
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        # while True:
        #     output = process.stdout.readline()
        #     if output:
        #         print(output.strip())
        #     if process.poll() is not None:
        #         break

        process.stdout.close()
        return_code = process.wait()

        if return_code != 0:
            print(f"Error processing snapshot {snap_num}")
            return

        # if process.returncode != 0:
        #     print(f"Error processing snapshot {snap_num}:")
        #     print(process.stderr)
        #     print()
        #     return
        # else:
        #     print(f"Snapshot {snap_num} processed successfully.\n")


    if calc_profiles:
        print('\nCALCULATING EXTRA PROPERTIES OF PROFILES\n')
        
        # Arguments of calc script
        args = [
            sys.executable, '-u', CALC_SCRIPT,
            '--dir', profiles_dir,
            '--name', save_file,
            '--caesar_file', caesar_file,
            '--suffix', suffix,
            '--code', code,
            '--ndim', ndim,
            '--calc_thermo_props',
            '--calc_metal_props',
            # '--calc_gradients',
            '--calc_log_props'
        ]
        
        # Run calc script as a subprocess with additional arguments
        # process = subprocess.run(args, capture_output=True, text=True)#, shell=True)
        # print(process.stdout)

        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            # text=True,
            # bufsize=1,
            universal_newlines=True
        )
        # Read output in real-time
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        # while True:
        #     output = process.stdout.readline()
        #     if output:
        #         print(output.strip())
        #     if process.poll() is not None:
        #         break

        process.stdout.close()
        return_code = process.wait()

        if return_code != 0:
            print(f"Error processing snapshot {snap_num}")
            return

        # if process.returncode != 0:
        #     print(f"Error processing snapshot {snap_num}:")
        #     print(process.stderr)
        #     print()
        #     return
        # else:
        #     print(f"Snapshot {snap_num} processed successfully.\n")


def main():
    for snap_num, halo_id in zip(snap_nums, halo_ids):
        run_gen_script(snap_num, halo_id)

if __name__ == "__main__":
    # subprocess.run(['source', '/scratch/aspadawe/igrm-turbulent-diffusion/pyenvs/main/bin/activate'], shell=True) # Activate the virtual environment
    main()
    # subprocess.run(['deactivate'], shell=True)  # Deactivate the virtual environment