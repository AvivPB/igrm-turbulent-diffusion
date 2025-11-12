## For dougpython: module load NiaEnv/2022a gcc/11.3.0 openssl/1.1.1k sqlite/3.35.5 hdf5/1.12.3

import yt
import unyt

import caesar
import numpy as np
# import matplotlib.pyplot as plt

import os
from pathlib import Path
import argparse


parser = argparse.ArgumentParser(prog='identify_halos.py', description='Find halo id and minpotpos of target halo in zoom simulation based on provided halo properties and contamination (optional) from caesar file')

# parser.add_argument('--snap_dir', action='store', type=str, required=True, 
#                     help='directory containing snapshots')
parser.add_argument('--caesar_dir', action='store', type=str, required=True, 
                    help='directory containing caesar files')
parser.add_argument('--snap_nums', action='store', nargs='*', type=int, required=True, 
                    help='Snapshot numbers')
parser.add_argument('--caesar_base', action='store', type=str, default='caesar_',
                    help='Base name for caesar files, e.g. caesar_')
parser.add_argument('--caesar_suffix', action='store', type=str, default='',
                    help='Suffix for caesar files, e.g. _haloid-fof_lowres-[2]')

parser.add_argument('--output_file', action='store', type=str, required=True,
                    help='Full path of output file')
# parser.add_argument('--clear_output_file', action='store', type=bool, default=False, choices=[True, False],
                    # help='Whether to clear the output file initially before writing to it')
parser.add_argument('--clear_output_file', action=argparse.BooleanOptionalAction, default=False, 
                    help='Whether to clear the output file initially before writing to it')

parser.add_argument('--target_property', action='store', nargs='*', type=str, default=[],
                    choices=['m2500c', 'm500c', 'm200c', 'r2500c', 'r500c', 'r200c', 'temperature'],
                    help='Halo property employed to identify target halo')
parser.add_argument('--domain', action='store', type=str, default='inside', choices=['inside', 'outside'],
                    help='Look for halos with target property inside or outside min and max target values')
parser.add_argument('--target_value_min', action='store', nargs='*', type=float,
                    help='Minimum value of target_property')
parser.add_argument('--target_value_max', action='store', nargs='*', type=float,
                    help='Maximum value of target_property')
parser.add_argument('--target_units', action='store', nargs='*', type=str,
                    help='Units of target_property')
# parser.add_argument('--target_value', action='store',
#                     help='Value of target_property')
# parser.add_argument('--target_value_tol', action='store', type=float,
#                     help='Tolerance of target halo value as a fraction, i.e. all halos with target_property value within +/- target_value_tol*target_value (inclusive) will be included')
# parser.add_argument('--target_value_is_log', action='store', type=bool, choices=[True, False],
#                     help='Whether provided value of target_property is linear (False) or log (True)')

# parser.add_argument('--use_contamination', action='store', type=bool, choices=[True, False],
#                     help='Use lowres contamination property to select halo')
parser.add_argument('--use_contamination', action=argparse.BooleanOptionalAction, default=False,
                    help='Use lowres contamination property to select halo')
parser.add_argument('--contamination_min', action='store', type=float, default=0,
                    help='Minimum contamination of target halo')
parser.add_argument('--contamination_max', action='store', type=float, default=0,
                    help='Maximum contamination of target halo')
# parser.add_argument('--contamination', action='store', type=float,
#                     help='Contamination of target halo')
# parser.add_argument('--contamination_tol', action='store', type=float,
                    # help='Tolerance of target halo contamination as a fraction, i.e. all halos with contamination within +/- contamination_tol*contamination (inclusive) will be included')

args = parser.parse_args()

# print()
# print(args.clear_output_file)
# print(args.use_contamination)
# print()


# if args.target_value_is_log:
#     target_units = None
# else:
#     target_units = args.target_units

# target_value = unyt.unyt_array(args.target_value, target_units)
# target_value_min = target_value * (1 - unyt.unyt_array(args.target_value_tol, target_units))
# target_value_max = target_value * (1 + unyt.unyt_array(args.target_value_tol, target_units))

# contamination_min = args.contamination * (1 - args.contamination_tol)
# contamination_max = args.contamination * (1 + args.contamination_tol)


if not os.path.exists(args.output_file):
    print('Making output file')
    # output_file = Path(args.output_file)
    # output_file.parent.mkdir(exist_ok=True, parents=True)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    f = open(args.output_file, 'w')
    f.close()
    print()

if args.clear_output_file:
    print('Clearing output file')
    f = open(args.output_file, 'w')
    f.close()
    print()

if not os.path.exists(f'{args.output_file}.header'):
    print('Making output file header')
    # os.makedirs(f'{args.output_file}.header')
    f = open(f'{args.output_file}.header', 'w')
    f.close()
    print()

print('Writing to output file header')
header = f'caesar_dir\tsnap_num\tz\thalo_id\tminpotpos_x (Mpccm/h)\tminpotpos_y (Mpccm/h)\tminpotpos_z (Mpccm/h)'
for target_property, target_units in zip(args.target_property, args.target_units):
    header += f'\t{target_property} ({target_units})'
if args.use_contamination:
    header += '\tcontamination'
with open(f'{args.output_file}.header', 'w') as f:
    f.write(header)
print()


# target_value_min = unyt.unyt_array(args.target_value_min, args.target_units)
# target_value_max = unyt.unyt_array(args.target_value_max, args.target_units)

# print()
# print(target_value_min)
# print(target_value_max)
# print()

for snap_num in args.snap_nums:
    # snap_file = os.path.join(args.snap_dir, f'{args.snap_base}{snap_num:03}.hdf5')
    # snap = yt.load(snap_file)

    ## Load caesar file
    caesar_file = os.path.join(args.caesar_dir, f'{args.caesar_base}{snap_num:03}{args.caesar_suffix}.hdf5')
    try:
        obj = caesar.load(caesar_file)
    except Exception as e:
        print(f'Error reading caesar file {caesar_file}: {e}')
        continue

    
    ## Get all relevant halo properties
    # halo_ids = unyt.unyt_array([halo.GroupID for halo in obj.halos])
    halo_ids = np.array([halo.GroupID for halo in obj.halos])
    
    # halo_minpotposes = unyt.unyt_array([halo.minpotpos.in_units('Mpccm/h') for halo in obj.halos])
    halo_minpotposes = np.array([halo.minpotpos.in_units('Mpccm/h') for halo in obj.halos])

    halo_prop_dict = {
        target_property:np.array([halo.virial_quantities[target_property].in_units(target_units) for halo in obj.halos]) for target_property, target_units in zip(args.target_property, args.target_units)
    }
    # print()
    # print(halo_prop_dict)
    # print()
    # halo_prop_values = unyt.unyt_array([halo.virial_quantities[args.target_property].in_units(args.target_units) for halo in obj.halos])
    # halo_prop_values = np.array([halo.virial_quantities[args.target_property].in_units(args.target_units) for halo in obj.halos])
    # print(halo_prop_values)
    # print(halo_prop_values.units)
    # print()
    # if args.target_value_is_log:
    #     halo_prop_value = np.log10(halo_prop_value)
        
    # halo_contaminations = unyt.unyt_array([halo.contamination for halo in obj.halos])
    if args.use_contamination:
        halo_contaminations = np.array([halo.contamination for halo in obj.halos])


    ## Find desired halo
    # target_filter = (halo_prop_values >= target_value_min) & (halo_prop_values <= target_value_max) & (halo_contaminations >= contamination_min) & (halo_contaminations <= contamination_max)
    # target_filter = (halo_prop_values >= args.target_value_min) & (halo_prop_values <= args.target_value_max) & (halo_contaminations >= args.contamination_min) & (halo_contaminations <= args.contamination_max)
    target_filter = np.full(shape=len(obj.halos), fill_value=True)
    # print()
    # print(len(obj.halos))
    # print(target_filter)
    # print()
    for target_property, target_value_min, target_value_max, in zip(args.target_property, args.target_value_min, args.target_value_max):
        halo_prop_values = halo_prop_dict[target_property]
        if args.domain == 'inside':
            target_filter = target_filter & (halo_prop_values >= target_value_min) & (halo_prop_values <= target_value_max)
        elif args.domain == 'outside':
            target_filter = target_filter & (halo_prop_values <= target_value_min) & (halo_prop_values >= target_value_max)
    # target_filter = (halo_prop_values >= args.target_value_min) & (halo_prop_values <= args.target_value_max)
    if args.use_contamination:
        target_filter = target_filter & (halo_contaminations >= args.contamination_min) & (halo_contaminations <= args.contamination_max)
    target_halo_indexes = np.where(target_filter)

    
    target_halo_ids = halo_ids[target_halo_indexes]
    target_halo_minpotposes = halo_minpotposes[target_halo_indexes]
    # target_halo_prop_values = halo_prop_values[target_halo_indexes]
    target_halo_prop_dict = {
        target_property:halo_prop_dict[target_property][target_filter] for target_property in args.target_property
    }
    if args.use_contamination:
        target_halo_contaminations = halo_contaminations[target_halo_indexes]


    print(f'Writing to output file for snap {snap_num}')
    with open(args.output_file, 'a') as f:
        for index in range(len(target_halo_ids)):
            halo_id = target_halo_ids[index]
            halo_minpotpos = target_halo_minpotposes[index]
            halo_props = [target_halo_prop_dict[target_property][index] for target_property in args.target_property]
            
            output = f'{args.caesar_dir}\t{snap_num}\t{obj.simulation.redshift}\t{halo_id}\t{halo_minpotpos[0]}\t{halo_minpotpos[1]}\t{halo_minpotpos[2]}'
            for halo_prop in halo_props:
                output += f'\t{halo_prop}'
                
            if args.use_contamination:
                halo_contamination = target_halo_contaminations[index]
                output += f'\t{halo_contamination}'
                
            output += '\n'
            f.write(output)
            
        # for halo_id, halo_minpotpos, halo_contamination, halo_prop_value in zip(target_halo_ids, target_halo_minpotposes, target_halo_contaminations, target_halo_prop_values):
        #     f.write(f'{args.caesar_dir}\t{snap_num}\t{halo_id}\t{halo_minpotpos}\t{halo_contamination}\t{halo_prop_value}\n')

    print()
    print()

    

    

    # halo_nonzero_contamination_index = np.nonzero(np.array(halo_contamination)!=0)[0]
    # halo_zero_contamination_index = np.nonzero(np.array(halo_contamination)==0)[0]

    # halo_prop_value_nonzero_contamination = np.array(halo_prop_value)[halo_nonzero_contamination_index]
    # halo_prop_value_zero_contamination = np.array(halo_prop_value)[halo_zero_contamination_index]

    # target_halo_index = halo_contamination0_index[np.argmin(np.abs(halo_m500c_contamination0 - target_m500c))]