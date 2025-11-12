import os
import argparse

import yt
import caesar
import numpy as np

## For dougpython: module load NiaEnv/2022a gcc/11.3.0 openssl/1.1.1k sqlite/3.35.5 hdf5/1.12.3

parser = argparse.ArgumentParser(prog='identify_halo_progenitors_daisychain.py', description='Find progenitors of halos in caesar files by daisy-chaining')
parser.add_argument('--snap_dir', action='store', type=str, required=True, 
                    help='directory containing snapshots')
parser.add_argument('--snap_base', action='store', type=str, default='snapshot_',
                    help='base name for snapshots, e.g. snapshot_')
parser.add_argument('--caesar_dir', action='store', type=str, required=True, 
                    help='directory containing caesar files')
parser.add_argument('--caesar_base', action='store', type=str, default='caesar_',
                    help='base name for caesar files, e.g. caesar_')
parser.add_argument('--source_snap_nums', action='store', nargs='*', type=int, required=True, 
                    help='Snapshot numbers for halos of which to find progenitors')
parser.add_argument('--target_snap_nums', action='store', nargs='*', type=int, required=True, 
                    help='Farthest back snapshot numbers in which to find halo progenitors')
parser.add_argument('--source_halo_ids', action='store', nargs='*', type=int, required=True, 
                    help='Ids of source halos')
parser.add_argument('--n_most', action='store', type=int, default=1, choices=[None, 1, 2],
                    help='caesar progen n_most option; find n_most progenitors/descendents (None = all)')
parser.add_argument('--nproc', action='store', type=int, default=1,
                    help='caesar progen nproc option')

parser.add_argument('--output_file', action='store', type=str, required=True,
                    help='Full path of output file')
parser.add_argument('--clear_output_file', action=argparse.BooleanOptionalAction, default=False, 
                    help='Whether to clear the output file initially before writing to it')
args = parser.parse_args()

print(args.n_most)


if not os.path.exists(args.output_file):
    print('Making output file')
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
    f = open(f'{args.output_file}.header', 'w')
    f.close()
    print()

print('Writing to output file header')
# header = 'caesar_dir\tsource_snap_num\tsource_z\tsource_halo_id\tsource_m500c (Msun)\tsource_contamination\ttarget_snap_num\ttarget_z\ttarget_halo_id\ttarget_m500c (Msun)\ttarget_contamination'
header = 'caesar_dir\tsource_snap_num\tsource_z\tsource_halo_id\tsource_m500c (Msun)\tsource_contamination\ttarget_snap_num\ttarget_z'
for n in range(args.n_most):
    header += f'\ttarget_halo_id-{n+1}\ttarget_m500c-{n+1} (Msun)\ttarget_contamination-{n+1}'
with open(f'{args.output_file}.header', 'w') as f:
    f.write(header)
print()


## Loop through source and target caesar files
for source_snap_num, source_halo_id, target_snap_num in zip(args.source_snap_nums, args.source_halo_ids, args.target_snap_nums):
    print(f'Source snap num: {source_snap_num}')
    print(f'Source halo id: {source_halo_id}')
    print(f'Target snap num: {target_snap_num}')
    if source_snap_num >= target_snap_num:
        intermediate_snap_nums = range(source_snap_num, target_snap_num-1, -1)
    else:
        intermediate_snap_nums = range(source_snap_num, target_snap_num+1, 1)
    print(f'Intermediate snap nums: {intermediate_snap_nums}\n\n')

    source_int_snap_nums = intermediate_snap_nums[:-1]
    #print(len(source_int_snap_nums))
    target_int_snap_nums = intermediate_snap_nums[1:]
    source_int_halo_ids = [source_halo_id]
    curr_source_halo_id = source_halo_id

    #for source_int_snap_num, target_int_snap_num in zip(intermediate_snap_nums[:-1], intermediate_snap_nums[1:]):
    for ii in range(len(source_int_snap_nums)):
        source_int_snap_num = source_int_snap_nums[ii]
        target_int_snap_num = target_int_snap_nums[ii]
        #source_int_halo_id = source_int_halo_ids[ii]
        source_int_halo_id = curr_source_halo_id

        print(f'Intermediate source snap num: {source_int_snap_num}')
        print(f'Intermediate target snap num: {target_int_snap_num}')
        print(f'Intermediate source halo id: {source_int_halo_id}\n')

        ## Load source snapshot
        source_snap_file = os.path.join(args.snap_dir, f'{args.snap_base}{source_int_snap_num:03}.hdf5')
        try:
            source_snap = yt.load(source_snap_file)
            z_source = source_snap.current_redshift
        except Exception as error:
            z_source = -1
            print(f'Error occurred loading source snapshot: {error}')
            print()
            # continue
    
        ## Load source caesar file
        source_caesar_file = os.path.join(args.caesar_dir, f'{args.caesar_base}{source_int_snap_num:03}.hdf5')
        try:
            source_obj = caesar.load(source_caesar_file)
        except Exception as error:
            print(f'Error occurred loading source caesar file: {error}')
            print()
            continue
        print(f'Source caesar file: {source_caesar_file}, z={z_source}')

        print(f'Source halo id: {source_int_halo_id}\n')
        source_int_halo = source_obj.halos[source_int_halo_id]


        ## Load target snapshot
        target_snap_file = os.path.join(args.snap_dir, f'{args.snap_base}{target_int_snap_num:03}.hdf5')
        try:
            target_snap = yt.load(target_snap_file)
            z_target = target_snap.current_redshift
        except Exception as error:
            z_target = -1
            print(f'Error occurred loading target snapshot: {error}')
            print()
	    
        ## Load target caesar file
        target_caesar_file = os.path.join(args.caesar_dir, f'{args.caesar_base}{target_int_snap_num:03}.hdf5')    
        try:
            target_obj = caesar.load(target_caesar_file)
        except Exception as error:
            print(f'Error occurred loading target caesar file: {error}')
            print()
            continue
        print(f'Target caesar file: {target_caesar_file}, z={z_target}\n')



        ## Link halos in snapshots with caesar progen
        # caesar.progen.check_if_progen_is_present(target_caesar_file, 'progen_halo_dm')
        progens = caesar.progen.progen_finder(obj_current=source_obj, obj_target=target_obj, 
                                              caesar_file=source_caesar_file, snap_dir=args.snap_dir,
                                              data_type='halo', part_type='dm', recompute=True,
                                              save=False, n_most=args.n_most, min_in_common=0.1, nproc=args.nproc,
                                              match_frac=True, reverse_match=False)
        print('\nprogens:')
        print(progens)
        print()
        
        ## For n_most=1
        '''
        target_halo_id = progens[0][source_halo_id][0]  # with match_frac=True
        # target_halo_id = progens[source_halo_id][0]  # with match_frac=False
        target_halo = target_obj.halos[target_halo_id]
        print(f'Target halo id: {target_halo_id}')
        print(f'Target halo m500c: {target_halo.virial_quantities['m500c']}')
        print(f'Target halo contamination: {target_halo.contamination}')
        print()
        '''
        ## For any n_most
        target_int_halo_ids = progens[0][source_int_halo_id]#[0]  # with match_frac=True
        # target_int_halo_ids = progens[source_int_halo_id]#[0]  # with match_frac=False
        target_int_halos = [target_obj.halos[target_int_halo_id] for target_int_halo_id in target_int_halo_ids]
        target_int_halo_m500c = [target_int_halo.virial_quantities['m500c'] for target_int_halo in target_int_halos]
        target_int_halo_contamination = [target_int_halo.contamination for target_int_halo in target_int_halos]
        print(f'Intermediate target halo ids: {target_int_halo_ids}')
        print(f'Intermediate target halo m500c: {target_int_halo_m500c}')
        print(f'Intermediate target halo contamination: {target_int_halo_contamination}')
        print()


        print(f'Writing to output file for source snap {source_int_snap_num}, target snap {target_int_snap_num}, source halo id {source_int_halo_id}')
        with open(args.output_file, 'a') as f:                    
            #output = f'{args.caesar_dir}\t{source_snap_num}\t{z_source}\t{source_halo_id}\t{source_halo.virial_quantities['m500c']}\t{source_halo.contamination}\t{target_snap_num}\t{z_target}\t{target_halo_id}\t{target_halo.virial_quantities['m500c']}\t{target_halo.contamination}\n'
            output = f'{args.caesar_dir}\t{source_int_snap_num}\t{z_source}\t{source_int_halo_id}\t{source_int_halo.virial_quantities['m500c']}\t{source_int_halo.contamination}\t{target_int_snap_num}\t{z_target}'
            for n in range(len(target_int_halo_ids)):
                output += f'\t{target_int_halo_ids[n]}\t{target_int_halo_m500c[n]}\t{target_int_halo_contamination[n]}'
            output += '\n'
            f.write(output)

        
        ## Saving intermediate target halo id for next source halo id in daisy-chain -> only using most massive halo
        source_int_halo_ids.append(target_int_halo_ids[0])
        curr_source_halo_id = target_int_halo_ids[0]

        print('\n\n')

    print()
    print()

print('done')
