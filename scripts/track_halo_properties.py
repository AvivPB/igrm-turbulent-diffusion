## module load NiaEnv/2022a python/3.11.5
## Python evironment: gen_profiles

import sys
import os
import argparse
import dill
import gc
import copy
from timeit import default_timer as timer

import yt
import unyt
import caesar
import numpy as np

import pprint

gc.isenabled()


def save_object_with_dill(obj, filename, mode='rb'):
    with open(filename, mode) as f:  # mode='wb' overwrites any existing file.
        dill.dump(obj, f, dill.HIGHEST_PROTOCOL)

def euclidean_distance(a, b):
    assert np.shape(a) == np.shape(b), f'Shapes of a and b are different'
    return(np.sqrt(np.sum((a-b)**2, axis=np.ndim(a)-1, keepdims=True)))


parser = argparse.ArgumentParser(prog='track_halo_properties.py', description='Track properties of halo and central galaxy across snapshots using progenitors/descendants.')
parser.add_argument('--snap_dir', action='store', type=str, required=True, 
                    help='directory containing snapshots')
parser.add_argument('--snap_base', action='store', type=str, default='snapshot_',
                    help='base name for snapshots, e.g. snapshot_')
parser.add_argument('--caesar_dir', action='store', type=str, required=True, 
                    help='directory containing caesar files')
parser.add_argument('--caesar_base', action='store', type=str, default='caesar_',
                    help='base name for caesar files, e.g. caesar_')
parser.add_argument('--caesar_suffix', action='store', type=str, default='',
                    help='suffix for caesar files, e.g. _haloid-fof_lowres-[2]')
parser.add_argument('--source_snap_num', action='store', type=int, required=True, 
                    help='Snapshot number for halo of which to find progenitor/descendant properties')
parser.add_argument('--target_snap_nums', action='store', nargs='*', type=int, required=True, 
                    help='Snapshot numbers in which to find halo progenitors/descendants')
parser.add_argument('--source_halo_id', action='store', type=int, required=True, 
                    help='Id of source halo')
# parser.add_argument('--n_most', action='store', type=int, default=1, choices=[None, 1, 2],
#                     help='caesar progen n_most option; find n_most progenitors/descendents (None = all)')
parser.add_argument('--nproc', action='store', type=int, default=1,
                    help='caesar progen nproc option')

parser.add_argument('--output_file', action='store', type=str, required=True,
                    help='Full path of output file')
parser.add_argument('--clear_output_file', action=argparse.BooleanOptionalAction, default=False, 
                    help='Whether to clear the output file initially before writing to it')
args = parser.parse_args()

# print(args.n_most)



## Add all properties to dict, to save in pickle file
# prop_dict = {}
# for prop_name in prop_names:
#     prop_dict[prop_name] = []
# prop_dict = {
#     'halo':{},
#     'central_halo':{},
#     'central':{},
#     'halo_central':{},
# }

def first_time(d, key, init=[]):
    # print('start')
    if key not in list(d.keys()):
        d[key] = copy.deepcopy(init)
        # print('finish')

# def append_value(d, val, replace_val=0, replace_units=''):
#     try:
#         d.append(val)
#     except:
#         d.append(

# init_dict = {'init':None}
init_dict = {}

if not os.path.exists(args.output_file):
    print('Making output file path')
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    save_object_with_dill(init_dict, args.output_file)
    # f = open(args.output_file, 'w')
    # f.close()
    print()

if args.clear_output_file:
    print('Clearing output file')
    save_object_with_dill(init_dict, args.output_file, mode='wb')
    # f = open(args.output_file, 'w')
    # f.close()
    print()

with open(args.output_file, 'rb') as f:
    prop_dict = dill.load(f)

# if not os.path.exists(f'{args.output_file}.header'):
#     print('Making output file header')
#     f = open(f'{args.output_file}.header', 'w')
#     f.close()
#     print()

# print('Writing to output file header')
# # header = 'caesar_dir\tsource_snap_num\tsource_z\tsource_halo_id\tsource_m500c (Msun)\tsource_contamination\ttarget_snap_num\ttarget_z\ttarget_halo_id\ttarget_m500c (Msun)\ttarget_contamination'
# header = 'caesar_dir\tsource_snap_num\tsource_z\tsource_halo_id\tsource_m500c (Msun)\tsource_contamination\ttarget_snap_num\ttarget_z'
# for n in range(args.n_most):
#     header += f'\ttarget_halo_id-{n+1}\ttarget_m500c-{n+1} (Msun)\ttarget_contamination-{n+1}'
# with open(f'{args.output_file}.header', 'w') as f:
#     f.write(header)
# print()



###########################################################################################
## Set up halo/galaxy properties to save ##################################################

## Sphere for checking mass ratios of halos with target halo
sphere_radius_type = 'r500c'
sphere_radius_factor = 10.
mass_ratio_type = 'm500c'
major_merger_mass_ratio = 5.

halo_types = ['halo', 'central_halo']
central_types = ['central', 'halo_central']


# prop_names = []

delta_values = ['2500', '500', '200']
virial_quantities = ['circular_velocity', 'spin_param', 'temperature']
halo_sfr_types = ['', '_100']
halo_mass_types = ['gas', 'stellar', 'dm', 'dust', 'H2']
halo_radii_types = ['gas', 'stellar', 'dm', 'baryon', 'total']
halo_radii_XX = ['half_mass', 'r20', 'r80']
halo_metallicity_types = ['mass_weighted', 'sfr_weighted', 'stellar', 'mass_weighted_cgm', 'temp_weighted_cgm']
halo_velocity_dispersion_types = ['gas', 'stellar', 'dm', 'baryon', 'total']
# halo_rotation_types = ['gas', 'stellar', 'dm', 'baryon', 'total']
# halo_rotation_XX = ['L', 'ALPHA', 'BETA', 'BoverT', 'kappa_rot']
halo_age_types = ['mass_weighted', 'metal_weighted']
halo_temperature_types = ['mass_weighted', 'mass_weighted_cgm']#, 'temp_weighted_cgm']
halo_local_density_types = ['300', '1000', '3000']

# for halo_type in halo_types:
#     prop_names.append(f'snap_num_{halo_type}')
#     prop_names.append(f'age_{halo_type}')
#     prop_names.append(f'z_{halo_type}')
#     prop_names.append(f'id_{halo_type}')
#     prop_names.append(f'contamination_{halo_type}')
#     prop_names.append(f'minpotpos_{halo_type}')
#     prop_names.append(f'num_major_mergers_{halo_type}')
#     # prop_names.append(f'bh_mdot_{halo_type}')
#     # prop_names.append(f'bh_fedd_{halo_type}')
#     # prop_names.append(f'bh_mdot_edd_{halo_type}')

#     for delta_value in delta_values:
#         prop_names.append(f'm{delta_value}c_{halo_type}')
#         prop_names.append(f'r{delta_value}c_{halo_type}')

#     for quant in virial_quantities:
#         prop_names.append(f'{quant}_{halo_type}')
    
#     for sfr_type in halo_sfr_types:
#         prop_names.append(f'sfr{sfr_type}_{halo_type}')
#         prop_names.append(f'ssfr{sfr_type}_{halo_type}')
#         # for central_type in central_types:
#         #     prop_names.append(f'sfr{sfr_type}_central{central_type}')
#         #     prop_names.append(f'ssfr{sfr_type}_central{central_type}')
    
#     for halo_mass_type in halo_mass_types:
#         prop_names.append(f'{halo_mass_type}_mass_{halo_type}')
    
#     for halo_radii_type in halo_radii_types:
#         for XX in halo_radii_XX:
#             prop_names.append(f'{halo_radii_type}_{XX}_radius_{halo_type}')
    
#     for halo_metal_type in halo_metallicity_types:
#         prop_names.append(f'{halo_metal_type}_metallicity_{halo_type}')
    
#     for halo_vel_disp_type in halo_velocity_dispersion_types:
#         prop_names.append(f'{halo_vel_disp_type}_velocity_dispersion_{halo_type}')
    
#     # for rot_type in halo_rotation_types:
#     #     for rot_XX in halo_rotation_XX:
#     #         prop_names.append(f'{rot_type}_{rot_XX}_rotation_{halo_type}')
    
#     for age_type in halo_age_types:
#         prop_names.append(f'{age_type}_stellar_age_{halo_type}')
    
#     for temp_type in halo_temperature_types:
#         prop_names.append(f'{temp_type}_temperature_{halo_type}')
    
#     for dens_type in halo_local_density_types:
#         prop_names.append(f'local_mass_density_{dens_type}kpccm_{halo_type}')
#         prop_names.append(f'local_number_density_{dens_type}kpccm_{halo_type}')


central_mass_types = ['gas', 'stellar', 'bh', 'dust', 'HI', 'H2']#'dm',
central_mass_apertures = ['', '_30kpc']
central_radii_types = ['gas', 'stellar', 'baryon', 'total']#'dm',
central_radii_XX = ['half_mass', 'r20', 'r80']
central_sfr_types = ['', '_100']
central_metallicity_types = ['mass_weighted', 'sfr_weighted', 'stellar']
central_velocity_dispersion_types = ['gas', 'stellar', 'baryon', 'total']#'dm',
central_age_types = ['mass_weighted', 'metal_weighted']
central_temperature_types = ['mass_weighted', 'mass_weighted_cgm']#, 'temp_weighted_cgm']
# central_rotation_types = ['gas', 'stellar', 'dm', 'baryon', 'total']
# central_rotation_XX = ['L', 'ALPHA', 'BETA', 'BoverT', 'kappa_rot']

# for central_type in central_types:
#     prop_names.append(f'snap_num_{central_type}')
#     prop_names.append(f'age_{central_type}')
#     prop_names.append(f'z_{central_type}')
#     prop_names.append(f'id_{central_type}')
#     prop_names.append(f'minpotpos_{central_type}')
#     prop_names.append(f'bh_mdot_{central_type}')
#     prop_names.append(f'bh_fedd_{central_type}')
#     prop_names.append(f'bh_mdot_edd_{central_type}')
    
#     for central_mass_type in central_mass_types:
#         for aperture in central_mass_apertures:
#             if central_mass_type in ['stellar', 'dust'] and aperture=='_30kpc': continue
#             prop_names.append(f'{central_mass_type}{aperture}_mass_{central_type}')

#     for central_radii_type in central_radii_types:
#         for XX in central_radii_XX:
#             prop_names.append(f'{central_radii_type}_{XX}_radius_{central_type}')

#     for sfr_type in central_sfr_types:
#         prop_names.append(f'sfr{sfr_type}_{central_type}')
#         prop_names.append(f'ssfr{sfr_type}_{central_type}')

#     for central_metal_type in central_metallicity_types:
#         prop_names.append(f'{central_metal_type}_metallicity_{central_type}')
    
#     for central_vel_disp_type in central_velocity_dispersion_types:
#         prop_names.append(f'{central_vel_disp_type}_velocity_dispersion_{central_type}')
    
#     for age_type in central_age_types:
#         prop_names.append(f'{age_type}_stellar_age_{central_type}')
    
#     for temp_type in central_temperature_types:
#         prop_names.append(f'{temp_type}_temperature_{central_type}')

#     # for rot_type in central_rotation_types:
#     #     for rot_XX in central_rotation_XX:
#     #         prop_names.append(f'{rot_type}_{rot_XX}_rotation_{central_type}')


###########################################################################################



## Load source snapshot
source_snap_file = os.path.join(args.snap_dir, f'{args.snap_base}{args.source_snap_num:03}.hdf5')
# try:
source_snap = yt.load(source_snap_file)
z_source = source_snap.current_redshift
# except Exception as error:
#     z_source = -1
#     print(f'Error occurred loading source snapshot: {error}')

## Load source caesar file
source_caesar_file = os.path.join(args.caesar_dir, f'{args.caesar_base}{args.source_snap_num:03}{args.caesar_suffix}.hdf5')
# try:
source_obj = caesar.load(source_caesar_file)
# except Exception as error:
#     print(f'Error occurred loading source caesar file: {error}')
#     print()
#     continue
print(f'Source caesar file: {source_caesar_file}, z={z_source}\n')

print(f'Source halo id: {args.source_halo_id}')
source_halo = source_obj.halos[args.source_halo_id]

try:
    source_central = source_halo.central_galaxy
    source_central_id = source_central.GroupID
except:
    source_central = None
    source_central_id = -1
print(f'Source halo central galaxy id: {source_central_id}\n\n')


# with open(args.output_file, 'rb') as f:
#     prop_dict = dill.load(f)
    
## Loop through target caesar files
for target_snap_num in args.target_snap_nums:
    print(f'Target snap num: {target_snap_num}')
    
    ## Load target snapshot
    target_snap_file = os.path.join(args.snap_dir, f'{args.snap_base}{target_snap_num:03}.hdf5')
    try:
        target_snap = yt.load(target_snap_file)
        z_target = target_snap.current_redshift
    except Exception as error:
        z_target = -1
        print(f'Error occurred loading target snapshot: {error}')
        print()
        
    ## Load target caesar file
    target_caesar_file = os.path.join(args.caesar_dir, f'{args.caesar_base}{target_snap_num:03}{args.caesar_suffix}.hdf5')
    try:
        target_obj = caesar.load(target_caesar_file)
    except Exception as error:
        print(f'Error occurred loading target caesar file: {error}')
        print()
        continue
    print(f'Target caesar file: {target_caesar_file}, z={z_target}\n')

    
    ## Link halos in snapshots with caesar progen
    # caesar.progen.check_if_progen_is_present(target_caesar_file, 'progen_halo_dm')
    halo_progens = caesar.progen.progen_finder(obj_current=source_obj, obj_target=target_obj, 
                                          caesar_file=source_caesar_file, snap_dir=args.snap_dir,
                                          data_type='halo', part_type='dm', recompute=True,
                                          save=False, n_most=1, min_in_common=0.1, nproc=args.nproc,
                                          match_frac=True, reverse_match=False)
    print('\nHalo progens:')
    print(halo_progens)
    print()
    
    ## For n_most=1
    target_halo_id = halo_progens[0][args.source_halo_id][0]  # with match_frac=True
    # target_halo_id = progens[source_halo_id][0]  # with match_frac=False
    print(f'Target halo id: {target_halo_id}')
    if target_halo_id < 0:
        target_halo = None
    else:
        target_halo = target_obj.halos[target_halo_id]
        print(f"Target halo m500c: {target_halo.virial_quantities['m500c']}")
        print(f'Target halo contamination: {target_halo.contamination}\n')
        first_time(prop_dict, 'halo', init={})

    try:
        target_halo_central = target_halo.central_galaxy
        target_halo_central_id = target_halo_central.GroupID
        first_time(prop_dict, 'halo_central', init={})
    except:
        target_halo_central = None
        target_halo_central_id = -1
    # if target_central is not None:
    #     target_central_index = target_central.GroupID
    print(f'Target halo central galaxy id: {target_halo_central_id}\n')
    
    ## For any n_most
    # target_halo_ids = progens[0][source_halo_id]#[0]  # with match_frac=True
    # # target_halo_ids = progens[source_halo_id]#[0]  # with match_frac=False
    # target_halos = [target_obj.halos[target_halo_id] for target_halo_id in target_halo_ids]
    # target_halo_m500c = [target_halo.virial_quantities['m500c'] for target_halo in target_halos]
    # target_halo_contamination = [target_halo.contamination for target_halo in target_halos]
    # print(f'Target halo ids: {target_halo_ids}')
    # print(f'Target halo m500c: {target_halo_m500c}')
    # print(f'Target halo contamination: {target_halo_contamination}')
    # print()



    ## Link galaxies in snapshots with caesar progen
    # caesar.progen.check_if_progen_is_present(target_caesar_file, 'progen_halo_dm')
    if source_central is not None:
        try:
            gal_progens = caesar.progen.progen_finder(obj_current=source_obj, obj_target=target_obj, 
                                                  caesar_file=source_caesar_file, snap_dir=args.snap_dir,
                                                  data_type='galaxy', part_type='star', recompute=True,
                                                  save=False, n_most=1, min_in_common=0.1, nproc=args.nproc,
                                                  match_frac=True, reverse_match=False)
            print('\nGalaxy progens:')
            print(gal_progens)
            print()
            
            ## For n_most=1
            target_central_id = gal_progens[0][source_central_id][0]  # with match_frac=True
            # target_halo_id = progens[source_halo_id][0]  # with match_frac=False
            if target_central_id < 0:
                target_central = None
            else:
                target_central = target_obj.galaxies[target_central_id]
                # print(f'Target central id: {target_central_id}\n')
                first_time(prop_dict, 'central', init={})
        
            try:
                target_central_halo = target_central.halo
                target_central_halo_id = target_central_halo.GroupID
                first_time(prop_dict, 'central_halo', init={})
            except:
                target_central_halo = None
                target_central_halo_id = -1
            # print(f'Target central galaxy halo id: {target_central_halo_id}')
            # if target_central_halo is not None:
            #     print(f"Target central galaxy halo m500c: {target_central_halo.virial_quantities['m500c']}")
            #     print(f'Target central galaxy halo contamination: {target_central_halo.contamination}\n')
        except:
            gal_progens = None
            
            target_central_id = -1
            target_central = None
            
            target_central_halo = None
            target_central_halo_id = -1

    else:
        gal_progens = None
        
        target_central_id = -1
        target_central = None
        
        target_central_halo = None
        target_central_halo_id = -1

    print(f'Target central id: {target_central_id}')
    print(f'Target central galaxy halo id: {target_central_halo_id}\n')
    if target_central_halo is not None:
        print(f"Target central galaxy halo m500c: {target_central_halo.virial_quantities['m500c']}")
        print(f'Target central galaxy halo contamination: {target_central_halo.contamination}\n')



    # print()
    # pprint.pprint(prop_dict)
    # print()


    ## Get properties and save them to dict ########
    print('\nCalculating Properties\n')
    
    ## Halo properties
    for halo_type, halo in zip(halo_types, [target_halo, target_central_halo]):
        if halo is None:
            continue

        print(f'\n{halo_type}\n')

        # print()
        # pprint.pprint(prop_dict)
        # print()
        first_time(prop_dict[halo_type], 'snap_num')
        if target_snap_num in prop_dict[halo_type]['snap_num']:
            print(f'\nsnap_num {target_snap_num} for {halo_type} already saved\n')
            continue
        # print()
        # pprint.pprint(prop_dict)
        # print()
        prop_dict[halo_type]['snap_num'].append(unyt.unyt_array(target_snap_num, ''))
        # print()
        # pprint.pprint(prop_dict)
        # print()

        first_time(prop_dict[halo_type], 'age')
        # print()
        # pprint.pprint(prop_dict)
        # print()
        prop_dict[halo_type]['age'].append(target_snap.current_time.in_units('Gyr'))
        # print()
        # pprint.pprint(prop_dict)
        # print()

        # sys.exit()

        first_time(prop_dict[halo_type], 'z')
        prop_dict[halo_type]['z'].append(unyt.unyt_array(target_snap.current_redshift, ''))

        first_time(prop_dict[halo_type], 'id')
        prop_dict[halo_type]['id'].append(unyt.unyt_array(halo.GroupID, ''))

        first_time(prop_dict[halo_type], 'contamination')
        prop_dict[halo_type]['contamination'].append(unyt.unyt_array(halo.contamination, ''))

        first_time(prop_dict[halo_type], 'minpotpos')
        prop_dict[halo_type]['minpotpos'].append(halo.minpotpos.in_units('Mpccm'))

        for delta_value in delta_values:
            first_time(prop_dict[halo_type], f'm{delta_value}c')
            prop_dict[halo_type][f'm{delta_value}c'].append(halo.virial_quantities[f'm{delta_value}c'])

            first_time(prop_dict[halo_type], f'r{delta_value}c')
            prop_dict[halo_type][f'r{delta_value}c'].append(halo.virial_quantities[f'r{delta_value}c'])

        for quant in virial_quantities:
            first_time(prop_dict[halo_type], f'{quant}')
            prop_dict[halo_type][f'{quant}'].append(halo.virial_quantities[f'{quant}'])

        first_time(prop_dict[halo_type], 'sfr')
        try:
            prop_dict[halo_type]['sfr'].append(halo.sfr.in_units('Msun/yr'))
        except:
            print(f'Bad {halo_type} sfr')
            prop_dict[halo_type]['sfr'].append(unyt.unyt_array(0, 'Msun/yr'))

        first_time(prop_dict[halo_type], 'sfr_100')
        try:
            prop_dict[halo_type]['sfr_100'].append(halo.sfr_100.in_units('Msun/yr'))
        except:
            print(f'Bad {halo_type} sfr_100')
            prop_dict[halo_type]['sfr_100'].append(unyt.unyt_array(0, 'Msun/yr'))

        for mass_type in halo_mass_types:
            first_time(prop_dict[halo_type], f'{mass_type}_mass')
            try:
                prop_dict[halo_type][f'{mass_type}_mass'].append(halo.masses[mass_type].in_units('Msun'))
            except:
                print(f'Bad {halo_type} {mass_type}_mass')
                prop_dict[halo_type][f'{mass_type}_mass'].append(unyt.unyt_array(0, 'Msun'))

        for radii_type in halo_radii_types:
            for XX in halo_radii_XX:
                first_time(prop_dict[halo_type], f'{radii_type}_{XX}_radius')
                try:
                    prop_dict[halo_type][f'{radii_type}_{XX}_radius'].append(halo.radii[f'{radii_type}_{XX}'].in_units('kpc'))
                except:
                    print(f'Bad {halo_type} {radii_type}_{XX}_radius')
                    prop_dict[halo_type][f'{radii_type}_{XX}_radius'].append(unyt.unyt_array(np.nan, 'kpc'))

        for metal_type in halo_metallicity_types:
            first_time(prop_dict[halo_type], f'{metal_type}_metallicity')
            try:
                prop_dict[halo_type][f'{metal_type}_metallicity'].append(halo.metallicities[metal_type])
            except:
                print(f'Bad {halo_type} {metal_type}_metallicity')
                prop_dict[halo_type][f'{metal_type}_metallicity'].append(unyt.unyt_array(np.nan, ''))

        for vel_disp_type in halo_velocity_dispersion_types:
            first_time(prop_dict[halo_type], f'{vel_disp_type}_velocity_dispersion')
            try:
                prop_dict[halo_type][f'{vel_disp_type}_velocity_dispersion'].append(halo.velocity_dispersions[vel_disp_type].in_units('km/s'))
            except:
                print(f'Bad {halo_type} {vel_disp_type}_velocity_dispersion')
                prop_dict[halo_type][f'{vel_disp_type}_velocity_dispersion'].append(unyt.unyt_array(np.nan, 'km/s'))

        for age_type in halo_age_types:
            first_time(prop_dict[halo_type], f'{age_type}_stellar_age')
            try:
                prop_dict[halo_type][f'{age_type}_stellar_age'].append(halo.ages[age_type].in_units('Gyr'))
            except:
                print(f'Bad {halo_type} {age_type}_stellar_age')
                prop_dict[halo_type][f'{age_type}_stellar_age'].append(unyt.unyt_array(np.nan, 'Gyr'))

        for temp_type in halo_temperature_types:
            first_time(prop_dict[halo_type], f'{temp_type}_temperature')
            try:
                prop_dict[halo_type][f'{temp_type}_temperature'].append(halo.temperatures[temp_type].in_units('K'))
            except:
                print(f'Bad {halo_type} {temp_type}_temperature')
                prop_dict[halo_type][f'{temp_type}_temperature'].append(unyt.unyt_array(np.nan, 'K'))

        for dens_type in halo_local_density_types:
            first_time(prop_dict[halo_type], f'local_mass_density_{dens_type}kpccm')
            try:
                prop_dict[halo_type][f'local_mass_density_{dens_type}kpccm'].append(halo.local_mass_density[dens_type].in_units('Msun/kpccm**3'))
            except:
                print(f'Bad {halo_type} local_mass_density_{dens_type}kpccm')
                prop_dict[halo_type][f'local_mass_density_{dens_type}kpccm'].append(unyt.unyt_array(np.nan, 'Msun/kpccm**3'))

            first_time(prop_dict[halo_type], f'local_number_density_{dens_type}kpccm')
            try:
                prop_dict[halo_type][f'local_number_density_{dens_type}kpccm'].append(halo.local_number_density[dens_type].in_units('kpccm**-3'))
            except:
                print(f'Bad {halo_type} local_number_density_{dens_type}kpccm')
                prop_dict[halo_type][f'local_v_density_{dens_type}kpccm'].append(unyt.unyt_array(np.nan, 'kpccm**-3'))


        ############# Check for nearby halos that qualify as major mergers ###########################
        ## Make sphere around target halo
        sphere_radius = sphere_radius_factor*halo.virial_quantities[sphere_radius_type]
        # sphere = target_snap.sphere(halo.minpotpos, sphere_radius)
    
        ## Find all halos whose centres are within sphere
        halo_minpotpos = unyt.unyt_array([_halo.minpotpos.in_units('Mpc') for _halo in target_obj.halos])
        distance_from_target_halo = unyt.unyt_array(np.zeros(len(halo_minpotpos)), halo_minpotpos.units)
        for ii in range(len(halo_minpotpos)):
            _minpotpos = halo_minpotpos[ii]
            distance_from_target_halo[ii] = euclidean_distance(_minpotpos, halo.minpotpos.in_units('Mpc'))
        halo_within_sphere_index = np.nonzero(distance_from_target_halo <= sphere_radius)[0]
        halo_within_sphere_index = np.setdiff1d(halo_within_sphere_index, halo.GroupID) # Remove id of target halo
    
        ## Check for major merger mass ratio
        target_halo_m500c = halo.virial_quantities[mass_ratio_type]
        halo_m500c = unyt.unyt_array([_halo.virial_quantities[mass_ratio_type] for _halo in target_obj.halos])
        halos_within_sphere_m500c = halo_m500c[halo_within_sphere_index]
        halo_m500c_ratio = []
        for halo_within_sphere_m500c in halos_within_sphere_m500c:
            if target_halo_m500c >= halo_within_sphere_m500c:
                halo_m500c_ratio.append(target_halo_m500c/halo_within_sphere_m500c)
            else:
                halo_m500c_ratio.append(halo_within_sphere_m500c/target_halo_m500c)
        halo_m500c_ratio = np.array(halo_m500c_ratio)
        major_merger_indexes = np.where(halo_m500c_ratio <= major_merger_mass_ratio)[0]
        num_major_mergers = unyt.unyt_array(len(major_merger_indexes), '')

        first_time(prop_dict[halo_type], 'num_major_mergers')
        prop_dict[halo_type]['num_major_mergers'].append(num_major_mergers)
    
        print()
    
        ## dynamically deallocate variables to free up memory
        del halo_m500c, halos_within_sphere_m500c, halo_m500c_ratio, major_merger_indexes #, sphere
        gc.collect()




    ## Central galaxy properties
    for central_type, central in zip(central_types, [target_central, target_halo_central]):
        if central is None:
            continue

        print(f'\n{central_type}\n')

        first_time(prop_dict[central_type], 'snap_num')
        if target_snap_num in prop_dict[central_type]['snap_num']:
            print(f'\nsnap_num {target_snap_num} for {central_type} already saved\n')
            continue
        prop_dict[central_type]['snap_num'].append(unyt.unyt_array(target_snap_num, ''))

        first_time(prop_dict[central_type], 'age')
        prop_dict[central_type]['age'].append(target_snap.current_time.in_units('Gyr'))

        first_time(prop_dict[central_type], 'z')
        prop_dict[central_type]['z'].append(unyt.unyt_array(target_snap.current_redshift, ''))

        first_time(prop_dict[central_type], 'id')
        prop_dict[central_type]['id'].append(unyt.unyt_array(central.GroupID, ''))

        first_time(prop_dict[central_type], 'minpotpos')
        prop_dict[central_type]['minpotpos'].append(central.minpotpos.in_units('Mpccm'))

        first_time(prop_dict[central_type], 'bh_mdot')
        try:
            prop_dict[central_type]['bh_mdot'].append(central.bhmdot.in_units('Msun/yr'))
        except:
            print(f'Bad {central_type} bh_mdot')
            prop_dict[central_type]['bh_mdot'].append(unyt.unyt_array(np.nan, 'Msun/yr'))

        first_time(prop_dict[central_type], 'bh_fedd')
        try:
            prop_dict[central_type]['bh_fedd'].append(central.bh_fedd.in_units(''))
        except:
            print(f'Bad {central_type} bh_fedd')
            prop_dict[central_type]['bh_fedd'].append(unyt.unyt_array(np.nan, ''))

        # first_time(prop_dict[central_type], 'bh_mdot_edd')
        # prop_dict[central_type]['bh_mdot_edd'].append(unyt.unyt_array(prop_dict[central_type]['bh_mdot'])/unyt.unyt_array(prop_dict[central_type]['bh_fedd']))

        first_time(prop_dict[central_type], 'sfr')
        try:
            prop_dict[central_type]['sfr'].append(central.sfr.in_units('Msun/yr'))
        except:
            print(f'Bad {central_type} sfr')
            prop_dict[central_type]['sfr'].append(unyt.unyt_array(0, 'Msun/yr'))

        first_time(prop_dict[central_type], 'sfr_100')
        try:
            prop_dict[central_type]['sfr_100'].append(central.sfr_100.in_units('Msun/yr'))
        except:
            print(f'Bad {central_type} sfr_100')
            prop_dict[central_type]['sfr_100'].append(unyt.unyt_array(0, 'Msun/yr'))

        for mass_type in central_mass_types:
            for aperture in central_mass_apertures:
                if mass_type in ['stellar', 'dust'] and aperture=='_30kpc': continue
                first_time(prop_dict[central_type], f'{mass_type}{aperture}_mass')
                try:
                    prop_dict[central_type][f'{mass_type}{aperture}_mass'].append(central.masses[f'{mass_type}{aperture}'].in_units('Msun'))
                except:
                    print(f'Bad {central_type} {mass_type}{aperture}_mass')
                    prop_dict[central_type][f'{mass_type}{aperture}_mass'].append(unyt.unyt_array(0, 'Msun'))

        for radii_type in central_radii_types:
            for XX in central_radii_XX:
                first_time(prop_dict[central_type], f'{radii_type}_{XX}_radius')
                try:
                    prop_dict[central_type][f'{radii_type}_{XX}_radius'].append(central.radii[f'{radii_type}_{XX}'].in_units('kpc'))
                except:
                    print(f'Bad {central_type} {radii_type}_{XX}_radius')
                    prop_dict[central_type][f'{radii_type}_{XX}_radius'].append(unyt.unyt_array(np.nan, 'kpc'))

        for metal_type in central_metallicity_types:
            first_time(prop_dict[central_type], f'{metal_type}_metallicity')
            try:
                prop_dict[central_type][f'{metal_type}_metallicity'].append(central.metallicities[metal_type])
            except:
                print(f'Bad {central_type} {metal_type}_metallicity')
                prop_dict[central_type][f'{metal_type}_metallicity'].append(unyt.unyt_array(np.nan, ''))

        for vel_disp_type in central_velocity_dispersion_types:
            first_time(prop_dict[central_type], f'{vel_disp_type}_velocity_dispersion')
            try:
                prop_dict[central_type][f'{vel_disp_type}_velocity_dispersion'].append(central.velocity_dispersions[vel_disp_type].in_units('km/s'))
            except:
                print(f'Bad {central_type} {vel_disp_type}_velocity_dispersion')
                prop_dict[central_type][f'{vel_disp_type}_velocity_dispersion'].append(unyt.unyt_array(np.nan, 'km/s'))

        for age_type in central_age_types:
            first_time(prop_dict[central_type], f'{age_type}_stellar_age')
            try:
                prop_dict[central_type][f'{age_type}_stellar_age'].append(central.ages[age_type].in_units('Gyr'))
            except:
                print(f'Bad {central_type} {age_type}_stellar_age')
                prop_dict[central_type][f'{age_type}_stellar_age'].append(unyt.unyt_array(np.nan, 'Gyr'))

        for temp_type in central_temperature_types:
            first_time(prop_dict[central_type], f'{temp_type}_temperature')
            try:
                prop_dict[central_type][f'{temp_type}_temperature'].append(central.temperatures[temp_type].in_units('K'))
            except:
                print(f'Bad {central_type} {temp_type}_temperature')
                prop_dict[central_type][f'{temp_type}_temperature'].append(unyt.unyt_array(np.nan, 'K'))


    ## dynamically deallocate variables to free up memory
    del target_snap, target_obj, halo_progens, gal_progens, target_halo, target_halo_central, target_central, target_central_halo
    gc.collect()



    # Save properties dictionary
    print('\nSaving properties dictionary to file')
    time_start = timer()
    save_object_with_dill(prop_dict, args.output_file, mode='wb+')
    # dill.dump(prop_dict, f, dill.HIGHEST_PROTOCOL)
    time_end = timer()
    print(args.output_file)
    print(f'Time to save file: {time_end-time_start} s')
    print('DONE\n')


print()
pprint.pprint(prop_dict)
print()


## A few extra properties that can be calculated from those already saved
for halo_type in halo_types:
    try:
        prop_dict[halo_type]['ssfr'] = unyt.unyt_array(prop_dict[halo_type]['sfr'])/unyt.unyt_array(prop_dict[halo_type]['stellar_mass'])
        prop_dict[halo_type]['ssfr_100'] = unyt.unyt_array(prop_dict[halo_type]['sfr_100'])/unyt.unyt_array(prop_dict[halo_type]['stellar_mass'])
    except Exception as error:
        print(f'Error calculating extra halo props: {error}')
        continue

for central_type in central_types:
    try:
        prop_dict[central_type]['ssfr'] = unyt.unyt_array(prop_dict[central_type]['sfr'])/unyt.unyt_array(prop_dict[central_type]['stellar_mass'])
        prop_dict[central_type]['ssfr_100'] = unyt.unyt_array(prop_dict[central_type]['sfr_100'])/unyt.unyt_array(prop_dict[central_type]['stellar_mass'])
        prop_dict[central_type]['bh_mdot_edd'] = unyt.unyt_array(prop_dict[central_type]['bh_mdot'])/unyt.unyt_array(prop_dict[central_type]['bh_fedd'])
    except Exception as error:
        print(f'Error calculating extra central props: {error}')
        continue
        
    


## Save properties dictionary
print('\nSaving properties dictionary to file')
save_object_with_dill(prop_dict, args.output_file, mode='wb+')
print(args.output_file)
print('DONE\n')