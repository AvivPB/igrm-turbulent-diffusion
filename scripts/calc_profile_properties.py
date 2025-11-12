## module load NiaEnv/2022a python/3.11.5
## Python evironment: gen_profiles

## Import libraries
import os
import copy
import argparse
import dill
import pprint

import numpy as np
import pynbody as pnb
import caesar

import astropy.units as u
import astropy.constants as astroc
from astropy.cosmology import FlatLambdaCDM

from astro_constants import NA_no_units, NA, kB_no_units, kB, mH, mp_no_units, mp, mu_e, mu, G_no_units, G
from solar_abundances import info_Asplund09, info_Lodders09, Z_tot_Asplund09, Z_tot_Lodders09

mp_xigrm = astroc.m_p.cgs.value # mass of proton in cgs units
kB_xigrm = astroc.k_B.cgs.value # Boltzmann constant in cgs units



## Command line keyword arguments
parser = argparse.ArgumentParser(description="Calculate more properties of profiles.")
parser.add_argument('--dir', action='store', type=str, required=True, 
                   help='Path to directory containing profiles file')
parser.add_argument('--name', action='store', type=str, required=True, 
                   help='Name of profiles file, not including .pkl extension')
parser.add_argument('--caesar_file', action='store', type=str, required=True, 
                   help='Full path to caesar file')
parser.add_argument('--suffix', action='store', type=str, required=False, default='-',
                   help='keywords to add to end of saved file name to make it recognizable')

parser.add_argument('--code', action='store', type=str, required=True, choices=['Simba', 'Simba-C'],
                   help='Simulation type (currently either just Simba or Simba-C')
# parser.add_argument('--size', action='store', type=str, required=True, 
#                    help='Simulation size, eg. N512L50 or N1024L100')
parser.add_argument('--ndim', action='store', type=int, required=True, choices=[2, 3],
                   help='Number of dimensions for profiles (2 or 3)')

## Choose which extra properties to compute
parser.add_argument('--calc_thermo_props', action=argparse.BooleanOptionalAction, default=False,
                    help='Calculate additional thermal profiles for each halo (default=False)')
parser.add_argument('--calc_metal_props', action=argparse.BooleanOptionalAction, default=False,
                    help='Calculate additional abundance/metallicity profiles for each halo (default=False)')
parser.add_argument('--calc_gradients', action=argparse.BooleanOptionalAction, default=False,
                    help='Calculate radial gradient profiles for each halo (default=False)')
parser.add_argument('--calc_log_props', action=argparse.BooleanOptionalAction, default=False,
                    help='Calculate logarithmic profiles for each halo (default=False)')

args = parser.parse_args()


## Save input parameters to output with results
parameters = {}
for arg, arg_val in vars(args).items():
    parameters[arg] = arg_val


## Input parameters as global variables
PROFILES_DIR = args.dir
PROFILES_NAME = args.name
CAESAR_FILE = args.caesar_file
SUFFIX = args.suffix

CODE = args.code
# SIZE = args.size
NDIM = args.ndim

CALC_THERMO_PROPS = args.calc_thermo_props
CALC_METAL_PROPS = args.calc_metal_props
CALC_GRADIENTS = args.calc_gradients
CALC_LOG_PROPS = args.calc_log_props



## Axis of all arrays of all properties along which the radii increase
RADIAL_AXIS = 0



## Set base units
R_UNITS = 'kpc'
M_UNITS = 'Msol'

R_UNITS_CAESAR = 'kpc'
M_UNITS_CAESAR = 'Msun'


## Simulation cosmological properties
COSMOLOGY_SIM = FlatLambdaCDM(name=r'Flat $\Lambda$CDM', H0=68 * u.km / u.s / u.Mpc, Om0=0.3)
h_sim = COSMOLOGY_SIM.h


## Indexes of different metals in Simba and Simba-C
metals_idx_simbac = {'H': 0, 'He': 1, 'Li': 2, 'Be': 3, 'B': 4, 'C': 5, 'N': 6, 'O': 7, 'F': 8, 'Ne': 9, 'Na': 10, 'Mg': 11, 'Al':12,
              'Si': 13, 'P': 14, 'S': 15, 'Cl': 16, 'Ar': 17, 'K': 18,'Ca': 19, 'Sc': 20, 'Ti': 21, 'V': 22, 'Cr': 23,
              'Mn': 24, 'Fe': 25, 'Co': 26, 'Ni': 27, 'Cu': 28, 'Zn': 29}
metals_idx_simba = {'H':0, 'He': 1, 'C': 2, 'N': 3, 'O': 4, 'Ne': 5, 'Mg': 6, 'Si': 7, 'S': 8, 'Ca': 9, 'Fe': 10}
metals_idx = {'Simba':metals_idx_simba, 'Simba-C':metals_idx_simbac}



## For getting correct units of scaling properties in 2d vs 3d
geometric_units = {
    ## Num dimensions
    2:{
        'volume':'cm**2',
        '1/volume':'cm**-2',
    },
    3:{
        'volume':'cm**3',
        '1/volume':'cm**-3',
    },
}



## Calculate temperature of hot gas
def temp(profile):
    '''
    Convert internal energy to temperature, following 
    the instructions from `GIZMO documentation <http://www.tapir.caltech.edu/~phopkins/Site/GIZMO_files/gizmo_documentation.html#snaps>`_
    '''
    #XH = 1-sim.gas['metals'][:,0]-sim.gas['metals'][:,1]
    GAMMA = 5.0/3
    # If ne = 0 gas is unionized
    mu = (4.0 / (3.0 * profile['Xh-bin'] + 4.0 * profile['Xh-bin'] * profile['fe'] + 1.0))#.in_units('1')
    print(f'\nmu: {mu} {mu.units}\n')
    # mu = 0.6
    result = pnb.array.SimArray(profile['u-bin'].in_units('cm**2 s**-2').view(np.ndarray) * (GAMMA - 1) * mu * (mp_xigrm / kB_xigrm))
    result.units = 'K'
    return result


    
## Load profiles file
def save_object_with_dill(obj, filename, protocol=dill.HIGHEST_PROTOCOL):
    with open(filename, 'wb') as f:  # Overwrites any existing file.
        dill.dump(obj, f, protocol)

profiles_file = os.path.join(PROFILES_DIR, PROFILES_NAME + '.pkl')
with open(profiles_file, 'rb') as f:
    profiles = dill.load(f)

## Load caesar file
obj = caesar.load(CAESAR_FILE)




## Function for calculating derivatives of profiles with units
def gradient(x, f, axis=None):
    ## Assume x and f are both pnb.array.SimArray
    return pnb.array.SimArray(np.gradient(f, x, axis=axis, edge_order=1), units=f.units/x.units)



## All x and y properties
part_type = list(profiles['halo_ids'].keys())[0]
halo_id = profiles['halo_ids'][part_type]['full'][0]
print()
print()
print(profiles['halo_profiles'][part_type][halo_id]['full']['x'].keys())
print()
print(profiles['halo_profiles'][part_type][halo_id]['full']['y'].keys())
# print(profiles['halo_profiles']['igrm'][profiles['halo_ids']['igrm']['full'][0]]['full']['y'].keys())
# print()
# print(profiles['halo_profiles']['all_particles'][profiles['halo_ids']['all_particles']['full'][0]]['full']['y'].keys())
print()
print()



## Xray bands to use
# xray_bands = ['0.5-1.8keV_total', '0.5-2.0keV_total', '0.1-2.4keV_total', '0.5-7.0keV_total', '0.5-10.0keV_total', 
#               '0.5-1.8keV_cont', '0.5-2.0keV_cont', '0.1-2.4keV_cont', '0.5-7.0keV_cont', '0.5-10.0keV_cont']
# xray_bands = ['0.5-10.0keV_total', '0.5-10.0keV_cont']
xray_bands = ['0.5-10.0keV_total']




## Properties to weight properly, sum properly, and metallicity types to compute

# weighted_props = ['Z', 'nX/nH', 'T', 'ne', 'K', 'P'] + ['tcool_' + xband for xband in xray_bands]
weighted_prop_weights = ['mass', 'volume'] + ['Lx_' + xband for xband in xray_bands]
# weighted_prop_weights = ['mass', 'volume'] + ['Lx_' + xband for xband in xray_bands]
weighted_prop_weights = [f'{weight}_weighted_mean' for weight in weighted_prop_weights]
weighted_prop_weights += ['median', 'mean']

sum_props = ['n', 'Ne', 'Nh', 'mass'] + ['Lx_' + xband for xband in xray_bands]
# avg_props = ['mass'] + ['Lx_' + xband for xband in xray_bands]

# metallicity_types = ['Z', 'nX/nH']
metallicity_types = ['Z']


## dicts for scaling properties by characteristic values, eg. M500
# mass_scaling_dict = {
#     'mass':{
#         'scale_prop':'M',
#         'scale_types':['500'],
#     },
#     'mass_enc':{
#         'scale_prop':'M',
#         'scale_types':['500'],
#     },
# }

thermo_scaling_dict = {}

for T_weight in ['spec', 'bin'] + weighted_prop_weights:
    if T_weight != 'T':
        thermo_scaling_dict['T-' + T_weight] = {
            'scale_prop':'T',
            'scale_types':['500ideal']
        }
        
# for kT_weight in ['median', 'spec'] + weighted_prop_weights:
#     if kT_weight != 'T':
#         thermo_scaling_dict['kT_' + kT_weight] = {
#             'scale_prop':'T',
#             'scale_types':['spec_corr', 'spec500']
#         }
        
# Can't use this for 2d profiles since ne500 will always be in units of cm**-3
# Anything that has ne in it
for ne_weight in ['bin'] + weighted_prop_weights:
    if ne_weight not in ['rho', 'ne', 'nh']:
        thermo_scaling_dict['ne-' + ne_weight] = {
            'scale_prop':'ne',
            'scale_types':['500ideal']
        }
        
for K_weight in ['bin'] + weighted_prop_weights + [weight_+'_bin_T' for weight_ in ['spec'] + weighted_prop_weights] + [weight_+'_bin_T_ne' for weight_ in ['spec'] + weighted_prop_weights]:
# for K_weight in ['median'] + weighted_prop_weights:
    if K_weight not in ['T_bin_T', 'T_bin_T_ne', 'spec_bin_T_ne', 'ne_bin_T_ne']:
        thermo_scaling_dict['K-' + K_weight] = {
            'scale_prop':'K',
            'scale_types':['500ideal']
        }
        
for P_weight in ['bin'] + weighted_prop_weights + [weight_+'_bin_T' for weight_ in ['spec'] + weighted_prop_weights] + [weight_+'_bin_T_ne' for weight_ in ['spec'] + weighted_prop_weights]:
# for P_weight in ['median'] + weighted_prop_weights:
    if P_weight not in ['T_bin_T', 'T_bin_T_ne', 'spec_bin_T_ne', 'ne_bin_T_ne']:
        thermo_scaling_dict['P-' + P_weight] = {
            'scale_prop':'P',
            'scale_types':['500ideal']
        }

        
        

# dict for scaling metallicities by solar abundances
metallicity_scaling_dict = {
    'asplund09':{
        'info':info_Asplund09,
        'Z_name':'ZX',
        'nX/nH_name':'NX_over_NH',
        'total':Z_tot_Asplund09,
    },
#     'lodders09':{
#         'info':info_Lodders09,
#         'Z_name':'ZX',
#         'nX/nH_name':'NX_over_NH',
#         'total':Z_tot_Lodders09,
#     },
}





## Compute all extra properties for each halo
print()
print('Computing extra profile properties')
print()

## need to make sure 3d and 2d profile values calculated before profiles for CC/NCC classification
# profiles_to_compute_names = ['three_d_profiles', 'two_d_profiles', 'profiles', 'res_profiles']
# profiles_to_compute = [three_d_profiles, two_d_profiles, profiles, res_profiles]
# profiles_ndims = [3, 2, NDIM, 3 if PROFILES_ARE_2D_BUT_RES_FILE_IS_3D else 2] ## last one won't always work
# compute_properties = [True, True, True, False]
# compute_metallicities = [False, False, True, False]

# particle_types = ['igrm', 'all_particles']
particle_types = list(profiles['halo_profiles'].keys())
print(f'\n{particle_types}\n')

# profiles_to_compute = ['core', 'big_core', 'full']
profiles_to_compute = ['full']
# profile_types_to_compute = ['core', 'big_core', 'full']
# profile_ids_to_compute = ['core', 'all_id']
compute_properties = [True]#, True, True]
# compute_metallicities = [False, False, True]
compute_metallicities = [True]

# profiles['halo_info'] = {}

for particle_type in particle_types:
    print()
    print()
    print(particle_type)
    print()

    for profile_to_compute, compute_props, compute_metals in zip(profiles_to_compute, compute_properties, compute_metallicities):
        # print()
        print()
        print(profile_to_compute)
        print()
    
        for halo_id in profiles['halo_ids'][particle_type][profile_to_compute]:
            print(f'Halo id {halo_id}')
    
            try:    
                x_profile = profiles['halo_profiles'][particle_type][halo_id][profile_to_compute]['x']
                profile = profiles['halo_profiles'][particle_type][halo_id][profile_to_compute]['y']
            except:
                print(profile_to_compute, 'does not exist for halo', halo_id)
                continue
    
            ## Do x quantity calculations
            xscale_values = {
                'R500':pnb.array.SimArray(obj.halos[halo_id].virial_quantities['r500c'].in_units(R_UNITS), units=R_UNITS),
                'R200':pnb.array.SimArray(obj.halos[halo_id].virial_quantities['r200c'].in_units(R_UNITS), units=R_UNITS),
                'Physical':pnb.array.SimArray(1, units=''),
            }
            xscale = profiles['xaxis'][profile_to_compute]['scale']
            xscale_value = xscale_values[xscale]
    
            ## I made it so that the radial bins that come with the individual halos
            ## are always physical (but this could be different!)
            # x_profile['physical_rbins'] = copy.deepcopy(x_profile['rbins'])
            # x_profile['rbins'] = x_profile['physical_rbins'] * xscale_values['Physical']/xscale_value
            x_profile['rbins_r500'] = x_profile['physical_rbins'].in_units(R_UNITS)/xscale_values['R500']
            x_profile['log_rbins_r500'] = np.log10(x_profile['rbins_r500'])
            x_profile['rbin_r500_centres'] = (x_profile['rbins_r500'][:-1] + x_profile['rbins_r500'][1:])/2.
            x_profile['log_rbin_r500_centres'] = (x_profile['log_rbins_r500'][:-1] + x_profile['log_rbins_r500'][1:])/2.
    
            ## don't think I need the below info ##
            # x_profile['log_rbins'] = np.log10(x_profile['rbins'])
            # x_profile['rbin_centres'] = (x_profile['rbins'][:-1] + x_profile['rbins'][1:])/2.
            # x_profile['log_rbin_centres'] = (x_profile['log_rbins'][:-1] + x_profile['log_rbins'][1:])/2.
    
            # x_profile['log_physical_rbins'] = np.log10(x_profile['physical_rbins'])
            # x_profile['physical_rbin_centres'] = (x_profile['physical_rbins'][:-1] + x_profile['physical_rbins'][1:])/2.
            # x_profile['log_physical_rbin_centres'] = (x_profile['log_physical_rbins'][:-1] + x_profile['log_physical_rbins'][1:])/2.
    
    
    
            ## Do y quantity calculations
    
            # Give number of particles a unit
            profile['n'] = pnb.array.SimArray(profile['n'].astype(float), units='')
            
            if compute_props:

                if particle_type.lower() == 'all_particles':
                    print(f'Computing {particle_type} properties\n')
                    
                    profile['volume_enc'] = pnb.array.SimArray(np.nancumsum(profile['volume']), units=profile['volume'].units)

                    for sum_prop in ['n', 'mass']:
                        profile[sum_prop + '_enc'] = pnb.array.SimArray(np.nancumsum(profile[sum_prop], axis=RADIAL_AXIS), units=profile[sum_prop].units)
                        profile[sum_prop + '_density'] = profile[sum_prop] / profile['volume']
                        profile[sum_prop + '_density_enc'] = profile[sum_prop + '_enc'] / profile['volume_enc']


                    ## Gravitational potential
                    profile['-GM/r_mid'] = -1. * G * profile['mass_enc'] / x_profile['physical_rbin_centres']
                    profile['-GM/r_mid^2'] = -1. * G * profile['mass_enc'] / x_profile['physical_rbin_centres']**2
                    profile['-GM/r_out'] = -1. * G * profile['mass_enc'] / x_profile['physical_rbins'][1:]
                    profile['-GM/r_out^2'] = -1. * G * profile['mass_enc'] / x_profile['physical_rbins'][1:]**2

                    profile['-GMrho/r_mid^2'] = -1. * G * profile['mass_enc'] * profile['mass_density'] / x_profile['physical_rbin_centres']**2
                    profile['-GMrho/r_out^2'] = -1. * G * profile['mass_enc'] * profile['mass_density'] / x_profile['physical_rbins'][1:]**2





                elif particle_type.lower() == 'dm':
                    print(f'Computing {particle_type} properties\n')
                    
                    profile['volume_enc'] = pnb.array.SimArray(np.nancumsum(profile['volume']), units=profile['volume'].units)

                    for sum_prop in ['n', 'mass']:
                        profile[sum_prop + '_enc'] = pnb.array.SimArray(np.nancumsum(profile[sum_prop], axis=RADIAL_AXIS), units=profile[sum_prop].units)
                        profile[sum_prop + '_density'] = profile[sum_prop] / profile['volume']
                        profile[sum_prop + '_density_enc'] = profile[sum_prop + '_enc'] / profile['volume_enc']

                    try:
                        profile['f_mass-total'] = profile['mass_enc'] / profiles['halo_profiles']['all_particles'][halo_id][profile_to_compute]['y']['mass_enc']
                    except:
                        print(f'\nCould not compute f_mass-total for halo {halo_id}, particle type {particle_type}, profile {profile_to_compute}\n')





                elif particle_type.lower() == 'bh':
                    print(f'Computing {particle_type} properties\n')
                    
                    profile['volume_enc'] = pnb.array.SimArray(np.nancumsum(profile['volume']), units=profile['volume'].units)

                    for sum_prop in ['n', 'mass', 'BH_Mass', 'BH_Mass_AlphaDisk']:
                        profile[sum_prop + '_enc'] = pnb.array.SimArray(np.nancumsum(profile[sum_prop], axis=RADIAL_AXIS), units=profile[sum_prop].units)
                        profile[sum_prop + '_density'] = profile[sum_prop] / profile['volume']
                        profile[sum_prop + '_density_enc'] = profile[sum_prop + '_enc'] / profile['volume_enc']

                    try:
                        profile['f_mass-total'] = profile['mass_enc'] / profiles['halo_profiles']['all_particles'][halo_id][profile_to_compute]['y']['mass_enc']
                    except:
                        print(f'\nCould not compute f_mass-total for halo {halo_id}, particle type {particle_type}, profile {profile_to_compute}\n')

                

                

                elif particle_type.lower() == 'gas' or particle_type.lower() == 'igrm':
                    print(f'Computing {particle_type} properties\n')
    
                    # ## Halo information for scaling/normalizing profiles
                    # halo_info = {
                    #     'M':{
                    #         '500':pnb.array.SimArray(obj.halos[halo_id].virial_quantities['m500c'].in_units(M_UNITS_CAESAR), units=M_UNITS),
                    #     },
                    #     'R':{
                    #         '500':pnb.array.SimArray(obj.halos[halo_id].virial_quantities['r500c'].in_units(R_UNITS_CAESAR), units=R_UNITS),
                    #     },
                    # }
        
                    # halo_info['T'] = {
                    #     '500ideal':(G * halo_info['M']['500'] * mu * mp / (kB * halo_info['R']['500'])).in_units('K'),
                    # }
                    # halo_info['ne'] = {
                    #     '500ideal':(500. * profiles['cosmo_props']['rho_gas_scaling'] / (mu_e * mp)).in_units('cm**-3'),
                    # }
                    # halo_info['K'] = {
                    #     '500ideal':(kB * halo_info['T']['500ideal'] * halo_info['ne']['500ideal']**(-2./3.)).in_units('keV cm**2'),
                    # }
                    # halo_info['P'] = {
                    #     '500ideal':(kB * halo_info['T']['500ideal'] * halo_info['ne']['500ideal']).in_units('keV cm**-3'),
                    # }
        
                    # profiles['halo_info'][halo_id] = halo_info
                    
                    
        
                    ## Always calculate some basic quantities
                    profile['volume_enc'] = pnb.array.SimArray(np.nancumsum(profile['volume']), units=profile['volume'].units)
        
        #                 print('volume units:', profile['volume'].units)
        
                    profile['ne-bin'] = (profile['Ne']/profile['volume']).in_units(geometric_units[NDIM]['1/volume'])
                    profile['nh-bin'] = (profile['Nh']/profile['volume']).in_units(geometric_units[NDIM]['1/volume'])
                    profile['fe'] = profile['Ne']/profile['Nh']
                    profile['Xh-bin'] = profile['H_mass']/profile['mass']
        
                    for sum_prop in sum_props:
                        profile[sum_prop + '_enc'] = pnb.array.SimArray(np.nancumsum(profile[sum_prop], axis=RADIAL_AXIS), units=profile[sum_prop].units)
                        profile[sum_prop + '_density'] = profile[sum_prop] / profile['volume']
                        profile[sum_prop + '_density_enc'] = profile[sum_prop + '_enc'] / profile['volume_enc']
        
                        ## Add extra axis so can be plotted with other metals as if it is an element,
                        ## by giving its element as H (ie. index 0)
                        profile[sum_prop + '_for_metals'] = profile[sum_prop][:,np.newaxis]
                        profile[sum_prop + '_enc_for_metals'] = profile[sum_prop + '_enc'][:,np.newaxis]
                        profile[sum_prop + '_density_for_metals'] = profile[sum_prop + '_density'][:,np.newaxis]
                        profile[sum_prop + '_density_enc_for_metals'] = profile[sum_prop + '_density_enc'][:,np.newaxis]
        
                    profile['mass_density_fb_rhocrit'] = profile['mass_density'] / profiles['cosmo_props']['rho_gas_scaling']
                    profile['mass_density_enc_fb_rhocrit'] = profile['mass_density_enc'] / profiles['cosmo_props']['rho_gas_scaling']
                    profile['mass_density_rhocrit'] = profile['mass_density'] / profiles['cosmo_props']['rho_crit']
                    profile['mass_density_enc_rhocrit'] = profile['mass_density_enc'] / profiles['cosmo_props']['rho_crit']

                    try:
                        profile['f_mass-total'] = profile['mass_enc'] / profiles['halo_profiles']['all_particles'][halo_id][profile_to_compute]['y']['mass_enc']
                    except:
                        print(f'\nCould not compute f_mass-total for halo {halo_id}, particle type {particle_type}, profile {profile_to_compute}\n')
        
                    ## Add extra axis so can be plotted with other metals as if it is an element, by giving its element as H (ie. index 0)
                    profile['mass_density_fb_rhocrit_for_metals'] = profile['mass_density_fb_rhocrit'][:,np.newaxis]
                    profile['mass_density_enc_fb_rhocrit_for_metals'] = profile['mass_density_enc_fb_rhocrit'][:,np.newaxis]
                    profile['mass_density_rhocrit_for_metals'] = profile['mass_density_rhocrit'][:,np.newaxis]
                    profile['mass_density_enc_rhocrit_for_metals'] = profile['mass_density_enc_rhocrit'][:,np.newaxis]
                    
                    
                    # for mean_prop in mean_props:
                    #     profile[mean_prop + '_mean'] = profile[mean_prop]/profile['n']
        
        
                    # Calculate normalized temperature, pressure and entropy profiles
                    if CALC_THERMO_PROPS:

                        profile['u-bin'] = profile['U']/profile['mass']

                        if False:
                            ## Energies and energies per unit mass/volume
                            # profile['u-bin'] = profile['U']/profile['mass']
                            profile['U_density-bin'] = profile['U']/profile['volume']

                            # profile['u_v1-bin'] = profile['U_v1']/profile['mass']
                            # profile['U_v1_density-bin'] = profile['U_v1']/profile['volume']

                            # profile['u_v2-bin'] = profile['U_v2']/profile['mass']
                            # profile['U_v2_density-bin'] = profile['U_v2']/profile['volume']

                            # profile['u_v3-bin'] = profile['U_v3']/profile['mass']
                            # profile['U_v3_density-bin'] = profile['U_v3']/profile['volume']

                            profile['KE_total_per_mass-bin'] = profile['KE_total']/profile['mass']
                            profile['KE_total_density-bin'] = profile['KE_total']/profile['volume']

                            profile['E-bin'] = profile['U'] + profile['KE_total']
                            profile['E_per_mass-bin'] = profile['E-bin']/profile['mass']
                            profile['E_density-bin'] = profile['E-bin']/profile['volume']

                            # profile['E_v1-bin'] = profile['U_v1'] + profile['KE_total']
                            # profile['E_v1_per_mass-bin'] = profile['E_v1-bin']/profile['mass']
                            # profile['E_v1_density-bin'] = profile['E_v1-bin']/profile['volume']

                            # profile['E_v2-bin'] = profile['U_v2'] + profile['KE_total']
                            # profile['E_v2_per_mass-bin'] = profile['E_v2-bin']/profile['mass']
                            # profile['E_v2_density-bin'] = profile['E_v2-bin']/profile['volume']

                            # profile['E_v3-bin'] = profile['U_v3'] + profile['KE_total']
                            # profile['E_v3_per_mass-bin'] = profile['E_v3-bin']/profile['mass']
                            # profile['E_v3_density-bin'] = profile['E_v3-bin']/profile['volume']


                        ## Calculate temperate directly from u-bin and Xh-bin
                        profile['T-bin'] = temp(profile)
                        profile['kT-bin'] = kB * profile['T-bin']
                        profile['K-bin'] = profile['kT-bin'] * profile['ne-bin']**(-2,3)
                        profile['P-bin'] = profile['kT-bin'] * profile['ne-bin']
                        for xband in xray_bands:
                            profile['tcool-bin'] = 6 * profile['ne-bin'] * profile['kT-bin'] / (2 * profile['fe'] * profile['Lx_'+ xband] / profile['volume'])

                        
        
                        ## Entropy and pressure profiles calculated directly from the temperature and ne profiles
                        for weight in weighted_prop_weights + ['spec']:
                            if weight[0] != 'T':
                                try:
                                    profile['kT-' + weight] = kB * profile['T-' + weight]
                                    
                                    profile['K-' + weight + '_bin_T'] = profile['kT-' + weight] * profile['ne-bin']**(-2,3)
                                    profile['P-' + weight + '_bin_T'] = profile['kT-' + weight] * profile['ne-bin']
        
                                    if weight not in ['ne', 'spec']:
                                        profile['K-' + weight + '_bin_T_ne'] = profile['kT-' + weight] * profile['ne-' + weight]**(-2,3) 
                                        profile['P-' + weight + '_bin_T_ne'] = profile['kT-' + weight] * profile['ne-' + weight]
                                        
                                except Exception as error:
                                    print('Failed computing kT and/or bin versions of thermal properties')
                                    print(f'Error: {error}\n')
                                    continue
        
        
        
                        ## Cooling time profiles calculated directly from the ne, T, Lx, volume, fe profiles
                        for weight in weighted_prop_weights + ['spec']:
                            if weight[0] != 'T':
                                try:
                                    for xband in xray_bands:
                                        profile['tcool_' + xband + '-' + weight + '_bin'] = 6 * profile['ne-bin'] * profile['kT-' + weight] / (2 * profile['fe'] * profile['Lx_'+ xband] / profile['volume'])
                                        
                                except Exception as error:
                                    print('Failed computing tcool properties')
                                    print(f'Error: {error}\n')
                                    continue
        
                        
                        ## Scaling properties by their relevant scaling factors
                        for prop, scaling_factors in thermo_scaling_dict.items():
                            scale_prop = scaling_factors['scale_prop']
                            scale_types = scaling_factors['scale_types']
        
                            for scale_type in scale_types:
                                scaled_prop_name = prop + '-' + scale_prop + scale_type
        
                                ## CHANGE THIS SO CAN SCALE BY VALUES THAT HAVE dIFFERENT UNITS THAN PROP
                                
                                # profile[scaled_prop_name] = profile[prop] / halo_info[scale_prop][scale_type]
                                # profile[scaled_prop_name] = profile[scaled_prop_name].simplify()
                                # simplified_units = profile[scaled_prop_name].units.simplify()
                                try:
                                    profile[scaled_prop_name] = profile[prop] / profiles['halo_info'][halo_id][scale_prop][scale_type].in_units(profile[prop].units)
                                    profile[scaled_prop_name].units = ''
        
                                    if scale_prop in ['T', 'K', 'P']:
                                        profile[scaled_prop_name+'-x2'] = 2*profile[scaled_prop_name]
                                except:
                                    profile[scaled_prop_name] = profile[prop] / profiles['halo_info'][halo_id][scale_prop][scale_type]
        
        
        
        
        
                    ## Calculate extra properties related to metallicity
                    if CALC_METAL_PROPS and compute_metals:
        
                        for metallicity_type in metallicity_types:
                            for weight in weighted_prop_weights:
                                
                                try:
        #                                 print()
        #                                 print()
        #                                 print(weight.upper() + ' start')
        #                                 print()
        #                                 print()
        
                                    ## Correct metallicity units from NoUnit to 1
                                    profile[metallicity_type + '-' + weight].units = ''
        
        
                                    ## Calculate metal ratios
                                    metal_profile = profile[metallicity_type + '-' + weight]
                                    Z_shape = np.shape(metal_profile)
                                    Z_num = np.repeat(np.expand_dims(metal_profile, axis=2), repeats=Z_shape[1], axis=2)
                                    Z_denom = np.swapaxes(Z_num, axis1=1, axis2=2)
                                    Z_ratio = Z_num/Z_denom
                                    Z_ratio = pnb.array.SimArray(Z_ratio, units='')
                                    profile[metallicity_type + '-' + weight + '-ratio'] = Z_ratio
        
                                    ## Check metal ratio calculations
                #                     Z_profile = profile[metallicity_type + '_' + weight]
                #                     Z_shape = np.shape(metal_profile)
                #                     n_rbins = Z_shape[0]
                #                     n_metals = Z_shape[1]
                #                     Z_ratio = pnb.array.SimArray(np.zeros((n_rbins, n_metals, n_metals)), units='')
                #                     for element_num in range(n_metals):
                #                         for element_denom in range(n_metals):
                #                             Z_ratio[:, element_num, element_denom] = Z_profile[:,element_num]/Z_profile[:,element_denom]
                #                     profile[metallicity_type + '_' + weight + '_ratio_v2'] = Z_ratio
        
        
                                    ## Metallicity and metallicity ratios relative to solar metallicity
                                    for normalization, norm_info in metallicity_scaling_dict.items():
                                        profile[metallicity_type+'-'+weight+'-solar_'+normalization] = copy.deepcopy(profile[metallicity_type+'-' +weight])
                                        profile[metallicity_type+'-'+weight+'-ratio_solar_'+normalization] = copy.deepcopy(profile[metallicity_type+'-' +weight+'-ratio'])
                #                         profile[metallicity_type+'_'+weight+'_ratio_v2_solar_'+normalization] = copy.deepcopy(profile[metallicity_type+'_' +weight+'_ratio_v2'])
        
                                        if metallicity_type.lower() == 'z' and weight not in ['median', 'mean']:
                                            profile['Ztot-' + weight + '-solar_' + normalization] = profile['Ztot-' + weight] / norm_info['total']
        
                                        for element, element_idx in metals_idx[CODE].items():
                                            profile[metallicity_type+'-'+weight+'-solar_'+normalization][:,element_idx] /= norm_info['info'][element][norm_info[metallicity_type+'_name']]
        
                                            for element_denom, element_denom_idx in metals_idx[CODE].items():
                                                profile[metallicity_type+'-'+weight+'-ratio_solar_'+normalization][:,element_idx,element_denom_idx] /= norm_info['info'][element][norm_info[metallicity_type+'_name']]/norm_info['info'][element_denom][norm_info[metallicity_type+'_name']]
                #                                 profile[metallicity_type+'_'+weight+'_ratio_v2_solar_'+normalization][:,element_idx,element_denom_idx] /= norm_info['info'][element][norm_info[metallicity_type+'_name']]/norm_info['info'][element_denom][norm_info[metallicity_type+'_name']]
                
        #                                 print()
        #                                 print()
        #                                 print(weight.upper() + ' finish')
        #                                 print()
        #                                 print()
                
                
                                except Exception as error:
                                    print()
                                    print(f'Error calculating metallicities: {error}')
                                    print()
                                    continue
        
        
        
                        ## Calculate metal mass and metal mass volume density profiles
        
                        expanded_mass = np.expand_dims(profile['mass'], 
                                                       tuple(i for i in range(profile['Z-mass_weighted_mean'].ndim) if i != RADIAL_AXIS))
                        expanded_mass_enc = np.expand_dims(profile['mass_enc'], 
                                                           tuple(i for i in range(profile['Z-mass_weighted_mean'].ndim) if i != RADIAL_AXIS))
                        expanded_mass_density = np.expand_dims(profile['mass_density'], 
                                                               tuple(i for i in range(profile['Z-mass_weighted_mean'].ndim) if i != RADIAL_AXIS))
                        expanded_mass_density_enc = np.expand_dims(profile['mass_density_enc'], 
                                                                   tuple(i for i in range(profile['Z-mass_weighted_mean'].ndim) if i != RADIAL_AXIS))
                        expanded_volume = np.expand_dims(profile['volume'], 
                                                         tuple(i for i in range(profile['Z-mass_weighted_mean'].ndim) if i != RADIAL_AXIS))
                        expanded_volume_enc = np.expand_dims(profile['volume_enc'], 
                                                             tuple(i for i in range(profile['Z-mass_weighted_mean'].ndim) if i != RADIAL_AXIS))
        
                        profile['metal_mass'] = profile['Z-mass_weighted_mean'] * expanded_mass
                        profile['metal_mass_enc'] = pnb.array.SimArray(np.nancumsum(profile['metal_mass'], axis=RADIAL_AXIS), units=profile['metal_mass'].units)
                        profile['metal_mass_density'] = profile['metal_mass']/expanded_volume
                        profile['metal_mass_density_enc'] = profile['metal_mass_enc']/expanded_volume_enc
        
                        profile['metal_mass_Migrm'] = profile['metal_mass']/expanded_mass ## should be same as Z_mass
                        profile['metal_mass_enc_Migrmenc'] = profile['metal_mass_enc']/expanded_mass_enc
        
                        profile['metal_mass_density_rhoigrm'] = profile['metal_mass_density']/expanded_mass_density ## also should be same as Z_mass
                        profile['metal_mass_density_enc_rhoigrmenc'] = profile['metal_mass_density_enc']/expanded_mass_density_enc
        
                        profile['metal_mass_density_fb_rhocrit'] = profile['metal_mass_density']/(profiles['cosmo_props']['rho_gas_scaling'])
                        profile['metal_mass_density_enc_fb_rhocrit'] = profile['metal_mass_density_enc']/(profiles['cosmo_props']['rho_gas_scaling'])
        
                        profile['metal_mass_density_Zmass_fb_rhocrit'] = profile['metal_mass_density']/(profile['Z-mass_weighted_mean'] * profiles['cosmo_props']['rho_gas_scaling'])
                        profile['metal_mass_density_enc_Zmassenc_fb_rhocrit'] = profile['metal_mass_density_enc']/(profile['metal_mass_enc_Migrmenc'] * profiles['cosmo_props']['rho_gas_scaling'])
        
                        profile['mass_density_X_Y_fb_rhocrit'] = profile['mass_density'] / (profile['Z-mass_weighted_mean'][:,metals_idx[CODE]['H']] * profile['Z-mass_weighted_mean'][:,metals_idx[CODE]['He']] * profiles['cosmo_props']['rho_gas_scaling'])
                        profile['mass_density_enc_Xenc_Yenc_fb_rhocrit'] = profile['mass_density_enc'] / (profile['metal_mass_enc_Migrmenc'][:,metals_idx[CODE]['H']] * profile['metal_mass_enc_Migrmenc'][:,metals_idx[CODE]['He']] * profiles['cosmo_props']['rho_gas_scaling'])
        
        
                        profile['total_metal_mass'] = pnb.array.SimArray(np.nansum(profile['metal_mass'][:,2:], axis=1), units=profile['metal_mass'].units) ## exclude H and He
                        profile['total_metal_mass_enc'] = pnb.array.SimArray(np.nancumsum(profile['total_metal_mass'], axis=0), units=profile['metal_mass'].units)
                        profile['total_metal_mass_density'] = profile['total_metal_mass']/expanded_volume
                        profile['total_metal_mass_density_enc'] = profile['total_metal_mass_enc']/expanded_volume_enc
        
                        profile['total_metal_mass_density_rhoigrm'] = profile['total_metal_mass_density']/expanded_mass_density ## also should be same as Z_mass_tot
                        profile['total_metal_mass_density_enc_rhoigrmenc'] = profile['total_metal_mass_density_enc']/expanded_mass_density_enc
        
                        profile['total_metal_mass_density_fb_rhocrit'] = profile['total_metal_mass_density']/(profiles['cosmo_props']['rho_gas_scaling'])
                        profile['total_metal_mass_density_enc_fb_rhocrit'] = profile['total_metal_mass_density_enc']/(profiles['cosmo_props']['rho_gas_scaling'])
        
                        ## Add extra axis so can be plotted with other metals as if it is an element, by giving its element as H (ie. index 0)
                        profile['total_metal_mass_for_metals'] = profile['total_metal_mass'][:,np.newaxis]
                        profile['total_metal_mass_enc_for_metals'] = profile['total_metal_mass_enc'][:,np.newaxis]
                        profile['total_metal_mass_density_for_metals'] = profile['total_metal_mass_density'][:,np.newaxis]
                        profile['total_metal_mass_density_enc_for_metals'] = profile['total_metal_mass_density_enc'][:,np.newaxis]
                        profile['total_metal_mass_density_rhoigrm_for_metals'] = profile['total_metal_mass_density_rhoigrm'][:,np.newaxis]
                        profile['total_metal_mass_density_enc_rhoigrmenc_for_metals'] = profile['total_metal_mass_density_enc_rhoigrmenc'][:,np.newaxis]
                        profile['total_metal_mass_density_fb_rhocrit_for_metals'] = profile['total_metal_mass_density_fb_rhocrit'][:,np.newaxis]
                        profile['total_metal_mass_density_enc_fb_rhocrit_for_metals'] = profile['total_metal_mass_density_enc_fb_rhocrit'][:,np.newaxis]
        
        

                ## Calculate radial derivatives (gradients)
                if CALC_GRADIENTS:
                    print('\nCalculating gradient profiles\n')
                    d_props = copy.deepcopy(list(profile.keys()))
                    for prop in d_props:
                        # print(prop)
                        prop_vals = profile[prop]
                        try:
                            profile[f'd_{prop}'] = gradient(x_profile['physical_rbin_centres'], prop_vals, axis=0)
                        except Exception as error:
                            print(f'Error calculating gradient profile for {prop}: {error}\n')
                
        
                # Add in log versions of all properties
                if CALC_LOG_PROPS:
                    log_prop_names = []
                    log_prop_vals = []
                    for prop in profile.keys():
                #             if prop not in x_axis_props:
                        log_prop_names.append('log_'+prop)
                        log_prop_vals.append(np.log10(profile[prop]))
    
                    for log_prop_name, log_prop_val in zip(log_prop_names, log_prop_vals):
                        profile[log_prop_name] = log_prop_val

            print()
            
            
print()
print('done')
print()
print()






## Save extended profiles and profiles stats to file
print('Saving profiles and parameters to file')

# defaults to allow_pickle=True
save_file = os.path.join(PROFILES_DIR, PROFILES_NAME + '-' + SUFFIX + '-extra_props' + '.pkl')
print(save_file)
save_object_with_dill(profiles, save_file)

save_file = os.path.join(PROFILES_DIR, PROFILES_NAME + '-' + SUFFIX + '-parameters' + '.pkl')
print(save_file)
save_object_with_dill(parameters, save_file,)

print()
print('DONE')