## Generate radial profiles #################################################################################

## module load NiaEnv/2022a python/3.11.5
## Python evironment: gen_profiles

## Import libraries
import os
import copy
import argparse
import dill
# import h5py

import numpy as np
import pynbody as pnb
from pynbody.analysis.profile import Profile, VerticalProfile, InclinedProfile, QuantileProfile

import caesar

import XIGrM.gas_properties as g_p
from XIGrM.gas_properties import m_p, k_B, default_elements
import XIGrM.X_properties as x_p
# import XIGrM.halo_analysis as h_a
import XIGrM.prepare_pyatomdb as ppat

from astro_constants import NA_no_units, NA, kB_no_units, kB, mH, mp_no_units, mp, mu_e, mu, G_no_units, G


parser = argparse.ArgumentParser(description="Create radial profiles of general X-ray properties of the IGrM.")

parser.add_argument('--code', action='store', type=str, required=True, 
                   help='Simulation type (currently either just Simba or Simba-C)')
# parser.add_argument('--redshift', action='store', type=float, required=True, 
#                    help='Simulation redshift')
parser.add_argument('--snap_file', action='store', type=str, required=True, 
                   help='Path to snapshot to analyze')
parser.add_argument('--caesar_file', action='store', type=str, required=True, 
                   help='Path to caesar file')
parser.add_argument('--halo_ids', action='store', nargs='*', type=int, required=True, 
                   help='Caesar halo ids')

parser.add_argument('--save_file', action='store', type=str, required=True, 
                   help='Path to file in which to save profiles')

parser.add_argument('--filter', action='store', type=str, default='Sphere', choices=['Sphere', 'Cylinder'],
                   help="Filter to place on each halo: 'Sphere' (for 3d or 2d profiles), 'Cylinder' (for 2d profiles)")
parser.add_argument('--profile_type', action='store', type=str, default='Profile', choices=['Profile', 'VerticalProfile', 'InclinedProfile', 'QuantileProfile'],
                   help="Type of profile to create ('Profile', 'VerticalProfile', 'InclinedProfile', 'QuantileProfile')")
parser.add_argument('--ndim', action='store', type=int, default=3, choices=[2, 3],
                   help='Number of dimensions for profile, ie. 2 is projected, 3 is spherically averaged (can only be 2 or 3)')
parser.add_argument('--weight_by', action='store', type=str, default='mass',
                   help='Property to weight averages in each radial bin by, defaults to mass')
parser.add_argument('--xscale', action='store', type=str, default='Physical', choices=['Physical', 'R500', 'R200'],
                   help="Units of radial axis ('R500', 'R200', or 'Physical')")

parser.add_argument('--temp_cut', action='store', type=str,  default='5e5 K',
                   help='Temperature in K above which to keep gas particles for the hot diffuse gas')
parser.add_argument('--nh_cut', action='store', type=str, default='0.13 cm**-3',
                   help='Hydrogen number density in cm**-3 below which to keep gas particles for the hot diffuse gas')

parser.add_argument('--halo_particles', action=argparse.BooleanOptionalAction, default=False,
                    help='Calculate radial profiles of all particles in the halo')
parser.add_argument('--dm_particles', action=argparse.BooleanOptionalAction, default=False,
                    help='Calculate radial profiles of dm particles in the halo')
parser.add_argument('--bh_particles', action=argparse.BooleanOptionalAction, default=False,
                    help='Calculate radial profiles of bh particles in the halo')
parser.add_argument('--gas_particles', action=argparse.BooleanOptionalAction, default=False,
                    help='Calculate radial profiles of gas particles in the halo')
parser.add_argument('--igrm_particles', action=argparse.BooleanOptionalAction, default=False,
                    help='Calculate radial profiles of igrm particles in the halo')

args = parser.parse_args()


# save_file = args.save_file + '-temp_cut='+args.temp_cut+'-nh_cut='+args.nh_cut+'.pkl'
save_file = f'{args.save_file}.pkl'

profile_types = {
    'Profile':Profile,
    'VerticalProfile':VerticalProfile,
    'InclinedProfile':InclinedProfile,
    'QuantileProfile':QuantileProfile,
}
profile_type = profile_types[args.profile_type]



print()
print(os.path.dirname(save_file))
print()
if not os.path.exists(os.path.dirname(save_file)):
    print('Making profiles file')
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    print()




def save_object_with_dill(obj, filename):
    with open(filename, 'wb') as f:  # Overwrites any existing file.
        dill.dump(obj, f, dill.HIGHEST_PROTOCOL)




## Function for calculating derivatives of profiles with units
def gradient(x, f, axis=None):
    ## Assume x and f are both pnb.array.SimArray
    return pnb.array.SimArray(np.gradient(f, x, axis=axis, edge_order=1), units=f.units/x.units)

def VR(P0, P, V):
    ## Calculate radial velocity of point P (with velocity V) with respect to point P0
    ## Assume all are given in cartesian coordinates
    R = P - P0.in_units(P.units)
    distances = pnb.array.SimArray(np.linalg.norm(R, axis=1), units=R.units)
    # distances = np.linalg.norm(R, axis=1)
    # print(f'\ndistances: {distances} {distances.units}\n')
    dot_products = pnb.array.SimArray(np.sum(V*R, axis=1), units=V.units*R.units)
    # dot_products = np.sum(V*R, axis=1)
    # print(f'\ndot_products: {dot_products} {dot_products.units}\n')

    radial_velocities = pnb.array.SimArray(np.zeros_like(distances), units=V.units)
    # print(f'\nradial_velocities: {radial_velocities} {radial_velocities.units}\n')
    non_zero_dist_indices = np.where(distances != 0)[0]
    # print(f'\nnon_zero_dist_indices: {non_zero_dist_indices}\n')
    # print(f'\nradial_velocities[non_zero_dist_indices]: {radial_velocities[non_zero_dist_indices]} {radial_velocities[non_zero_dist_indices].units}\n')
    # print(f'\ndot_products[non_zero_dist_indices]: {dot_products[non_zero_dist_indices]} {dot_products[non_zero_dist_indices].units}\n')
    # print(f'\ndistances[non_zero_dist_indices]: {distances[non_zero_dist_indices]} {dot_products[non_zero_dist_indices].units}\n')
    radial_velocities[non_zero_dist_indices] = dot_products[non_zero_dist_indices] / distances[non_zero_dist_indices]

    return radial_velocities
    
    

    
    

## Function to turn mass fraction Z to abundance nX/nH for any element
def mass_fraction_to_relative_abundance(mass_fraction, elements=default_elements):
    ## Number density of each element relative to H number density
    
    atomicNumbers = ppat.elsymbs_to_z0s(elements)
    aMasses = ppat.get_atomic_masses(atomicNumbers)
    
    relative_massfraction = mass_fraction / mass_fraction[:, metals_idx_xigrm['H']].reshape(-1,1) # mass fraction relative to hydrogen
    relative_abundance = relative_massfraction * aMasses[0] / aMasses
    relative_abundance = pnb.array.SimArray(relative_abundance)
    relative_abundance.units = ''
    return relative_abundance
    



###### Profile Properties #####################################################################

@Profile.profile_property
def prop_sum(self, prop):
    with self.sim.immediate_mode:
        pprop_ = self.sim[prop]
        pprop = pprop_.view(np.ndarray)

    result = pnb.array.SimArray(np.zeros((self.nbins,) + pprop.shape[1:]), units=pprop_.units)
    for i in range(self.nbins):
        if len(self.binind[i]) == 0:
            result[i] = 0
        else:
            result[i] = np.nansum(pprop[self.binind[i]], axis=0)
    return result

@Profile.profile_property
def prop_mean(self, prop):
    with self.sim.immediate_mode:
        pprop_ = self.sim[prop]
        pprop = pprop_.view(np.ndarray)

    result = pnb.array.SimArray(np.zeros((self.nbins,) + pprop.shape[1:]), units=pprop_.units)
    for i in range(self.nbins):
        if len(self.binind[i]) == 0:
            result[i] = np.nan
        else:
            result[i] = np.nanmean(pprop[self.binind[i]], axis=0)
    return result

@Profile.profile_property
def prop_weighted_mean(self, prop, weight):
    with self.sim.immediate_mode:
        pprop_ = self.sim[prop]
        pprop = pprop_.view(np.ndarray)
        pweight_ = self.sim[weight]
        pweight = pweight_.view(np.ndarray)

    result = pnb.array.SimArray(np.zeros((self.nbins,) + pprop.shape[1:]), units=pprop_.units)
    for i in range(self.nbins):
        if len(self.binind[i]) == 0:
            result[i] = np.nan
        else:
            result[i] = np.ma.average(pprop[self.binind[i]], weights=pweight[self.binind[i]], axis=0)
    return result

@Profile.profile_property
def prop_median(self, prop):
    with self.sim.immediate_mode:
        pprop_ = self.sim[prop]
        pprop = pprop_.view(np.ndarray)
    
    result = pnb.array.SimArray(np.zeros((self.nbins,) + pprop.shape[1:]), units=pprop_.units)
    for i in range(self.nbins):
        if len(self.binind[i]) == 0:
            result[i] = np.nan
        else:
            result[i] = np.nanmedian(pprop[self.binind[i]], axis=0)
    return result

@Profile.profile_property
def prop_range(self, prop):
    with self.sim.immediate_mode:
        pprop_ = self.sim[prop]
        pprop = pprop_.view(np.ndarray)
    
    result = pnb.array.SimArray(np.zeros((self.nbins,) + pprop.shape[1:]), units=pprop_.units)
    for i in range(self.nbins):
        if len(self.binind[i]) == 0:
            result[i] = np.nan
        else:
            result[i] = 0.5*np.ptp(pprop[self.binind[i]], axis=0)
    return result

@Profile.profile_property
def prop_std(self, prop):
    with self.sim.immediate_mode:
        pprop_ = self.sim[prop]
        pprop = pprop_.view(np.ndarray)
    
    result = pnb.array.SimArray(np.zeros((self.nbins,) + pprop.shape[1:]), units=pprop_.units)
    for i in range(self.nbins):
        if len(self.binind[i]) == 0:
            result[i] = np.nan
        else:
            result[i] = np.nanstd(pprop[self.binind[i]], axis=0, ddof=1)
    return result

@Profile.profile_property
def prop_iqr(self, prop):
    with self.sim.immediate_mode:
        pprop_ = self.sim[prop]
        pprop = pprop_.view(np.ndarray)
    
    result = pnb.array.SimArray(np.zeros((self.nbins,) + pprop.shape[1:]), units=pprop_.units)
    for i in range(self.nbins):
        if len(self.binind[i]) == 0:
            result[i] = np.nan
        else:
            quantiles = np.nanquantile(pprop[self.binind[i]], [0.16, 0.84], axis=0)
            result[i] = 0.5*(quantiles[1] - quantiles[0])
    return result



## Geometric Properties

@Profile.profile_property
def volume(self):
    return self._binsize



## Total metallicity

@Profile.profile_property
def Ztot_weight(self, weight):
    with self.sim.immediate_mode:
        pmf_ = self.sim['metals'][:,metals_idx['Z']]
        pmf = pmf_.view(np.ndarray)
        pweight_ = self.sim[weight]
        pweight = pweight_.view(np.ndarray)

    metallicity = pnb.array.SimArray(np.zeros(self.nbins), units='')
    for i in range(self.nbins):
        if len(self.binind[i]) == 0:
            metallicity[i] = np.nan
        else:
            metallicity[i] = np.ma.average(pmf[self.binind[i]], weights=pweight[self.binind[i]], axis=0)
    return metallicity



## Number density/abundance profiles

# @Profile.profile_property
# def Ne(self):
#     with self.sim.immediate_mode:
#         pvol_ = self.sim['volume'].in_units('cm**3')
#         pvol = pvol_.view(np.ndarray)
#         pne_ = self.sim['ne'].in_units('cm**-3')
#         pne = pne_.view(np.ndarray)
    
#     Ne_ = pnb.array.SimArray(np.zeros(self.nbins), units=pvol_.units * pne_.units)
#     for i in range(self.nbins):
#         if len(self.binind[i]) == 0:
#             Ne_[i] = 0
#         else:
#             Ne_[i] = np.nansum(pvol[self.binind[i]] * pne[self.binind[i]], axis=0)#.sum()
#     return Ne_

# @Profile.profile_property
# def Nh(self):
#     with self.sim.immediate_mode:
#         pvol_ = self.sim['volume'].in_units('cm**3')
#         pvol = pvol_.view(np.ndarray)
#         pnh_ = self.sim['nh'].in_units('cm**-3')
#         pnh = pnh_.view(np.ndarray)
    
#     Nh_ = pnb.array.SimArray(np.zeros(self.nbins), units=pvol_.units * pnh_.units)
#     for i in range(self.nbins):
#         if len(self.binind[i]) == 0:
#             Nh_[i] = 0
#         else:
#             Nh_[i] = np.nansum(pvol[self.binind[i]] * pnh[self.binind[i]], axis=0)#.sum()
#     return Nh_

# @Profile.profile_property
# def NX(self):
#     with self.sim.immediate_mode:
#         pnXnH_ = self.sim['nX/nH'].in_units('')
#         pnXnH = pnXnH_.view(np.ndarray)
#         pnH_ = self.sim['nh'].in_units('cm**-3')
#         pnH = pnH_.view(np.ndarray).reshape(-1,1)    
#         pvol_ = self.sim['volume'].in_units('cm**3')
#         pvol = pvol_.view(np.ndarray).reshape(-1,1)
    
#     n_part, n_elements = pnXnH.shape
#     NX_ = pnb.array.SimArray(np.zeros((self.nbins, n_elements)), units=pnXnH_ * pnH_.units * pvol_.units)
#     for i in range(self.nbins):
#         if len(self.binind[i]) == 0:
#             NX_[i] = 0
#         else:
#             NX_[i] = np.nansum(pnXnH[self.binind[i]] * pnH[self.binind[i]] * pvol[self.binind[i]], axis=0)
#     return NX_



## Thermal Properties

# @Profile.profile_property
# def Lx_band(self, band):
#     '''
#     Luminosity within each bin, calculated via dirrectly summing.
#     '''
#     with self.sim.immediate_mode:
#         pLx_ = self.sim['Lx_' + band].in_units('erg s**-1')
#         pLx = pLx_.view(np.ndarray)
        
#     Lx = pnb.array.SimArray(np.zeros(self.nbins), units=pLx_.units)
#     for i in range(self.nbins):
#         if len(self.binind[i]) == 0:
#             Lx[i] = 0.0 #np.nan
#         else:
#             Lx[i] = np.nansum(pLx[self.binind[i]])#.sum()
#     return Lx

@Profile.profile_property
def T_spec(self):
    kT = pnb.array.SimArray(np.zeros(self.nbins), units='keV')
    subsnap = self.sim
    for i in range(self.nbins):
        if len(self.binind[i]) == 0:
            kT[i] = np.nan
        else:
            kT[i] = x_p.cal_tspec(hdgas=subsnap[self.binind[i]], cal_f=tspec_cal_file, datatype='gizmo')
    
    T = (kT/kB).in_units('K')
    return T




# def igrm_profile(snapshot, center, rbins, xray_bands, 
#                  profile_type=Profile, profile_ndims=3, weight_by='mass'):
#     tx = pnb.transformation.inverse_translate(snapshot, center)
#     boxsize = snapshot.properties['boxsize']    
    
#     result = {
#         'x':{},
#         'y':{},
#     }
#     with tx:
#         original_pos = snapshot['pos'].copy()
        
#         # Correct the position of particles crossing the box periodical boundary.
# #         h_a.correct_pos(snapshot['pos'], boxsize)
#         for i in range(3): # Correct the position of particles crossing the box periodical boundary.
#             index1, = np.where(snapshot['pos'][:, i] < -boxsize/2)
#             snapshot['pos'][index1, i] += boxsize
#             index2, = np.where(snapshot['pos'][:, i] > boxsize/2)
#             snapshot['pos'][index2, i] -= boxsize


#         ## By default, for ndim=2, particles are projected onto x-y axis (ie. with z-axis),
#         ## so disc filter is fine
#         pg = profile_type(snapshot, ndim=profile_ndims, bins=rbins, weight_by=weight_by)
        
        
        
#         ## x-axis values
        
#         result['x']['physical_rbins'] = rbins
#         result['x']['log_physical_rbins'] = np.log10(rbins)
#         result['x']['physical_rbin_centres'] = (rbins[:-1] + rbins[1:])/2.
#         result['x']['log_physical_rbin_centres'] = (result['x']['log_physical_rbins'][:-1] + result['x']['log_physical_rbins'][1:])/2.
#         # print(f"\nphysical_rbin_centres units: {result['x']['physical_rbin_centres'].units}\n")
        
        
        
#         ## y-axis values
        
#         result['y']['n'] = pnb.array.SimArray(pg['n'], units='')
#         result['y']['volume'] = pg['volume']
#         result['y']['volume_sph'] = prop_sum(pg, 'volume')
        
#         # result['y']['Ne_v2'] = pg['Ne']
#         # result['y']['Nh_v2'] = pg['Nh']
#         # result['y']['NX'] = pg['NX']
        
#         # result['y']['mass'] = pg['mass']
#         # for xband in xray_bands:
#         #     result['y']['Lx_' + xband + '_v2'] = Lx_band(pg, xband)

#         sum_props = ['mass', 'Ne', 'Nh', 'H_mass']
#         sum_props += ['NWindLaunches', 'DelayTime', 'ContaminationCount']
#         sum_props += ['U', 'KE_total', 'E']#, 'E_v1', 'E_v2', 'E_v3'] 'U_v1', 'U_v2', 'U_v3',
#         sum_props += ['Lx_' + xband for xband in xray_bands]
#         for prop in sum_props:
#             result['y'][prop] = prop_sum(pg, prop)
            
#         avg_props = ['mass', 'vr', 'Z', 'T', 'ne', 'K', 'P', 'phi']
#         avg_props += ['NWindLaunches', 'DelayTime', 'ContaminationCount', 'smooth']
#         avg_props += ['u', 'U_density']
#         # 'u_v1', 'u_v2', 'u_v3',
#         # 'U_v1_density', 'U_v2_density', 'U_v3_density'
#         avg_props += ['KE_total_per_mass', 'KE_total_density']
#         avg_props += ['E_per_mass']#, 'E_v1_per_mass', 'E_v2_per_mass', 'E_v3_per_mass']
#         avg_props += ['E_density']#, 'E_v1_density', 'E_v2_density', 'E_v3_density']
#         avg_props += ['Lx_' + xband for xband in xray_bands]
#         for prop in avg_props:
#             result['y'][prop + '-median'] = prop_median(pg, prop)
#             result['y'][prop + '-mean'] = prop_mean(pg, prop)
# #             result['y'][prop + '_mean'] = prop_weighted_avg(pg, prop, prop)
        
# #         props = ['Z', 'nX/nH', 'T', 'ne', 'K', 'P'] + ['tcool_' + xband for xband in xray_bands]
#         props = ['vr', 'Z', 'T', 'ne', 'K', 'P', 'phi']
#         props += ['NWindLaunches', 'DelayTime', 'ContaminationCount', 'smooth']
#         props += ['u', 'U_density']
#         # 'u_v1', 'u_v2', 'u_v3',
#         # 'U_v1_density', 'U_v2_density', 'U_v3_density'
#         props += ['KE_total_per_mass', 'KE_total_density']
#         props += ['E_per_mass']#, 'E_v1_per_mass', 'E_v2_per_mass', 'E_v3_per_mass']
#         props += ['E_density']#, 'E_v1_density', 'E_v2_density', 'E_v3_density']
#         props += ['tcool_' + xband for xband in xray_bands]
# #         weights = ['mass', 'volume', 'ne', 'T'] + ['Lx_' + xband for xband in xray_bands]
#         weights = ['mass', 'volume'] + ['Lx_' + xband for xband in xray_bands]
#         for prop in props:
#             # result['y'][prop + '_median'] = prop_median(pg, prop)
#             for weight in weights:
#                 if prop != weight:
#                     result['y'][prop + '-' + weight + '_weighted_mean'] = prop_weighted_mean(pg, prop, weight)
                    
#         for weight in weights:
#             result['y']['Ztot-' + weight + '_weighted_mean'] = Ztot_weight(pg, weight)
        
#         result['y']['T-spec'] = pg['T_spec']
        
# #         result['y']['tdyn'] = pg['dyntime']

#         disp_props = ['T', 'K', 'v_mag', 'vr']
#         for prop in disp_props:
#             result['y'][f'{prop}-range'] = prop_range(pg, prop)
#             result['y'][f'{prop}-std'] = prop_std(pg, prop)
#             result['y'][f'{prop}-iqr'] = prop_iqr(pg, prop)

#         # ## Calculate radial derivatives (gradients)
#         # d_props = copy.deepcopy(list(result['y'].keys()))
#         # for prop in d_props:
#         #     prop_vals = result['y'][prop]
#         #     result['y'][f'd_{prop}'] = gradient(result['x']['physical_rbin_centres'], prop_vals, axis=0)
#             # print(f"\n{result['y'][f'd_{prop}']}\n")
#         # result['y']['temp'] = pg['temp']
#         # result['y']['d_temp'] = pg['d_temp']
#         # result['y']['pot'] = pg['pot']
#         # result['y']['d_pot'] = pg['d_pot']
#         # result['y']['d_P-mass_weighted_mean']
#         # result['y']['P'] = pg['P']
#         # result['y']['d_P'] = pg['d_P']
        
        
#         snapshot['pos'] = original_pos
    
#     return result


def gas_profile(snapshot, center, rbins, xray_bands, 
                profile_type=Profile, profile_ndims=3, weight_by='mass'):
    tx = pnb.transformation.inverse_translate(snapshot, center)
    boxsize = snapshot.properties['boxsize']    
    
    result = {
        'x':{},
        'y':{},
    }
    with tx:
        original_pos = snapshot['pos'].copy()
        
        # Correct the position of particles crossing the box periodical boundary.
#         h_a.correct_pos(snapshot['pos'], boxsize)
        for i in range(3): # Correct the position of particles crossing the box periodical boundary.
            index1, = np.where(snapshot['pos'][:, i] < -boxsize/2)
            snapshot['pos'][index1, i] += boxsize
            index2, = np.where(snapshot['pos'][:, i] > boxsize/2)
            snapshot['pos'][index2, i] -= boxsize


        ## By default, for ndim=2, particles are projected onto x-y axis (ie. with z-axis),
        ## so disc filter is fine
        pg = profile_type(snapshot, ndim=profile_ndims, bins=rbins, weight_by=weight_by)
        
        
        
        ## x-axis values
        
        result['x']['physical_rbins'] = rbins
        result['x']['log_physical_rbins'] = np.log10(rbins)
        result['x']['physical_rbin_centres'] = (rbins[:-1] + rbins[1:])/2.
        result['x']['log_physical_rbin_centres'] = (result['x']['log_physical_rbins'][:-1] + result['x']['log_physical_rbins'][1:])/2.
        
        
        
        ## y-axis values
        
        result['y']['n'] = pnb.array.SimArray(pg['n'], units='')
        result['y']['volume'] = pg['volume']
        result['y']['volume_sph'] = prop_sum(pg, 'volume')

        sum_props = ['mass', 'Ne', 'Nh', 'H_mass']
        sum_props += ['NWindLaunches', 'DelayTime']
        if ContaminationCountFound:
            sum_props += ['ContaminationCount']
        sum_props += ['U']
        # sum_props += ['KE_total', 'E']
        sum_props += ['Lx_' + xband for xband in xray_bands]
        for prop in sum_props:
            result['y'][prop] = prop_sum(pg, prop)
            
        avg_props = ['mass', 'Z', 'T', 'ne', 'K', 'P', 'phi']
        # avg_props += ['vr']
        avg_props += ['NWindLaunches', 'DelayTime', 'smooth']
        if ContaminationCountFound:
            avg_props += ['ContaminationCount']
        # avg_props += ['u', 'U_density']
        # avg_props += ['KE_total_per_mass', 'KE_total_density']
        # avg_props += ['E_per_mass', 'E_density']
        avg_props += ['Lx_' + xband for xband in xray_bands]
        for prop in avg_props:
            result['y'][prop + '-median'] = prop_median(pg, prop)
            result['y'][prop + '-mean'] = prop_mean(pg, prop)
        
        props = ['Z', 'T', 'ne', 'K', 'P', 'phi']
        # props += ['vr']
        props += ['NWindLaunches', 'DelayTime', 'smooth']
        if ContaminationCountFound:
            props += ['ContaminationCount']
        # props += ['u', 'U_density']
        # props += ['KE_total_per_mass', 'KE_total_density']
        # props += ['E_per_mass', 'E_density']
        props += ['tcool_' + xband for xband in xray_bands]
        weights = ['mass', 'volume'] + ['Lx_' + xband for xband in xray_bands]
        for prop in props:
            for weight in weights:
                if prop != weight:
                    # print(f'{prop}-{weight}_weighted_mean')
                    result['y'][prop + '-' + weight + '_weighted_mean'] = prop_weighted_mean(pg, prop, weight)
                    
        for weight in weights:
            result['y']['Ztot-' + weight + '_weighted_mean'] = Ztot_weight(pg, weight)
        
        result['y']['T-spec'] = pg['T_spec']

        disp_props = ['T', 'K']
        # disp_props += ['v_mag', 'vr']
        for prop in disp_props:
            result['y'][f'{prop}-range'] = prop_range(pg, prop)
            result['y'][f'{prop}-std'] = prop_std(pg, prop)
            result['y'][f'{prop}-iqr'] = prop_iqr(pg, prop)
        
        
        snapshot['pos'] = original_pos
    
    return result


def dm_profile(snapshot, center, rbins, 
               profile_type=Profile, profile_ndims=3, weight_by='mass'):
    tx = pnb.transformation.inverse_translate(snapshot, center)
    boxsize = snapshot.properties['boxsize']    
    
    result = {
        'x':{},
        'y':{},
    }
    with tx:
        original_pos = snapshot['pos'].copy()
        
        # Correct the position of particles crossing the box periodical boundary.
#         h_a.correct_pos(snapshot['pos'], boxsize)
        for i in range(3): # Correct the position of particles crossing the box periodical boundary.
            index1, = np.where(snapshot['pos'][:, i] < -boxsize/2)
            snapshot['pos'][index1, i] += boxsize
            index2, = np.where(snapshot['pos'][:, i] > boxsize/2)
            snapshot['pos'][index2, i] -= boxsize


        ## By default, for ndim=2, particles are projected onto x-y axis (ie. with z-axis),
        ## so disc filter is fine
        pg = profile_type(snapshot, ndim=profile_ndims, bins=rbins, weight_by=weight_by)
        
        
        
        ## x-axis values
        
        result['x']['physical_rbins'] = rbins
        result['x']['log_physical_rbins'] = np.log10(rbins)
        result['x']['physical_rbin_centres'] = (rbins[:-1] + rbins[1:])/2.
        result['x']['log_physical_rbin_centres'] = (result['x']['log_physical_rbins'][:-1] + result['x']['log_physical_rbins'][1:])/2.
        
        
        
        ## y-axis values
        
        result['y']['n'] = pnb.array.SimArray(pg['n'], units='')
        result['y']['volume'] = pg['volume']

        sum_props = ['mass', 'KE_total']
        if ContaminationCountFound:
            sum_props += ['ContaminationCount']
        for prop in sum_props:
            result['y'][prop] = prop_sum(pg, prop)

        avg_props = ['vr']
        if ContaminationCountFound:
            avg_props += ['ContaminationCount']
        for prop in avg_props:
            result['y'][prop + '-median'] = prop_median(pg, prop)
            result['y'][prop + '-mean'] = prop_mean(pg, prop)

        props = ['vr']
        if ContaminationCountFound:
            props += ['ContaminationCount']
        weights = ['mass']
        for prop in props:
            for weight in weights:
                if prop != weight:
                    result['y'][prop + '-' + weight + '_weighted_mean'] = prop_weighted_mean(pg, prop, weight)

        disp_props = ['v_mag', 'vr']
        for prop in disp_props:
            result['y'][f'{prop}-range'] = prop_range(pg, prop)
            result['y'][f'{prop}-std'] = prop_std(pg, prop)
            result['y'][f'{prop}-iqr'] = prop_iqr(pg, prop)
        
        snapshot['pos'] = original_pos
    
    return result


def bh_profile(snapshot, center, rbins, 
               profile_type=Profile, profile_ndims=3, weight_by='mass'):
    tx = pnb.transformation.inverse_translate(snapshot, center)
    boxsize = snapshot.properties['boxsize']    
    
    result = {
        'x':{},
        'y':{},
    }
    with tx:
        original_pos = snapshot['pos'].copy()
        
        # Correct the position of particles crossing the box periodical boundary.
#         h_a.correct_pos(snapshot['pos'], boxsize)
        for i in range(3): # Correct the position of particles crossing the box periodical boundary.
            index1, = np.where(snapshot['pos'][:, i] < -boxsize/2)
            snapshot['pos'][index1, i] += boxsize
            index2, = np.where(snapshot['pos'][:, i] > boxsize/2)
            snapshot['pos'][index2, i] -= boxsize


        ## By default, for ndim=2, particles are projected onto x-y axis (ie. with z-axis),
        ## so disc filter is fine
        pg = profile_type(snapshot, ndim=profile_ndims, bins=rbins, weight_by=weight_by)
        
        
        
        ## x-axis values
        
        result['x']['physical_rbins'] = rbins
        result['x']['log_physical_rbins'] = np.log10(rbins)
        result['x']['physical_rbin_centres'] = (rbins[:-1] + rbins[1:])/2.
        result['x']['log_physical_rbin_centres'] = (result['x']['log_physical_rbins'][:-1] + result['x']['log_physical_rbins'][1:])/2.
        
        
        
        ## y-axis values
        
        result['y']['n'] = pnb.array.SimArray(pg['n'], units='')
        result['y']['volume'] = pg['volume']

        sum_props = ['mass', 'BH_Mass', 'BH_Mass_AlphaDisk', 'BH_Mdot']
        if ContaminationCountFound:
            sum_props += ['ContaminationCount']
        for prop in sum_props:
            result['y'][prop] = prop_sum(pg, prop)

        avg_props = ['mass', 'BH_Mass', 'BH_NProgs', 'BH_AccretionLength', 'BH_Mdot', 'aform']
        if ContaminationCountFound:
            avg_props += ['ContaminationCount']
        for prop in avg_props:
            result['y'][prop + '-median'] = prop_median(pg, prop)
            result['y'][prop + '-mean'] = prop_mean(pg, prop)

        props = ['BH_NProgs', 'BH_AccretionLength', 'BH_Mdot', 'aform']
        if ContaminationCountFound:
            props += ['ContaminationCount']
        weights = ['mass', 'BH_Mass']
        for prop in props:
            for weight in weights:
                if prop != weight:
                    result['y'][prop + '-' + weight + '_weighted_mean'] = prop_weighted_mean(pg, prop, weight)
        
        snapshot['pos'] = original_pos
    
    return result


def all_profile(snapshot, center, rbins, 
                profile_type=Profile, profile_ndims=3, weight_by='mass'):
    tx = pnb.transformation.inverse_translate(snapshot, center)
    boxsize = snapshot.properties['boxsize']    
    
    result = {
        'x':{},
        'y':{},
    }
    with tx:
        original_pos = snapshot['pos'].copy()
        
        # Correct the position of particles crossing the box periodical boundary.
#         h_a.correct_pos(snapshot['pos'], boxsize)
        for i in range(3): # Correct the position of particles crossing the box periodical boundary.
            index1, = np.where(snapshot['pos'][:, i] < -boxsize/2)
            snapshot['pos'][index1, i] += boxsize
            index2, = np.where(snapshot['pos'][:, i] > boxsize/2)
            snapshot['pos'][index2, i] -= boxsize


        ## By default, for ndim=2, particles are projected onto x-y axis (ie. with z-axis),
        ## so disc filter is fine
        pg = profile_type(snapshot, ndim=profile_ndims, bins=rbins, weight_by=weight_by)
        
        
        
        ## x-axis values
        
        result['x']['physical_rbins'] = rbins
        result['x']['log_physical_rbins'] = np.log10(rbins)
        result['x']['physical_rbin_centres'] = (rbins[:-1] + rbins[1:])/2.
        result['x']['log_physical_rbin_centres'] = (result['x']['log_physical_rbins'][:-1] + result['x']['log_physical_rbins'][1:])/2.
        
        
        
        ## y-axis values
        
        result['y']['n'] = pnb.array.SimArray(pg['n'], units='')
        result['y']['volume'] = pg['volume']

        sum_props = ['mass']
        if ContaminationCountFound:
            sum_props += ['ContaminationCount']
        for prop in sum_props:
            result['y'][prop] = prop_sum(pg, prop)

        avg_props = ['vr', 'phi']
        if ContaminationCountFound:
            avg_props += ['ContaminationCount']
        # avg_props += ['KE_total_per_mass', 'KE_total_density']
        for prop in avg_props:
            result['y'][prop + '-median'] = prop_median(pg, prop)
            result['y'][prop + '-mean'] = prop_mean(pg, prop)

        props = ['vr', 'phi']
        if ContaminationCountFound:
            props += ['ContaminationCount']
        # props += ['KE_total_per_mass', 'KE_total_density']s
        weights = ['mass']
        for prop in props:
            for weight in weights:
                if prop != weight:
                    result['y'][prop + '-' + weight + '_weighted_mean'] = prop_weighted_mean(pg, prop, weight)

        disp_props = ['v_mag', 'vr']
        for prop in disp_props:
            result['y'][f'{prop}-range'] = prop_range(pg, prop)
            result['y'][f'{prop}-std'] = prop_std(pg, prop)
            result['y'][f'{prop}-iqr'] = prop_iqr(pg, prop)
        
        snapshot['pos'] = original_pos
    
    return result



################################################# Start of Analysis ########################################################

## Load simulation
print()
print('Reading in simulation snapshot')
s = pnb.load(args.snap_file)
s.set_units_system(velocity='km a^0.5 s^-1', distance='kpc a h^-1', mass='1e10 Msol h^-1')
s.physical_units()
# s['pos'] *= 1e-3
# s['pos'] = s['pos'].in_units('Mpc')
# s['pos'].units = 'Mpc'
print(s['pos'])
print(s['pos'].units)
print(np.min(s['pos']))
print(np.max(s['pos']))
s.properties['boxsize'] *= 1e-3
print(s.properties['boxsize'])
if 'ContaminationCount' in s.loadable_keys():
    ContaminationCountFound = True
else:
    ContaminationCountFound = False
print('done')
print()

## Halo properties from caesar file
print('Loading caesar file')
obj = caesar.load(args.caesar_file)
# print(obj.info())
# print(obj.haloinfo(top=3))
print('done')
print()


## Cosmological Parameters (figure out how to change for different redshifts)
age = pnb.array.SimArray(pnb.analysis.cosmology.age(s, z=None, unit='Gyr'), units='Gyr')
z = pnb.array.SimArray(s.properties['Redshift'], units='1')
rho_crit = pnb.array.SimArray(pnb.analysis.cosmology.rho_crit(s, z=None, unit='Msol kpc^-3'), units='Msol kpc**-3')
h0 = pnb.array.SimArray(s.properties['h'], units='1')
omL0 = pnb.array.SimArray(s.properties['omegaL0'], units='1')
omB0 = pnb.array.SimArray(s.properties['omegaB0'], units='1')
omM0 = pnb.array.SimArray(s.properties['omegaM0'], units='1')

# s['pos'] *= h0**(-1)

## Scaling density for gas (universal for all halos)
f_b = pnb.array.SimArray(omB0/omM0, units='1')  # cosmic baryon fraction
rho_gas_scaling = rho_crit * f_b


## X-ray emissivity files with correct bands

# Aviv
# _05_18_band_file = '/project/b/babul/aspadawe/data/atomdb/Data/0.5-1.8keV_emissivity_all_elements_1000bins.hdf5'
# _05_2_band_file = '/project/b/babul/aspadawe/data/atomdb/Data/0.5-2.0keV_emissivity_all_elements_1000bins.hdf5'
# _01_24_band_file = '/project/b/babul/aspadawe/data/atomdb/Data/0.1-2.4keV_emissivity_all_elements_1000bins.hdf5'
# _05_7_band_file = '/project/b/babul/aspadawe/data/atomdb/Data/0.5-7.0keV_emissivity_all_elements_1000bins.hdf5'
_05_10_band_file = '/project/rrg-babul-ad/aspadawe/data/atomdb/Data/0.5-10.0keV_emissivity_all_elements_1000bins.hdf5'

# xray_bands = ['0.5-1.8keV_total', '0.5-2.0keV_total', '0.1-2.4keV_total', '0.5-7.0keV_total', '0.5-10.0keV_total', 
#               '0.5-1.8keV_cont', '0.5-2.0keV_cont', '0.1-2.4keV_cont', '0.5-7.0keV_cont', '0.5-10.0keV_cont']

# xray_bands = ['0.5-10.0keV_total', '0.5-10.0keV_cont']
xray_bands = ['0.5-10.0keV_total']

# xray_nbins = [1000, 1000, 1000, 1000, 1000, 
#               1000, 1000, 1000, 1000, 1000]
# xray_nbins = [1000, 1000]
xray_nbins = [1000]
              
              
# xray_band_limits = [[0.5,1.8], [0.5,2.0], [0.1,2.4], [0.5,7.0], [0.5,10.0], 
#                     [0.5,1.8], [0.5,2.0], [0.1,2.4], [0.5,7.0], [0.5,10.0]]
# xray_band_limits = [[0.5,10.0], [0.5,10.0]]
xray_band_limits = [[0.5,10.0]]

# xray_emissivity_files = [_05_18_band_file, _05_2_band_file, _01_24_band_file, _05_7_band_file, _05_10_band_file, 
#                          _05_18_band_file, _05_2_band_file, _01_24_band_file, _05_7_band_file, _05_10_band_file]
# xray_emissivity_files = [_05_10_band_file, _05_10_band_file]
xray_emissivity_files = [_05_10_band_file]

# xray_modes = ['total', 'total', 'total', 'total', 'total', 
#              'cont', 'cont', 'cont', 'cont', 'cont']
# xray_modes = ['total', 'cont']
xray_modes = ['total']



## Find proper pytspec calibration file. See pytspec documentation for details.
# cal_dat_dir = '/home/b/babul/aspadawe/project/data/tspec_calibration/'
cal_dat_dir = '/project/rrg-babul-ad/aspadawe/data/tspec_calibration/'
cal_dat_files = os.listdir(cal_dat_dir)
cal_redshift = []
for calfile in cal_dat_files:
    cal_redshift += [eval(calfile[10:17])]
calfile_idx = np.abs(np.array(cal_redshift) - s.properties['z']).argmin()
tspec_cal_file = (cal_dat_dir + cal_dat_files[calfile_idx]) ## good



## Metal indices
if args.code.lower() == 'simba':
    # Simba
    metals_idx = {'Z':0, 'He': 1, 'C': 2, 'N': 3, 'O': 4, 'Ne': 5, 'Mg': 6, 'Si': 7, 'S': 8, 'Ca': 9, 'Fe': 10}
    metals_idx_xigrm = {'H':0, 'He': 1, 'C': 2, 'N': 3, 'O': 4, 'Ne': 5, 'Mg': 6, 'Si': 7, 'S': 8, 'Ca': 9, 'Fe': 10}
    elems = ['H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'S', 'Ca', 'Fe']
elif args.code.lower() == 'simba-c':
    # Simba-C
    metals_idx = {'Z':0, 'H':1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al':13,
                  'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19,'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24,
                  'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30}
    metals_idx_xigrm = {'H':0, 'He': 1, 'Li': 2, 'Be': 3, 'B': 4, 'C': 5, 'N': 6, 'O': 7, 'F': 8, 'Ne': 9, 'Na': 10, 'Mg': 11, 
                        'Al':12, 'Si': 13, 'P': 14, 'S': 15, 'Cl': 16, 'Ar': 17, 'K': 18,'Ca': 19, 'Sc': 20, 'Ti': 21, 'V': 22, 'Cr': 23,
                      'Mn': 24, 'Fe': 25, 'Co': 26, 'Ni': 27, 'Cu': 28, 'Zn': 29}
    elems=['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si',
           'P', 'S', 'Cl', 'Ar', 'K','Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']#, 'Ga', 'Ge']\n",
else:
    raise SystemExit('Code is not Simba or Simba-C')








## Halo properties from caesar file
# r_units = 'kpc'
# M_units = 'Msol'

# halo_dict = {}
# for halo_id in args.halo_ids:
#     halo_dict[halo_id] = {
#         # 'm500c':pnb.array.SimArray(obj.halos[halo_id].virial_quantities['m500c'].in_units('Msun'), units='Msun'),
#         # 'm200c':pnb.array.SimArray(obj.halos[halo_id].virial_quantities['m200c'].in_units('Msun'), units='Msun'),
#         'r500c':pnb.array.SimArray(obj.halos[halo_id].virial_quantities['r500c'].in_units(r_units), units=r_units),
#         'r200c':pnb.array.SimArray(obj.halos[halo_id].virial_quantities['r200c'].in_units(r_units), units=r_units),
#         'minpotpos':pnb.array.SimArray(obj.halos[halo_id].minpotpos.in_units(r_units), units=r_units),
#     }

# for halo_id, halo_props in halo_dict.items():
#     halo_props['T500'] = 


# halo_props = {
#     'm500c':[],
#     'm200c':[],
#     'r500c':[],
#     'r200c':[],
# }
# halo_props = {
#     'm500c':pnb.array.SimArray([obj.halos[halo_id].virial_quantities['m500c'].in_units('Msun') for haloid in args.halo_ids], units='Msun'),
    
# }
# halo_m500c = pnb.array.SimArray([obj.halos[halo_id].virial_quantities['m500c'].in_units('Msun') for haloid in args.halo_ids], units='Msun')
# halo_m500c = pnb.array.SimArray([obj.halos[halo_id].virial_quantities['m500c'].in_units('Msun') for haloid in args.halo_ids], units='Msun')

# for halo_id in args.halo_ids:
#     for halo_prop, prop_list in halo_props.items():
#         prop_list.append(obj.halos[halo_id].virial_quantities[halo_prop])
#     # halo_props['m500c'].append(obj.halos[halo_id].virial_quantities['m500c'])
# halo_props['minpotpos'] = []
# for halo_id in args.halo_ids:
#     halo_props['minpotpos'].append(obj.halos[halo_id].minpotpos)

# halo_ids = [halo.GroupID for halo in obj.halos]
# halo_m500c = pnb.array.SimArray([halo.virial_quantities['m500c'].in_units('Msun') for halo in obj.halos], units='Msun')
# halo_m200c = pnb.array.SimArray([halo.virial_quantities['m200c'].in_units('Msun') for halo in obj.halos], units='Msun')
# halo_r500c = pnb.array.SimArray([halo.virial_quantities['r500c'].in_units('kpc') for halo in obj.halos], units='kpc')
# halo_r200c = pnb.array.SimArray([halo.virial_quantities['r200c'].in_units('kpc') for halo in obj.halos], units='kpc')



## Calculate profiles

print()
print('Calculating profiles:')
print()

r_units = 'kpc' #'kpc a h**-1'
r_units_caesar = 'kpc' #'kpccm/h'

m_units = 'Msol'
m_units_caesar = 'Msun'

v_units = 'km s**-1'
v_units_caesar = 'km/s'


xaxis_units_dict = {
    'R500':'',
    # 'M500':r_units + ' ' + M_units,
    'R200':'',
    'Physical':r_units,
}



if args.xscale.lower() == 'r500':
    ## Zooms
    # _rbins = np.append([0], np.logspace(-3, 1, 50))
    # _rbins = np.append([0], np.logspace(-3, 0, 50))
    _rbins = np.append([0], np.logspace(-4, np.log10(5), 45))

elif args.xscale.lower() == 'physical':
    ## Zooms
    _rbins = np.append([0], np.logspace(0, 4, 50))

else:
    raise SystemExit('xscale is not r500 or physical')


# _rbins = np.unique(np.append(_rbin1, _rbin2))
rbins = pnb.array.SimArray(_rbins, units=xaxis_units_dict[args.xscale])
log_rbins = np.log10(rbins)
bin_centres = (rbins[:-1] + rbins[1:])/2.
log_bin_centres = (log_rbins[:-1] + log_rbins[1:])/2.


xunit_converter = {
    'R500':'R500',
    'R200':'R200',
    'Physical':r_units,
}

print('rbins [%s]:' % xunit_converter[args.xscale])
print(rbins)
print()


# ## Radial bins to get the core for CC/NCC classification (set to be in units of R500)
# _rbins_core = np.array([0, 0.048, 0.05])
# rbins_core = pnb.array.SimArray(_rbins_core, units='')
# log_rbins_core = np.log10(rbins_core)
# bin_centres_core = (rbins_core[:-1] + rbins_core[1:])/2.
# log_bin_centres_core = (log_rbins_core[:-1] + log_rbins_core[1:])/2.


# ## Larger radial bins to get the core for CC/NCC classification (set to be in units of R500)
# _rbins_big_core = np.array([0, 0.1, 0.15])
# rbins_big_core = pnb.array.SimArray(_rbins_big_core, units='')
# log_rbins_big_core = np.log10(rbins_big_core)
# bin_centres_big_core = (rbins_big_core[:-1] + rbins_big_core[1:])/2.
# log_bin_centres_big_core = (log_rbins_big_core[:-1] + log_rbins_big_core[1:])/2.


## Add in option to create profiles with various different scalings of x axis (R500, R200, Physical, ...)

final_result = {}
final_result['xaxis'] = {
    'full':{
        'scale':args.xscale,
        'bins':rbins,
        'log_bins':log_rbins,
        'bin_centres':bin_centres,
        'log_bin_centres':log_bin_centres,
    },
    # 'core':{
    #     'scale':'R500',
    #     'bins':rbins_core,
    #     'log_bins':log_rbins_core,
    #     'bin_centres':bin_centres_core,
    #     'log_bin_centres':log_bin_centres_core,
    # },
    # 'big_core':{
    #     'scale':'R500',
    #     'bins':rbins_big_core,
    #     'log_bins':log_rbins_big_core,
    #     'bin_centres':bin_centres_big_core,
    #     'log_bin_centres':log_bin_centres_big_core,
    # },
}
final_result['halo_profiles'] = {}
if args.halo_particles:
    final_result['halo_profiles']['all_particles'] = {}
if args.dm_particles:
    final_result['halo_profiles']['dm'] = {}
if args.bh_particles:
    final_result['halo_profiles']['bh'] = {}
if args.gas_particles:
    final_result['halo_profiles']['gas'] = {}
if args.igrm_particles:
    final_result['halo_profiles']['igrm'] = {}
# final_result['halo_profiles'] = {
#     'igrm':{},
#     'gas':{},
#     'dm':{},
#     'bh':{},
#     'all_particles':{},
# }
final_result['halo_info'] = {}
# final_result['all_profiles'] = {}
# final_result['all_core_profiles'] = {}
# final_result['all_big_core_profiles'] = {}
halo_ids_to_remove_igrm = np.array([])
halo_ids_to_remove_gas = np.array([])
halo_ids_to_remove_dm = np.array([])
halo_ids_to_remove_bh = np.array([])
halo_ids_to_remove_all = np.array([])
# halo_core_ids = []
# halo_big_core_ids = []

for halo_id in args.halo_ids:

    print(f'Halo {halo_id}')


    ## Halo information for scaling/normalizing profiles
    halo_info = {
        'M':{
            '500':pnb.array.SimArray(obj.halos[halo_id].virial_quantities['m500c'].in_units(m_units_caesar), units=m_units),
        },
        'R':{
            '500':pnb.array.SimArray(obj.halos[halo_id].virial_quantities['r500c'].in_units(r_units_caesar), units=r_units),
        },
    }

    halo_info['T'] = {
        '500ideal':(G * halo_info['M']['500'] * mu * mp / (kB * halo_info['R']['500'])).in_units('K'),
    }
    halo_info['ne'] = {
        '500ideal':(500. * rho_gas_scaling / (mu_e * mp)).in_units('cm**-3'),
    }
    halo_info['K'] = {
        '500ideal':(kB * halo_info['T']['500ideal'] * halo_info['ne']['500ideal']**(-2./3.)).in_units('keV cm**2'),
    }
    halo_info['P'] = {
        '500ideal':(kB * halo_info['T']['500ideal'] * halo_info['ne']['500ideal']).in_units('keV cm**-3'),
    }

    final_result['halo_info'][halo_id] = halo_info
    
    
    xscale_values = {
        'R500':pnb.array.SimArray(obj.halos[halo_id].virial_quantities['r500c'].in_units(r_units_caesar), units=r_units),
        'R200':pnb.array.SimArray(obj.halos[halo_id].virial_quantities['r200c'].in_units(r_units_caesar), units=r_units),
        'Physical':pnb.array.SimArray(1, units=''),
    }
    
    xscale_value = xscale_values[args.xscale]
    print(args.xscale, ':', xscale_value, xscale_value.units)

    if xscale_value <= 0: continue
    
    
    ## For binning
    temp_rbins = rbins * xscale_value
    temp_radius = pnb.array.SimArray(temp_rbins[-1], units=r_units)
    # temp_radius = f'{temp_rbins[-1]} {r_units}'
    print(f'\ntemp_rbins: {temp_rbins} {temp_rbins.units}\n')
    print(f'temp_radius: {temp_radius} {temp_radius.units}\n')

    # ## For determining CC/NCC classification from tcool of core
    # temp_rbins_core = rbins_core * xscale_values['R500']
    # temp_radius_core = pnb.array.SimArray(temp_rbins_core[-1], units=r_units)
    
    # temp_rbins_big_core = rbins_big_core * xscale_values['R500']
    # temp_radius_big_core = pnb.array.SimArray(temp_rbins_big_core[-1], units=r_units)
    # # print(temp_radius_big_core)
    
    
    # center = halo.center[halo_id - 1]
    # center = halo_dict[halo_id]['minpotpos']
    minpotpos = np.array(obj.halos[halo_id].minpotpos.in_units(r_units_caesar).value)
    # center = obj.halos[halo_id].minpotpos.in_units(r_units).value
    center = pnb.array.SimArray(minpotpos, units=r_units)#, sim=s)
    # center = pnb.snapshot.simsnap.SimArray(minpotpos, units=r_units)
    # center = f'{minpotpos.value} {minpotpos.units}'
    # center = np.array([f'{minpotpos[0]} {r_units}', f'{minpotpos[1]} {r_units}', f'{minpotpos[2]} {r_units}'])
    # center = s.array(minpotpos, r_units)
    try:
        print('minpotpos :', center, center.units)
    except:
        print('minpotpos :', center)

    compos = np.array(obj.halos[halo_id].pos.in_units(r_units_caesar))
    compos = pnb.array.SimArray(compos, units=r_units)
    try:
        print('compos:', compos, compos.units)
    except:
        print('compos :', compos)

    minpotvel = np.array(obj.halos[halo_id].minpotvel.in_units(v_units_caesar))
    minpotvel = pnb.array.SimArray(minpotvel, units=v_units)
    try:
        print('minpotvel:', minpotvel, minpotvel.units)
    except:
        print('minpotvel :', minpotvel)

    comvel = np.array(obj.halos[halo_id].vel.in_units(v_units_caesar))
    comvel = pnb.array.SimArray(comvel, units=v_units)
    try:
        print('comvel:', comvel, comvel.units)
    except:
        print('comvel :', comvel)
    
    ## Extra factor of 1.1 just to make sure all particles are included
    if args.filter.lower() == 'sphere':
        filter_full = pnb.filt.Sphere(radius=1.1*temp_radius, cen=center)
        # filter_core = pnb.filt.Sphere(radius=1.1*temp_radius_core, cen=center)
        # filter_big_core = pnb.filt.Sphere(radius=1.1*temp_radius_big_core, cen=center)
    elif args.filter.lower() == 'cylinder':
        ## By default, oriented along the z-axis
        filter_full = pnb.filt.Disc(radius=1.1*temp_radius, height=3*xscale_values['R500'], cen=center)
        # filter_core = pnb.filt.Disc(radius=1.1*temp_radius_core, height=3*xscale_values['R500'], cen=center)
        # filter_big_core = pnb.filt.Disc(radius=1.1*temp_radius_big_core, height=3*xscale_values['R500'], cen=center)


    print()
    print(filter_full)
    print(filter_full.cen)
    # print(filter_full.cen.units)
    print(filter_full.radius)
    # print(filter_full.radius.units)
    # print(filter_full.where(s))
    print(s[filter_full])
    # print(filter_full.where(igrm))
    print()




    halo_particles = s[filter_full]
    print('Length of halo_particles:', len(halo_particles))

    ## Calculate profiles of all particles ###################################################################
    if args.halo_particles:
        print('Calculating profiles of all particles in halo\n')
        
        # halo_particles = s[filter_full]
        # # temp_gas = filter_full.where(igrm)
        # print('Length of halo_particles:', len(halo_particles))
        
        # halo_particles_core = s[filter_core]
        # print('Length of halo_particles_core:', len(halo_particles_core))
        
        # halo_particles_big_core = s[filter_big_core]
        # print('Length of halo_particles_big_core:', len(halo_particles_big_core))
        
        print()

        with halo_particles.offset_velocity(-minpotvel):
            halo_particles['v_mag'] = pnb.array.SimArray((halo_particles['vel'][:,0]**2 + halo_particles['vel'][:,1]**2 + halo_particles['vel'][:,2]**2)**0.5, units=halo_particles['vel'].units)
            halo_particles['vr'] = VR(center, halo_particles['pos'], halo_particles['vel'])
        
        # Need halo to not be empty of particless
        if len(halo_particles)>0:
            final_result['halo_profiles']['all_particles'][halo_id] = {}
            full_profile = all_profile(halo_particles, center, temp_rbins, 
                                    profile_type=profile_type, profile_ndims=args.ndim, weight_by=args.weight_by)
            # full_profile = profile(igrm, center, temp_rbins, xray_bands, 
            #                        profile_type=profile_type, profile_ndims=args.ndim, weight_by=args.weight_by)
            # final_result['all_profiles'][halo_id] = temp_result
            final_result['halo_profiles']['all_particles'][halo_id]['full'] = full_profile
            # if len(halo_igrm_core)>0:
            #     core_profile = gas_profile(halo_igrm_core, center, temp_rbins_core, xray_bands, 
            #                                profile_type=profile_type, profile_ndims=args.ndim, weight_by=args.weight_by)
            #     # final_result['all_core_profiles'][halo_id] = temp_result_core
            #     final_result['halo_profiles']['all_particles'][halo_id]['core'] = core_profile
            #     halo_core_ids.append(halo_id)
            # if len(halo_igrm_big_core)>0:
            #     big_core_profile = gas_profile(halo_igrm_big_core, center, temp_rbins_big_core, xray_bands, 
            #                                    profile_type=profile_type, profile_ndims=args.ndim, weight_by=args.weight_by)
            #     # final_result['all_big_core_profiles'][halo_id] = temp_result_big_core
            #     final_result['halo_profiles']['all_particles'][halo_id]['big_core'] = big_core_profile
            #     halo_big_core_ids.append(halo_id)
        else:
            print(f"removing halo id {halo_id} from all_particle profiles")
            halo_ids_to_remove_all = np.append(halo_ids_to_remove_all, halo_id)

    #############################################################################################################




    ## Calculate profiles of dm particles ###################################################################
    if args.dm_particles:
        print('Calculating profiles of dm particles in halo\n')

        halo_dm = halo_particles.dm
        print('Length of halo_dm:', len(halo_dm))
        
        # halo_particles_core = s[filter_core]
        # print('Length of halo_particles_core:', len(halo_particles_core))
        
        # halo_particles_big_core = s[filter_big_core]
        # print('Length of halo_particles_big_core:', len(halo_particles_big_core))
        
        print()

        with halo_dm.offset_velocity(-minpotvel):
            halo_dm['v_mag'] = pnb.array.SimArray((halo_dm['vel'][:,0]**2 + halo_dm['vel'][:,1]**2 + halo_dm['vel'][:,2]**2)**0.5, units=halo_dm['vel'].units)
            halo_dm['vr'] = VR(center, halo_dm['pos'], halo_dm['vel'])

            # halo_dm['KE_total'] = 0.5 * halo_dm['mass'] * (halo_dm['vel'][:,0]**2 + halo_dm['vel'][:,1]**2 + halo_dm['vel'][:,2]**2) # Total kinetic energy
            halo_dm['KE_total'] = 0.5 * halo_dm['mass'] * halo_dm['v_mag']**2 # Total kinetic energy
            
        
            # Need halo to not be empty of dm
            if len(halo_dm)>0:
                final_result['halo_profiles']['dm'][halo_id] = {}
                full_profile = dm_profile(halo_dm, center, temp_rbins, 
                                        profile_type=profile_type, profile_ndims=args.ndim, weight_by=args.weight_by)
                final_result['halo_profiles']['dm'][halo_id]['full'] = full_profile
                # if len(halo_igrm_core)>0:
                #     core_profile = gas_profile(halo_igrm_core, center, temp_rbins_core, xray_bands, 
                #                                profile_type=profile_type, profile_ndims=args.ndim, weight_by=args.weight_by)
                #     # final_result['all_core_profiles'][halo_id] = temp_result_core
                #     final_result['halo_profiles']['all_particles'][halo_id]['core'] = core_profile
                #     halo_core_ids.append(halo_id)
                # if len(halo_igrm_big_core)>0:
                #     big_core_profile = gas_profile(halo_igrm_big_core, center, temp_rbins_big_core, xray_bands, 
                #                                    profile_type=profile_type, profile_ndims=args.ndim, weight_by=args.weight_by)
                #     # final_result['all_big_core_profiles'][halo_id] = temp_result_big_core
                #     final_result['halo_profiles']['all_particles'][halo_id]['big_core'] = big_core_profile
                #     halo_big_core_ids.append(halo_id)
            else:
                print(f"removing halo id {halo_id} from dm profiles")
                halo_ids_to_remove_dm = np.append(halo_ids_to_remove_dm, halo_id)

    #############################################################################################################




    ## Calculate profiles of bh particles ###################################################################
    if args.bh_particles:
        print('Calculating profiles of bh particles in halo\n')

        halo_bh = halo_particles.bh
        print('Length of halo_bh:', len(halo_bh))
        
        # halo_particles_core = s[filter_core]
        # print('Length of halo_particles_core:', len(halo_particles_core))
        
        # halo_particles_big_core = s[filter_big_core]
        # print('Length of halo_particles_big_core:', len(halo_particles_big_core))
        
        print()        
        
        # Need halo to not be empty of bhs
        if len(halo_bh)>0:
            final_result['halo_profiles']['bh'][halo_id] = {}
            full_profile = bh_profile(halo_bh, center, temp_rbins, 
                                    profile_type=profile_type, profile_ndims=args.ndim, weight_by=args.weight_by)
            final_result['halo_profiles']['bh'][halo_id]['full'] = full_profile
            # if len(halo_igrm_core)>0:
            #     core_profile = gas_profile(halo_igrm_core, center, temp_rbins_core, xray_bands, 
            #                                profile_type=profile_type, profile_ndims=args.ndim, weight_by=args.weight_by)
            #     # final_result['all_core_profiles'][halo_id] = temp_result_core
            #     final_result['halo_profiles']['all_particles'][halo_id]['core'] = core_profile
            #     halo_core_ids.append(halo_id)
            # if len(halo_igrm_big_core)>0:
            #     big_core_profile = gas_profile(halo_igrm_big_core, center, temp_rbins_big_core, xray_bands, 
            #                                    profile_type=profile_type, profile_ndims=args.ndim, weight_by=args.weight_by)
            #     # final_result['all_big_core_profiles'][halo_id] = temp_result_big_core
            #     final_result['halo_profiles']['all_particles'][halo_id]['big_core'] = big_core_profile
            #     halo_big_core_ids.append(halo_id)
        else:
            print(f"removing halo id {halo_id} from bh profiles")
            halo_ids_to_remove_bh = np.append(halo_ids_to_remove_bh, halo_id)

    #############################################################################################################




    ## Calculate profiles of gas particles ###################################################################
    if args.gas_particles:
        print('Calculating profiles of gas particles in halo\n')

        halo_gas = halo_particles.gas
        print('Length of halo_gas:', len(halo_gas))
        
        # halo_particles_core = s[filter_core]
        # print('Length of halo_particles_core:', len(halo_particles_core))
        
        # halo_particles_big_core = s[filter_big_core]
        # print('Length of halo_particles_big_core:', len(halo_particles_big_core))
        
        print()


        ## Calculate gas particle properties required for isolating hot diffuse gas (IGrM)
        print()
        print('Calculating gas properties for profiles and those reguired for isolating IGrM')
        
        halo_gas['u'] /= pnb.array.SimArray(s.properties['a'], units='1') ## Correction since pynbody reads units of internal energy incorrectly from gizmo snapshots
        
        # Hydrogen mass fraction
        if args.code.lower() == 'simba':
            halo_gas['X_H'] = 1-halo_gas['metals'][:,metals_idx['Z']]-halo_gas['metals'][:,metals_idx['He']]
        elif args.code.lower() == 'simba-c':
            halo_gas['X_H'] = halo_gas['metals'][:,metals_idx['H']]
        
        halo_gas['nh'] = g_p.nh(halo_gas) #* 1e9 # Hydrogen number density (1e9 for fixing pynbody issue with length units!!!)
        print()
        print('nh:', halo_gas['nh'], halo_gas['nh'].units)
        print()
        print()
        print('ElectronAbundance:', halo_gas['ElectronAbundance'])
        print()
        halo_gas['ElectronAbundance'].units = '1'
        # s.gas['T'] = g_p.temp(s) # From internal energy to temperature
        # s.gas['temp'] = s.gas['T']
        print(halo_gas['temp'].units)
        halo_gas['T'] = halo_gas['temp'] ## temp is already a derived array in pnb, and matches slightly more closely with that of yt than the XIGrM temp calculation
        
        print('DONE')
        print()    
        
        
        print()
        print('Calculating gas properties for profiles')

        print(f"\nVelocities before shift: {halo_gas['vel']}\n")
        ## Translate so all positional & velocity properties are relative to centre of halo
        # with halo_igrm.translate(-center).offset_velocity(-minpotvel):
        with halo_gas.offset_velocity(-minpotvel):
            print(f"\nVelocities after shift: {halo_gas['vel']}\n")
        
            halo_gas['ne'] = halo_gas['ElectronAbundance'] * halo_gas['nh'].in_units('cm**-3') # electron number density
            print('\nne:', halo_gas['ne'], halo_gas['ne'].units, '\n')
            halo_gas['volume'] = halo_gas['mass']/halo_gas['rho'] # effective volume of gas particles
            halo_gas['Ne'] = halo_gas['ne']*halo_gas['volume']
            halo_gas['Nh'] = halo_gas['nh']*halo_gas['volume']
            halo_gas['H_mass'] = halo_gas['mass']*halo_gas['X_H']

            halo_gas['v_mag'] = pnb.array.SimArray((halo_gas['vel'][:,0]**2 + halo_gas['vel'][:,1]**2 + halo_gas['vel'][:,2]**2)**0.5, units = halo_gas['vel'].units)
            halo_gas['vr'] = VR(center, halo_gas['pos'], halo_gas['vel'])
            
            halo_gas['K'] = kB * halo_gas['T'] * halo_gas['ne']**(-2,3) # entropy
            halo_gas['P'] = kB * halo_gas['T'] * halo_gas['ne'] # Pressure
            
            halo_gas['U'] = halo_gas['mass'] * halo_gas['u'] # u = internal energy per unit mass
            halo_gas['U_density'] = halo_gas['U']/halo_gas['volume']
            # print(type(igrm['U']), igrm['U'].shape, igrm['U'].units)
            # print(type(igrm['u']), igrm['u'].shape, igrm['u'].units)
            # print(type(igrm['U_density']), igrm['U_density'].shape, igrm['U_density'].units)
            # print()
            
            # halo_igrm['U_v1'] = 1.5 * kB * halo_igrm['T']
            # halo_igrm['u_v1'] = halo_igrm['U_v1']/halo_igrm['mass']
            # halo_igrm['U_v1_density'] = halo_igrm['u_v1']/halo_igrm['volume']
            # print(type(igrm['U_v1']), igrm['U_v1'].shape, igrm['U_v1'].units)#, igrm['U_v1'].in_units('km**2 Msol s**-2').units)
            # print(type(igrm['u_v1']), igrm['u_v1'].shape, igrm['u_v1'].units)
            # print(type(igrm['U_v1_density']), igrm['U_v1_density'].shape, igrm['U_v1_density'].units)
            # print()
            
            # igrm['U_v2'] = 1.5 * igrm['Nh'] * kB * igrm['T']
            # igrm['u_v2'] = igrm['U_v2']/igrm['mass']
            # igrm['U_v2_density'] = igrm['u_v2']/igrm['volume']
            
            # igrm['U_v3'] = 1.5 * igrm['Ne'] * kB * igrm['T']
            # igrm['u_v3'] = igrm['U_v3']/igrm['mass']
            # igrm['U_v3_density'] = igrm['u_v3']/igrm['volume']
            # igrm['U_APB_V2'] = 1.5 * igrm['ne'] * kB * igrm['T']
            
            ## Following properties should switch to proper reference frame when making profiles
            ## i.e. relative to centre of halo --> CONFIRM THIS!
            # halo_gas['KE_total'] = 0.5 * halo_gas['mass'] * (halo_gas['vel'][:,0]**2 + halo_gas['vel'][:,1]**2 + halo_gas['vel'][:,2]**2) # Total kinetic energy
            halo_gas['KE_total'] = 0.5 * halo_gas['mass'] * halo_gas['v_mag']**2
            halo_gas['KE_total_per_mass'] = halo_gas['KE_total']/halo_gas['mass'] # Total kinetic energy per unit mass
            halo_gas['KE_total_density'] = halo_gas['KE_total']/halo_gas['volume'] # Total kinetic energy per unit volume
            # igrm['KE_r'] = 0.5 * igrm['mass'] * (igrm[:,0]**2 + igrm[:,1]**2 + igrm[:,2]**2)
            # print(type(igrm['KE_total']), igrm['KE_total'].shape, igrm['KE_total'].units)
            # print(type(igrm['KE_total_per_mass']), igrm['KE_total_per_mass'].shape, igrm['KE_total_per_mass'].units)
            # print(type(igrm['KE_total_density']), igrm['KE_total_density'].shape, igrm['KE_total_density'].units)
            # print()
            
            halo_gas['E'] = halo_gas['U'] + halo_gas['KE_total'] # Total energy
            halo_gas['E_per_mass'] = halo_gas['u'] + halo_gas['KE_total_per_mass'] # Total energy per unit mass
            halo_gas['E_density'] = halo_gas['U_density'] + halo_gas['KE_total_density'] # Total energy per unit volume
            
            # igrm['E_v1'] = igrm['U_v1'] + igrm['KE_total'] # Total energy
            # igrm['E_v1_per_mass'] = igrm['u_v1'] + igrm['KE_total_per_mass'] # Total energy per unit mass
            # igrm['E_v1_density'] = igrm['U_v1_density'] + igrm['KE_total_density'] # Total energy per unit volume
            
            # igrm['E_v2'] = igrm['U_v2'] + igrm['KE_total'] # Total energy
            # igrm['E_v2_per_mass'] = igrm['u_v2'] + igrm['KE_total_per_mass'] # Total energy per unit mass
            # igrm['E_v2_density'] = igrm['U_v2_density'] + igrm['KE_total_density'] # Total energy per unit volume
            
            # igrm['E_v3'] = igrm['U_v3'] + igrm['KE_total'] # Total energy
            # igrm['E_v3_per_mass'] = igrm['u_v3'] + igrm['KE_total_per_mass'] # Total energy per unit mass
            # igrm['E_v3_density'] = igrm['U_v3_density'] + igrm['KE_total_density'] # Total energy per unit volume
            
            
            # Calculate mass fraction of the included elements. The first column must be Hydrogen mass fraction
            print()
            print('Calculating mass fractions')
            if args.code.lower() == 'simba':
                # Simba
                halo_gas['mass_fraction'] = np.zeros(halo_gas['metals'].shape)
                halo_gas['mass_fraction'][:, 0] = halo_gas['X_H']
                halo_gas['mass_fraction'][:, 1:] = halo_gas['metals'][:, 1:]
            elif args.code.lower() == 'simba-c':
                # Simba-C
                halo_gas['mass_fraction'] = halo_gas['metals'][:,1:-3]
            
            halo_gas['Z'] = halo_gas['mass_fraction']
            
            print('DONE')
            print()
            
            ## Gas property initialization
            print()
            print('Initializing Lx')
            Emission_type = []
            for xband_ in xray_bands:
                Emission_type.append('Lx_' + xband_)
            for i in Emission_type:
                halo_gas[i] = 0
                halo_gas[i].units = 'erg s**-1'
            
            print('DONE')
            print()
            
            
            ## Abundances relative to solar abundance, don't have to be in gizmo format. 
            
            ## New method to calculate abundances and luminosities that uses less memory
            print()
            # print('Calculating abundances, cooling rates, & X-ray luminosities')
            print('Calculating abundances & X-ray luminosities')

            ## Abundances in solar units of Anders & Grevesse (1989)
            halo_gas['abundance'] = g_p.abundance_to_solar(halo_gas['mass_fraction'], elements=elems)

            for xband_, xlims_, xfile_, xmode_, xnbins_ in zip(xray_bands, xray_band_limits, xray_emissivity_files, xray_modes, xray_nbins):
                print(xband_)
                ## Total X-ray luminosity
                halo_gas['Lx_'+ xband_] = g_p.calcu_luminosity(gas=halo_gas, filename=xfile_, mode=xmode_, band=xlims_, elements=elems, bins=xnbins_)
                
                ## Cooling time calculation from Braspenning+2024 eqn 18
                halo_gas['tcool_' + xband_] = 6 * halo_gas['ne'] * kB * halo_gas['T'] / (2 * halo_gas['ElectronAbundance'] * halo_gas['Lx_'+ xband_] / halo_gas['volume'].in_units('cm**3'))
            

            ###### Method below uses less memory - good for when snap/subsnap is large #######
            # nsteps = 500000
            # tmpgas = halo_igrm[0:10]
            
            # ## Abundances in solar units of Anders & Grevesse (1989)
            # tmpgas['abundance'] = g_p.abundance_to_solar(tmpgas['mass_fraction'], elements=elems)
            
            # print()
            # print(len(halo_igrm['abundance'][:,0]))
            # print()
            
            # # Convert mass fractions of each element to their densities relative to H density (ie. abundance X/H)
            # # tmpgas['nX/nH'] = mass_fraction_to_relative_abundance(tmpgas['mass_fraction'], elements=elems)
            
            # ## Is it important to calculate L_X with abundance in solar units which I will be using later, or does it not matter?
            # ## Or should L_X be calculated without any solar normalization? --> pyatomdb needs AG89 solar normalized abundances
            # for xband_, xlims_, xfile_, xmode_, xnbins_ in zip(xray_bands, xray_band_limits, xray_emissivity_files, xray_modes, xray_nbins):
            #     print(xband_)
            #     ## Total X-ray luminosity
            #     tmpgas['Lx_'+ xband_] = g_p.calcu_luminosity(gas=tmpgas, filename=xfile_, mode=xmode_, band=xlims_, elements=elems, bins=xnbins_)
                
            #     ## Cooling time calculation from Braspenning+2024 eqn 18
            #     tmpgas['tcool_' + xband_] = 6 * tmpgas['ne'] * kB * tmpgas['T'] / (2 * tmpgas['ElectronAbundance'] * tmpgas['Lx_'+ xband_] / 
            #                                                                        tmpgas['volume'].in_units('cm**3'))
            
            # # count = 0
            # for i in range(nsteps, len(halo_igrm['abundance'][:,0]), nsteps):
            #     print(i)
            #     tmpgas = halo_igrm[i-nsteps:i]
            #     tmpgas['abundance'] = g_p.abundance_to_solar(tmpgas['mass_fraction'], elements=elems)
            # #     tmpgas['nX/nH'] = mass_fraction_to_relative_abundance(tmpgas['mass_fraction'], elements=elems)
            #     for xband_, xlims_, xfile_, xmode_, xnbins_ in zip(xray_bands, xray_band_limits, xray_emissivity_files, xray_modes, xray_nbins):
            #         tmpgas['Lx_'+ xband_] = g_p.calcu_luminosity(gas=tmpgas, filename=xfile_, mode=xmode_, band=xlims_, elements=elems, bins=xnbins_)
            #         tmpgas['tcool_' + xband_] = 6 * tmpgas['ne'] * kB * tmpgas['T'] / (2 * tmpgas['ElectronAbundance'] * tmpgas['Lx_'+ xband_] / 
            #                                                                            tmpgas['volume'].in_units('cm**3'))
                    
            #     count = i
            #     # print(count)
            
            # tmpgas = halo_igrm[count:]
            # tmpgas['abundance'] = g_p.abundance_to_solar(tmpgas['mass_fraction'],elements=elems)
            # # tmpgas['nX/nH'] = mass_fraction_to_relative_abundance(tmpgas['mass_fraction'], elements=elems)
            # for xband_, xlims_, xfile_, xmode_, xnbins_ in zip(xray_bands, xray_band_limits, xray_emissivity_files, xray_modes, xray_nbins):
            #     tmpgas['Lx_'+ xband_] = g_p.calcu_luminosity(gas=tmpgas, filename=xfile_, mode=xmode_, band=xlims_, elements=elems, bins=xnbins_)
            #     tmpgas['tcool_' + xband_] = 6 * tmpgas['ne'] * kB * tmpgas['T'] / (2 * tmpgas['ElectronAbundance'] * tmpgas['Lx_'+ xband_] / 
            #                                                                        tmpgas['volume'].in_units('cm**3'))
            
            # # igrm['Nx'] = igrm['nX/nH'] * igrm['nh'] * igrm['volume']
            
            print('DONE')
            print()
        
        
            # Need halo to not be empty of gas
            if len(halo_gas)>0:
                final_result['halo_profiles']['gas'][halo_id] = {}
                full_profile = gas_profile(halo_gas, center, temp_rbins, xray_bands,
                                        profile_type=profile_type, profile_ndims=args.ndim, weight_by=args.weight_by)
                final_result['halo_profiles']['gas'][halo_id]['full'] = full_profile
                # if len(halo_igrm_core)>0:
                #     core_profile = gas_profile(halo_igrm_core, center, temp_rbins_core, xray_bands, 
                #                                profile_type=profile_type, profile_ndims=args.ndim, weight_by=args.weight_by)
                #     # final_result['all_core_profiles'][halo_id] = temp_result_core
                #     final_result['halo_profiles']['all_particles'][halo_id]['core'] = core_profile
                #     halo_core_ids.append(halo_id)
                # if len(halo_igrm_big_core)>0:
                #     big_core_profile = gas_profile(halo_igrm_big_core, center, temp_rbins_big_core, xray_bands, 
                #                                    profile_type=profile_type, profile_ndims=args.ndim, weight_by=args.weight_by)
                #     # final_result['all_big_core_profiles'][halo_id] = temp_result_big_core
                #     final_result['halo_profiles']['all_particles'][halo_id]['big_core'] = big_core_profile
                #     halo_big_core_ids.append(halo_id)
            else:
                print(f"removing halo id {halo_id} from gas profiles")
                halo_ids_to_remove_gas = np.append(halo_ids_to_remove_gas, halo_id)

    #############################################################################################################




    #############################################################################################################
    if args.igrm_particles:

        if args.gas_particles == False:

            halo_gas = halo_particles.gas
            print('Length of halo_gas:', len(halo_gas))
            
            # halo_particles_core = s[filter_core]
            # print('Length of halo_particles_core:', len(halo_particles_core))
            
            # halo_particles_big_core = s[filter_big_core]
            # print('Length of halo_particles_big_core:', len(halo_particles_big_core))
            
            print()


            ## Calculate gas particle properties required for isolating hot diffuse gas (IGrM)
            print()
            print('Calculating gas properties for profiles and those reguired for isolating IGrM')
            
            halo_gas['u'] /= pnb.array.SimArray(s.properties['a'], units='1') ## Correction since pynbody reads units of internal energy incorrectly from gizmo snapshots
            
            # Hydrogen mass fraction
            if args.code.lower() == 'simba':
                halo_gas['X_H'] = 1-halo_gas['metals'][:,metals_idx['Z']]-halo_gas['metals'][:,metals_idx['He']]
            elif args.code.lower() == 'simba-c':
                halo_gas['X_H'] = halo_gas['metals'][:,metals_idx['H']]
            
            halo_gas['nh'] = g_p.nh(halo_gas) #* 1e9 # Hydrogen number density (1e9 for fixing pynbody issue with length units!!!)
            print()
            print('nh:', halo_gas['nh'], halo_gas['nh'].units)
            print()
            print()
            print('ElectronAbundance:', halo_gas['ElectronAbundance'])
            print()
            halo_gas['ElectronAbundance'].units = '1'
            # s.gas['T'] = g_p.temp(s) # From internal energy to temperature
            # s.gas['temp'] = s.gas['T']
            print('halo_gas emp units:', halo_gas['temp'].units)
            halo_gas['T'] = halo_gas['temp'] ## temp is already a derived array in pnb, and matches slightly more closely with that of yt than the XIGrM temp calculation
            
            print('DONE')
            print()

        
        
        ## Get hot diffuse gas using provided cuts
        print()
        print('Isolating IGrM (temperature, hydrogen number density, and decoupled wind particle cuts)')
        ## default: temp_cut = '5e5 K'
        ## default: nh_cut = '0.13 cm**-3'
        temp_filt = pnb.filt.HighPass('T', args.temp_cut)
        density_filt = pnb.filt.LowPass('nh', args.nh_cut)
        wind_filt = ~pnb.filt.HighPass('DelayTime', 0)  # Cut out hot decoupled wind particles
        total_filt = temp_filt & density_filt & wind_filt
        # total_filt = wind_filt
        halo_igrm = halo_gas[total_filt]
        # igrm = s.gas
        print('DONE')
        print()



        if args.gas_particles == False:

            halo_igrm['ne'] = halo_igrm['ElectronAbundance'] * halo_igrm['nh'].in_units('cm**-3') # electron number density
            halo_igrm['volume'] = halo_igrm['mass']/halo_igrm['rho'] # effective volume of gas particles
            halo_igrm['Ne'] = halo_igrm['ne']*halo_igrm['volume']
            halo_igrm['Nh'] = halo_igrm['nh']*halo_igrm['volume']
            halo_igrm['H_mass'] = halo_igrm['mass']*halo_igrm['X_H']
            
            halo_igrm['K'] = kB * halo_igrm['T'] * halo_igrm['ne']**(-2,3) # entropy
            halo_igrm['P'] = kB * halo_igrm['T'] * halo_igrm['ne'] # Pressure
            halo_gas['U'] = halo_gas['mass'] * halo_gas['u'] # u = internal energy per unit mass




            # Calculate mass fraction of the included elements. The first column must be Hydrogen mass fraction
            print()
            print('Calculating mass fractions')
            if args.code.lower() == 'simba':
                # Simba
                halo_igrm['mass_fraction'] = np.zeros(halo_igrm['metals'].shape)
                halo_igrm['mass_fraction'][:, 0] = halo_igrm['X_H']
                halo_igrm['mass_fraction'][:, 1:] = halo_igrm['metals'][:, 1:]
            elif args.code.lower() == 'simba-c':
                # Simba-C
                halo_igrm['mass_fraction'] = halo_igrm['metals'][:,1:-3]
            
            halo_igrm['Z'] = halo_igrm['mass_fraction']
            
            print('DONE')
            print()
            
            ## Gas property initialization
            print()
            print('Initializing Lx')
            Emission_type = []
            for xband_ in xray_bands:
                Emission_type.append('Lx_' + xband_)
            for i in Emission_type:
                halo_igrm[i] = 0
                halo_igrm[i].units = 'erg s**-1'
            
            print('DONE')
            print()
            
            
            ## Abundances relative to solar abundance, don't have to be in gizmo format. 
            
            ## New method to calculate abundances and luminosities that uses less memory
            print()
            # print('Calculating abundances, cooling rates, & X-ray luminosities')
            print('Calculating abundances & X-ray luminosities')

            ## Abundances in solar units of Anders & Grevesse (1989)
            halo_igrm['abundance'] = g_p.abundance_to_solar(halo_igrm['mass_fraction'], elements=elems)

            for xband_, xlims_, xfile_, xmode_, xnbins_ in zip(xray_bands, xray_band_limits, xray_emissivity_files, xray_modes, xray_nbins):
                print(xband_)
                ## Total X-ray luminosity
                halo_igrm['Lx_'+ xband_] = g_p.calcu_luminosity(gas=halo_igrm, filename=xfile_, mode=xmode_, band=xlims_, elements=elems, bins=xnbins_)
                
                ## Cooling time calculation from Braspenning+2024 eqn 18
                halo_igrm['tcool_' + xband_] = 6 * halo_igrm['ne'] * kB * halo_igrm['T'] / (2 * halo_igrm['ElectronAbundance'] * halo_igrm['Lx_'+ xband_] / halo_igrm['volume'].in_units('cm**3'))
        
        
        
        # print()
        # print('Calculating IGrM properties for profiles')

        # print(f"\nVelocities before shift: {halo_igrm['vel']}\n")
        # ## Translate so all positional & velocity properties are relative to centre of halo
        # # with halo_igrm.translate(-center).offset_velocity(-minpotvel):
        # with halo_igrm.offset_velocity(-minpotvel):
        #     print(f"\nVelocities after shift: {halo_igrm['vel']}\n")
        
        #     halo_igrm['ne'] = halo_igrm['ElectronAbundance'] * halo_igrm['nh'].in_units('cm**-3') # electron number density
        #     print('\nne:', halo_igrm['ne'], halo_igrm['ne'].units, '\n')
        #     halo_igrm['volume'] = halo_igrm['mass']/halo_igrm['rho'] # effective volume of gas particles
        #     halo_igrm['Ne'] = halo_igrm['ne']*halo_igrm['volume']
        #     halo_igrm['Nh'] = halo_igrm['nh']*halo_igrm['volume']
        #     halo_igrm['H_mass'] = halo_igrm['mass']*halo_igrm['X_H']

        #     halo_igrm['vr'] = VR(center, halo_igrm['pos'], halo_igrm['vel'])
            
        #     halo_igrm['K'] = kB * halo_igrm['T'] * halo_igrm['ne']**(-2,3) # entropy
        #     halo_igrm['P'] = kB * halo_igrm['T'] * halo_igrm['ne'] # Pressure
            
        #     halo_igrm['U'] = halo_igrm['mass'] * halo_igrm['u'] # u = internal energy per unit mass
        #     halo_igrm['U_density'] = halo_igrm['U']/halo_igrm['volume']
        #     # print(type(igrm['U']), igrm['U'].shape, igrm['U'].units)
        #     # print(type(igrm['u']), igrm['u'].shape, igrm['u'].units)
        #     # print(type(igrm['U_density']), igrm['U_density'].shape, igrm['U_density'].units)
        #     # print()
            
        #     # halo_igrm['U_v1'] = 1.5 * kB * halo_igrm['T']
        #     # halo_igrm['u_v1'] = halo_igrm['U_v1']/halo_igrm['mass']
        #     # halo_igrm['U_v1_density'] = halo_igrm['u_v1']/halo_igrm['volume']
        #     # print(type(igrm['U_v1']), igrm['U_v1'].shape, igrm['U_v1'].units)#, igrm['U_v1'].in_units('km**2 Msol s**-2').units)
        #     # print(type(igrm['u_v1']), igrm['u_v1'].shape, igrm['u_v1'].units)
        #     # print(type(igrm['U_v1_density']), igrm['U_v1_density'].shape, igrm['U_v1_density'].units)
        #     # print()
            
        #     # igrm['U_v2'] = 1.5 * igrm['Nh'] * kB * igrm['T']
        #     # igrm['u_v2'] = igrm['U_v2']/igrm['mass']
        #     # igrm['U_v2_density'] = igrm['u_v2']/igrm['volume']
            
        #     # igrm['U_v3'] = 1.5 * igrm['Ne'] * kB * igrm['T']
        #     # igrm['u_v3'] = igrm['U_v3']/igrm['mass']
        #     # igrm['U_v3_density'] = igrm['u_v3']/igrm['volume']
        #     # igrm['U_APB_V2'] = 1.5 * igrm['ne'] * kB * igrm['T']
            
        #     ## Following properties should switch to proper reference frame when making profiles
        #     ## i.e. relative to centre of halo --> CONFIRM THIS!
        #     halo_igrm['KE_total'] = 0.5 * halo_igrm['mass'] * (halo_igrm['vel'][:,0]**2 + halo_igrm['vel'][:,1]**2 + halo_igrm['vel'][:,2]**2) # Total kinetic energy
        #     halo_igrm['KE_total_per_mass'] = halo_igrm['KE_total']/halo_igrm['mass'] # Total kinetic energy per unit mass
        #     halo_igrm['KE_total_density'] = halo_igrm['KE_total']/halo_igrm['volume'] # Total kinetic energy per unit volume
        #     # igrm['KE_r'] = 0.5 * igrm['mass'] * (igrm[:,0]**2 + igrm[:,1]**2 + igrm[:,2]**2)
        #     # print(type(igrm['KE_total']), igrm['KE_total'].shape, igrm['KE_total'].units)
        #     # print(type(igrm['KE_total_per_mass']), igrm['KE_total_per_mass'].shape, igrm['KE_total_per_mass'].units)
        #     # print(type(igrm['KE_total_density']), igrm['KE_total_density'].shape, igrm['KE_total_density'].units)
        #     # print()
            
        #     halo_igrm['E'] = halo_igrm['U'] + halo_igrm['KE_total'] # Total energy
        #     halo_igrm['E_per_mass'] = halo_igrm['u'] + halo_igrm['KE_total_per_mass'] # Total energy per unit mass
        #     halo_igrm['E_density'] = halo_igrm['U_density'] + halo_igrm['KE_total_density'] # Total energy per unit volume
            
        #     # igrm['E_v1'] = igrm['U_v1'] + igrm['KE_total'] # Total energy
        #     # igrm['E_v1_per_mass'] = igrm['u_v1'] + igrm['KE_total_per_mass'] # Total energy per unit mass
        #     # igrm['E_v1_density'] = igrm['U_v1_density'] + igrm['KE_total_density'] # Total energy per unit volume
            
        #     # igrm['E_v2'] = igrm['U_v2'] + igrm['KE_total'] # Total energy
        #     # igrm['E_v2_per_mass'] = igrm['u_v2'] + igrm['KE_total_per_mass'] # Total energy per unit mass
        #     # igrm['E_v2_density'] = igrm['U_v2_density'] + igrm['KE_total_density'] # Total energy per unit volume
            
        #     # igrm['E_v3'] = igrm['U_v3'] + igrm['KE_total'] # Total energy
        #     # igrm['E_v3_per_mass'] = igrm['u_v3'] + igrm['KE_total_per_mass'] # Total energy per unit mass
        #     # igrm['E_v3_density'] = igrm['U_v3_density'] + igrm['KE_total_density'] # Total energy per unit volume
            
            
        #     # Calculate mass fraction of the included elements. The first column must be Hydrogen mass fraction
        #     print()
        #     print('Calculating mass fractions')
        #     if args.code.lower() == 'simba':
        #         # Simba
        #         halo_igrm['mass_fraction'] = np.zeros(halo_igrm['metals'].shape)
        #         halo_igrm['mass_fraction'][:, 0] = halo_igrm['X_H']
        #         halo_igrm['mass_fraction'][:, 1:] = halo_igrm['metals'][:, 1:]
        #     elif args.code.lower() == 'simba-c':
        #         # Simba-C
        #         halo_igrm['mass_fraction'] = halo_igrm['metals'][:,1:-3]
            
        #     halo_igrm['Z'] = halo_igrm['mass_fraction']
            
        #     print('DONE')
        #     print()
            
        #     ## Gas property initialization
        #     print()
        #     print('Initializing Lx')
        #     Emission_type = []
        #     for xband_ in xray_bands:
        #         Emission_type.append('Lx_' + xband_)
        #     for i in Emission_type:
        #         halo_igrm[i] = 0
        #         halo_igrm[i].units = 'erg s**-1'
            
        #     print('DONE')
        #     print()
            
            
        #     ## Abundances relative to solar abundance, don't have to be in gizmo format. 
            
        #     ## New method to calculate abundances and luminosities that uses less memory
        #     print()
        #     # print('Calculating abundances, cooling rates, & X-ray luminosities')
        #     print('Calculating abundances & X-ray luminosities')

        #     ## Abundances in solar units of Anders & Grevesse (1989)
        #     halo_igrm['abundance'] = g_p.abundance_to_solar(halo_igrm['mass_fraction'], elements=elems)

        #     for xband_, xlims_, xfile_, xmode_, xnbins_ in zip(xray_bands, xray_band_limits, xray_emissivity_files, xray_modes, xray_nbins):
        #         print(xband_)
        #         ## Total X-ray luminosity
        #         halo_igrm['Lx_'+ xband_] = g_p.calcu_luminosity(gas=halo_igrm, filename=xfile_, mode=xmode_, band=xlims_, elements=elems, bins=xnbins_)
                
        #         ## Cooling time calculation from Braspenning+2024 eqn 18
        #         halo_igrm['tcool_' + xband_] = 6 * halo_igrm['ne'] * kB * halo_igrm['T'] / (2 * halo_igrm['ElectronAbundance'] * halo_igrm['Lx_'+ xband_] / halo_igrm['volume'].in_units('cm**3'))
            

        #     ###### Method below uses less memory - good for when snap/subsnap is large #######
        #     # nsteps = 500000
        #     # tmpgas = halo_igrm[0:10]
            
        #     # ## Abundances in solar units of Anders & Grevesse (1989)
        #     # tmpgas['abundance'] = g_p.abundance_to_solar(tmpgas['mass_fraction'], elements=elems)
            
        #     # print()
        #     # print(len(halo_igrm['abundance'][:,0]))
        #     # print()
            
        #     # # Convert mass fractions of each element to their densities relative to H density (ie. abundance X/H)
        #     # # tmpgas['nX/nH'] = mass_fraction_to_relative_abundance(tmpgas['mass_fraction'], elements=elems)
            
        #     # ## Is it important to calculate L_X with abundance in solar units which I will be using later, or does it not matter?
        #     # ## Or should L_X be calculated without any solar normalization? --> pyatomdb needs AG89 solar normalized abundances
        #     # for xband_, xlims_, xfile_, xmode_, xnbins_ in zip(xray_bands, xray_band_limits, xray_emissivity_files, xray_modes, xray_nbins):
        #     #     print(xband_)
        #     #     ## Total X-ray luminosity
        #     #     tmpgas['Lx_'+ xband_] = g_p.calcu_luminosity(gas=tmpgas, filename=xfile_, mode=xmode_, band=xlims_, elements=elems, bins=xnbins_)
                
        #     #     ## Cooling time calculation from Braspenning+2024 eqn 18
        #     #     tmpgas['tcool_' + xband_] = 6 * tmpgas['ne'] * kB * tmpgas['T'] / (2 * tmpgas['ElectronAbundance'] * tmpgas['Lx_'+ xband_] / 
        #     #                                                                        tmpgas['volume'].in_units('cm**3'))
            
        #     # # count = 0
        #     # for i in range(nsteps, len(halo_igrm['abundance'][:,0]), nsteps):
        #     #     print(i)
        #     #     tmpgas = halo_igrm[i-nsteps:i]
        #     #     tmpgas['abundance'] = g_p.abundance_to_solar(tmpgas['mass_fraction'], elements=elems)
        #     # #     tmpgas['nX/nH'] = mass_fraction_to_relative_abundance(tmpgas['mass_fraction'], elements=elems)
        #     #     for xband_, xlims_, xfile_, xmode_, xnbins_ in zip(xray_bands, xray_band_limits, xray_emissivity_files, xray_modes, xray_nbins):
        #     #         tmpgas['Lx_'+ xband_] = g_p.calcu_luminosity(gas=tmpgas, filename=xfile_, mode=xmode_, band=xlims_, elements=elems, bins=xnbins_)
        #     #         tmpgas['tcool_' + xband_] = 6 * tmpgas['ne'] * kB * tmpgas['T'] / (2 * tmpgas['ElectronAbundance'] * tmpgas['Lx_'+ xband_] / 
        #     #                                                                            tmpgas['volume'].in_units('cm**3'))
                    
        #     #     count = i
        #     #     # print(count)
            
        #     # tmpgas = halo_igrm[count:]
        #     # tmpgas['abundance'] = g_p.abundance_to_solar(tmpgas['mass_fraction'],elements=elems)
        #     # # tmpgas['nX/nH'] = mass_fraction_to_relative_abundance(tmpgas['mass_fraction'], elements=elems)
        #     # for xband_, xlims_, xfile_, xmode_, xnbins_ in zip(xray_bands, xray_band_limits, xray_emissivity_files, xray_modes, xray_nbins):
        #     #     tmpgas['Lx_'+ xband_] = g_p.calcu_luminosity(gas=tmpgas, filename=xfile_, mode=xmode_, band=xlims_, elements=elems, bins=xnbins_)
        #     #     tmpgas['tcool_' + xband_] = 6 * tmpgas['ne'] * kB * tmpgas['T'] / (2 * tmpgas['ElectronAbundance'] * tmpgas['Lx_'+ xband_] / 
        #     #                                                                        tmpgas['volume'].in_units('cm**3'))
            
        #     # # igrm['Nx'] = igrm['nX/nH'] * igrm['nh'] * igrm['volume']
            
        #     print('DONE')
        #     print()
        





        # with halo_igrm.offset_velocity(-minpotvel):
        
        ## Calculate IGrM Profiles
        print('Calculating profiles of igrm particles in halo\n')
        
        # halo_igrm = igrm[filter_full]
        # temp_gas = filter_full.where(igrm)
        print('Length of halo_igrm:', len(halo_igrm))
        
        # halo_igrm_core = igrm[filter_core]
        # print('Length of halo_igrm_core:', len(halo_igrm_core))
        
        # halo_igrm_big_core = igrm[filter_big_core]
        # print('Length of halo_igrm_big_core:', len(halo_igrm_big_core))
        
        print()
        
        # Need halo to not be empty of gas
        if len(halo_igrm)>0:
            final_result['halo_profiles']['igrm'][halo_id] = {}
            
            full_profile = gas_profile(halo_igrm, center, temp_rbins, xray_bands,
                                    profile_type=profile_type, profile_ndims=args.ndim, weight_by=args.weight_by)
            # full_profile = profile(igrm, center, temp_rbins, xray_bands, 
            #                        profile_type=profile_type, profile_ndims=args.ndim, weight_by=args.weight_by)
            # final_result['all_profiles'][halo_id] = temp_result
            final_result['halo_profiles']['igrm'][halo_id]['full'] = full_profile
            # if len(halo_igrm_core)>0:
            #     core_profile = gas_profile(halo_igrm_core, center, temp_rbins_core, xray_bands, 
            #                                profile_type=profile_type, profile_ndims=args.ndim, weight_by=args.weight_by)
            #     # final_result['all_core_profiles'][halo_id] = temp_result_core
            #     final_result['halo_profiles']['igrm'][halo_id]['core'] = core_profile
            #     halo_core_ids.append(halo_id)
            # if len(halo_igrm_big_core)>0:
            #     big_core_profile = gas_profile(halo_igrm_big_core, center, temp_rbins_big_core, xray_bands, 
            #                                    profile_type=profile_type, profile_ndims=args.ndim, weight_by=args.weight_by)
            #     # final_result['all_big_core_profiles'][halo_id] = temp_result_big_core
            #     final_result['halo_profiles']['igrm'][halo_id]['big_core'] = big_core_profile
            #     halo_big_core_ids.append(halo_id)
        else:
            print(f"removing halo id {halo_id} from igrm profiles")
            halo_ids_to_remove_igrm = np.append(halo_ids_to_remove_igrm, halo_id)

    #############################################################################################################

        
    print()
print()



# Remove the indices of halos that were empty of gas
# halo_ids = np.array([halo_id for halo_id in args.halo_ids if halo_id not in halo_ids_to_remove])
halo_ids_igrm = np.array([halo_id for halo_id in args.halo_ids if halo_id not in halo_ids_to_remove_igrm])
halo_ids_gas = np.array([halo_id for halo_id in args.halo_ids if halo_id not in halo_ids_to_remove_gas])
halo_ids_dm = np.array([halo_id for halo_id in args.halo_ids if halo_id not in halo_ids_to_remove_dm])
halo_ids_bh = np.array([halo_id for halo_id in args.halo_ids if halo_id not in halo_ids_to_remove_bh])
halo_ids_all = np.array([halo_id for halo_id in args.halo_ids if halo_id not in halo_ids_to_remove_all])
# halo_core_ids = np.array(halo_core_ids)
# halo_big_core_ids = np.array(halo_big_core_ids)

# final_result['halo_ids'] = {
#     'full':halo_ids,
#     'core':halo_core_ids,
#     'big_core':halo_big_core_ids,
# }

final_result['halo_ids'] = {}
if args.halo_particles:
    final_result['halo_ids']['all_particles'] = {
        'full':halo_ids_all,
    }
if args.dm_particles:
    final_result['halo_ids']['dm'] = {
        'full':halo_ids_dm,
    }
if args.bh_particles:
    final_result['halo_ids']['bh'] = {
        'full':halo_ids_bh,
    }
if args.gas_particles:
    final_result['halo_ids']['gas'] = {
        'full':halo_ids_gas,
    }
if args.igrm_particles:
    final_result['halo_ids']['igrm'] = {
        'full':halo_ids_igrm,
    }
# final_result['halo_ids'] = {
#     'igrm':{
#         'full':halo_ids_igrm,
#     },
#     'gas':{
#         'full':halo_ids_gas,
#     },
#     'dm':{
#         'full':halo_ids_dm,
#     },
#     'bh':{
#         'full':halo_ids_bh,
#     },
#     'all_particles':{
#         'full':halo_ids_all,
#     },
# }

final_result['code'] = args.code

final_result['cosmo_props'] = {
    'age':age,
    'z':z,
    'h0':h0,
    'OmegaL0':omL0,
    'OmegaM0':omM0,
    'OmegaB0':omB0,
    'rho_crit':rho_crit,
    'rho_gas_scaling':rho_gas_scaling,
}

print('DONE')
print()


# Save profiles
print()
print('Saving profiles to file')
save_object_with_dill(final_result, save_file)
print(save_file)
print('DONE\n')