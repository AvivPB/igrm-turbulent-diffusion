## Generate radial profiles #################################################################################

## module load NiaEnv/2022a python/3.11.5
## Python evironment: gen_profiles

## Import libraries
import os
import argparse
import dill
import h5py

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
parser.add_argument('--redshift', action='store', type=float, required=True, 
                   help='Simulation redshift')
parser.add_argument('--snap_file', action='store', type=str, required=True, 
                   help='Path to snapshot to analyze')
parser.add_argument('--caesar_file', action='store', type=str, required=True, 
                   help='Path to caesar file')
parser.add_argument('--halo_ids', action='store', nargs='*', type=int, required=True, 
                   help='Caesar halo ids')

parser.add_argument('--save_file', action='store', type=str, required=True, 
                   help='Path to file in which to save profiles')

# parser.add_argument('--filter', action='store', type=str, default='Sphere', choices=['Sphere', 'Cylinder'],
#                    help="Filter to place on each halo: 'Sphere' (for 3d or 2d profiles), 'Cylinder' (for 2d profiles)")
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

args = parser.parse_args()


save_file = args.save_file + '-temp_cut='+args.temp_cut+'-nh_cut='+args.nh_cut+'.pkl'

profile_types = {
    'Profile':Profile,
    'VerticalProfile':VerticalProfile,
    'InclinedProfile':InclinedProfile,
    'QuantileProfile':QuantileProfile,
}
profile_type = profile_types[args.profile_type] 




def save_object_with_dill(obj, filename):
    with open(filename, 'wb') as f:  # Overwrites any existing file.
        dill.dump(obj, f, dill.HIGHEST_PROTOCOL)
    
    

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
def prop_weighted_avg(self, prop, weight):
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
            result[i] = np.average(pprop[self.binind[i]], weights=pweight[self.binind[i]], axis=0)
    return result

@Profile.profile_property
def prop_median(self, prop):
    with self.sim.immediate_mode:
        pprop_ = self.sim[prop]
        pprop = pprop_.view(np.ndarray)
    
    median_prop = pnb.array.SimArray(np.zeros((self.nbins,) + pprop.shape[1:]), units=pprop_.units)
    for i in range(self.nbins):
        if len(self.binind[i]) == 0:
            median_prop[i] = np.nan
        else:
            median_prop[i] = np.nanmedian(pprop[self.binind[i]], axis=0)
    return median_prop



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
            metallicity[i] = np.average(pmf[self.binind[i]], weights=pweight[self.binind[i]], axis=0)
    return metallicity



## Number density/abundance profiles

@Profile.profile_property
def Ne(self):
    with self.sim.immediate_mode:
        pvol_ = self.sim['volume'].in_units('cm**3')
        pvol = pvol_.view(np.ndarray)
        pne_ = self.sim['ne'].in_units('cm**-3')
        pne = pne_.view(np.ndarray)
    
    Ne_ = pnb.array.SimArray(np.zeros(self.nbins), units=pvol_.units * pne_.units)
    for i in range(self.nbins):
        if len(self.binind[i]) == 0:
            Ne_[i] = 0
        else:
            Ne_[i] = np.nansum(pvol[self.binind[i]] * pne[self.binind[i]], axis=0)#.sum()
    return Ne_

@Profile.profile_property
def Nh(self):
    with self.sim.immediate_mode:
        pvol_ = self.sim['volume'].in_units('cm**3')
        pvol = pvol_.view(np.ndarray)
        pnh_ = self.sim['nh'].in_units('cm**-3')
        pnh = pnh_.view(np.ndarray)
    
    Nh_ = pnb.array.SimArray(np.zeros(self.nbins), units=pvol_.units * pnh_.units)
    for i in range(self.nbins):
        if len(self.binind[i]) == 0:
            Nh_[i] = 0
        else:
            Nh_[i] = np.nansum(pvol[self.binind[i]] * pnh[self.binind[i]], axis=0)#.sum()
    return Nh_

@Profile.profile_property
def NX(self):
    with self.sim.immediate_mode:
        pnXnH_ = self.sim['nX/nH'].in_units('')
        pnXnH = pnXnH_.view(np.ndarray)
        pnH_ = self.sim['nh'].in_units('cm**-3')
        pnH = pnH_.view(np.ndarray).reshape(-1,1)    
        pvol_ = self.sim['volume'].in_units('cm**3')
        pvol = pvol_.view(np.ndarray).reshape(-1,1)
    
    n_part, n_elements = pnXnH.shape
    NX_ = pnb.array.SimArray(np.zeros((self.nbins, n_elements)), units=pnXnH_ * pnH_.units * pvol_.units)
    for i in range(self.nbins):
        if len(self.binind[i]) == 0:
            NX_[i] = 0
        else:
            NX_[i] = np.nansum(pnXnH[self.binind[i]] * pnH[self.binind[i]] * pvol[self.binind[i]], axis=0)
    return NX_



## Thermal Properties

@Profile.profile_property
def Lx_band(self, band):
    '''
    Luminosity within each bin, calculated via dirrectly summing.
    '''
    with self.sim.immediate_mode:
        pLx_ = self.sim['Lx_' + band].in_units('erg s**-1')
        pLx = pLx_.view(np.ndarray)
        
    Lx = pnb.array.SimArray(np.zeros(self.nbins), units=pLx_.units)
    for i in range(self.nbins):
        if len(self.binind[i]) == 0:
            Lx[i] = 0.0 #np.nan
        else:
            Lx[i] = np.nansum(pLx[self.binind[i]])#.sum()
    return Lx

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



def profile(snapshot, center, rbins, xray_bands, 
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
        
        result['y']['Ne'] = pg['Ne']
        result['y']['Nh'] = pg['Nh']
#         result['y']['NX'] = pg['NX']
        
        result['y']['mass'] = pg['mass']
        for xband in xray_bands:
            result['y']['Lx_' + xband] = Lx_band(pg, xband)
            
        median_props = ['mass'] + ['Lx_' + xband for xband in xray_bands]
        for prop in median_props:
            result['y'][prop + '_median'] = prop_median(pg, prop)
#             result['y'][prop + '_mean'] = prop_weighted_avg(pg, prop, prop)
        
#         props = ['Z', 'nX/nH', 'T', 'ne', 'K', 'P'] + ['tcool_' + xband for xband in xray_bands]
        props = ['Z', 'T', 'ne', 'K', 'P'] + ['tcool_' + xband for xband in xray_bands]
#         weights = ['mass', 'volume', 'ne', 'T'] + ['Lx_' + xband for xband in xray_bands]
        weights = ['mass', 'volume'] + ['Lx_' + xband for xband in xray_bands]
        for prop in props:
            result['y'][prop + '_median'] = prop_median(pg, prop)
            for weight in weights:
                if prop != weight:
                    result['y'][prop + '_' + weight] = prop_weighted_avg(pg, prop, weight)
                    
        for weight in weights:
            result['y']['Ztot_' + weight] = Ztot_weight(pg, weight)
        
        result['y']['T_spec'] = pg['T_spec']
        
#         result['y']['tdyn'] = pg['dyntime']
        
        snapshot['pos'] = original_pos
    
    return result



################################################# Start of Analysis #########################################################

## Load simulation
print()
print('Reading in simulation snapshot')
s = pnb.load(args.snap_file)
s.set_units_system(velocity='km a**0.5 s**-1', distance='kpc a h**-1', mass='1e10 Msol h**-1')
s.physical_units()
# s['pos'] *= 1e-3
# s['pos'] = s['pos'].in_units('Mpc')
# s['pos'].units = 'Mpc'
print(s['pos'])
print(s['pos'].units)
print(np.min(s['pos']))
print(np.max(s['pos']))
# s.properties['boxsize'] *= 1e-3
print(s.properties['boxsize'])
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
_05_10_band_file = '/project/b/babul/aspadawe/data/atomdb/Data/0.5-10.0keV_emissivity_all_elements_1000bins.hdf5'

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
cal_dat_dir = '/project/b/babul/aspadawe/data/tspec_calibration/'
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



## Calculate gas particle properties required for isolating hot diffuse gas (IGrM)
print()
print('Calculating gas properties reguired for isolating IGrM')

s.gas['u'] /= pnb.array.SimArray(s.properties['a'], units='1')

# Hydrogen mass fraction
if args.code.lower() == 'simba':
    s.gas['X_H'] = 1-s.gas['metals'][:,metals_idx['Z']]-s.gas['metals'][:,metals_idx['He']]
elif args.code.lower() == 'simba-c':
    s.gas['X_H'] = s.gas['metals'][:,metals_idx['H']]

s.gas['nh'] = g_p.nh(s) * 1e9 # Hydrogen number density (1e9 for fixing pynbody issue with length units!!!)
print()
print('nh:', s.gas['nh'], s.gas['nh'].units)
print()
print()
print('ElectronAbundance:', s.gas['ElectronAbundance'])
print()
s.gas['ElectronAbundance'].units = '1'
s.gas['T'] = g_p.temp(s) # From internal energy to temperature
s.gas['temp'] = s.gas['T']

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
igrm = s.gas[total_filt]
# igrm = s.gas
print('DONE')
print()



print()
print('Calculating IGrM properties')

igrm['ne'] = igrm['ElectronAbundance'] * igrm['nh'].in_units('cm**-3') # electron number density
print()
print('ne:', igrm['ne'], igrm['ne'].units)
print()
igrm['volume'] = igrm['mass']/igrm['rho'] # effective volume of gas particles
igrm['K'] = kB * igrm['T'] * igrm['ne']**(-2,3) # entropy
igrm['P'] = kB * igrm['T'] * igrm['ne'] # Pressure


# Calculate mass fraction of the included elements. The first column must be Hydrogen mass fraction
print()
print('Calculating mass fractions')
if args.code.lower() == 'simba':
    # Simba
    igrm['mass_fraction'] = np.zeros(igrm['metals'].shape)
    igrm['mass_fraction'][:, 0] = igrm['X_H']
    igrm['mass_fraction'][:, 1:] = igrm['metals'][:, 1:]
elif args.code.lower() == 'simba-c':
    # Simba-C
    igrm['mass_fraction'] = igrm['metals'][:,1:-3]

igrm['Z'] = igrm['mass_fraction']

print('DONE')
print()

## Gas property initialization
print()
print('Initializing Lx')
Emission_type = []
for xband_ in xray_bands:
    Emission_type.append('Lx_' + xband_)
for i in Emission_type:
    igrm[i] = 0
    igrm[i].units = 'erg s**-1'

print('DONE')
print()


## Abundances relative to solar abundance, don't have to be in gizmo format. 

## New method to calculate abundances and luminosities that uses less memory
print()
# print('Calculating abundances, cooling rates, & X-ray luminosities')
print('Calculating abundances & X-ray luminosities')

nsteps = 500000
tmpgas = igrm[0:10]

## Abundances in solar units of Anders & Grevesse (1989)
tmpgas['abundance'] = g_p.abundance_to_solar(tmpgas['mass_fraction'], elements=elems)

# Convert mass fractions of each element to their densities relative to H density (ie. abundance X/H)
# tmpgas['nX/nH'] = mass_fraction_to_relative_abundance(tmpgas['mass_fraction'], elements=elems)

## Is it important to calculate L_X with abundance in solar units which I will be using later, or does it not matter?
## Or should L_X be calculated without any solar normalization? --> pyatomdb needs AG89 solar normalized abundances
for xband_, xlims_, xfile_, xmode_, xnbins_ in zip(xray_bands, xray_band_limits, xray_emissivity_files, xray_modes, xray_nbins):
    print(xband_)
    ## Total X-ray luminosity
    tmpgas['Lx_'+ xband_] = g_p.calcu_luminosity(gas=tmpgas, filename=xfile_, mode=xmode_, band=xlims_, elements=elems, bins=xnbins_)
    
    ## Cooling time calculation from Braspenning+2024 eqn 18
    tmpgas['tcool_' + xband_] = 6 * tmpgas['ne'] * kB * tmpgas['T'] / (2 * tmpgas['ElectronAbundance'] * tmpgas['Lx_'+ xband_] / 
                                                                       tmpgas['volume'].in_units('cm**3'))

for i in range(nsteps, len(igrm['abundance'][:,0]), nsteps):
    tmpgas = igrm[i-nsteps:i]
    tmpgas['abundance'] = g_p.abundance_to_solar(tmpgas['mass_fraction'], elements=elems)
#     tmpgas['nX/nH'] = mass_fraction_to_relative_abundance(tmpgas['mass_fraction'], elements=elems)
    for xband_, xlims_, xfile_, xmode_, xnbins_ in zip(xray_bands, xray_band_limits, xray_emissivity_files, xray_modes, xray_nbins):
        tmpgas['Lx_'+ xband_] = g_p.calcu_luminosity(gas=tmpgas, filename=xfile_, mode=xmode_, band=xlims_, elements=elems, bins=xnbins_)
        tmpgas['tcool_' + xband_] = 6 * tmpgas['ne'] * kB * tmpgas['T'] / (2 * tmpgas['ElectronAbundance'] * tmpgas['Lx_'+ xband_] / 
                                                                           tmpgas['volume'].in_units('cm**3'))
        
    count = i

tmpgas = igrm[count:]
tmpgas['abundance'] = g_p.abundance_to_solar(tmpgas['mass_fraction'],elements=elems)
# tmpgas['nX/nH'] = mass_fraction_to_relative_abundance(tmpgas['mass_fraction'], elements=elems)
for xband_, xlims_, xfile_, xmode_, xnbins_ in zip(xray_bands, xray_band_limits, xray_emissivity_files, xray_modes, xray_nbins):
    tmpgas['Lx_'+ xband_] = g_p.calcu_luminosity(gas=tmpgas, filename=xfile_, mode=xmode_, band=xlims_, elements=elems, bins=xnbins_)
    tmpgas['tcool_' + xband_] = 6 * tmpgas['ne'] * kB * tmpgas['T'] / (2 * tmpgas['ElectronAbundance'] * tmpgas['Lx_'+ xband_] / 
                                                                       tmpgas['volume'].in_units('cm**3'))

print('DONE')
print()




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


## Radial bins to get the core for CC/NCC classification (set to be in units of R500)
_rbins_core = np.array([0, 0.048, 0.05])
rbins_core = pnb.array.SimArray(_rbins_core, units='')
log_rbins_core = np.log10(rbins_core)
bin_centres_core = (rbins_core[:-1] + rbins_core[1:])/2.
log_bin_centres_core = (log_rbins_core[:-1] + log_rbins_core[1:])/2.


## Larger radial bins to get the core for CC/NCC classification (set to be in units of R500)
_rbins_big_core = np.array([0, 0.1, 0.15])
rbins_big_core = pnb.array.SimArray(_rbins_big_core, units='')
log_rbins_big_core = np.log10(rbins_big_core)
bin_centres_big_core = (rbins_big_core[:-1] + rbins_big_core[1:])/2.
log_bin_centres_big_core = (log_rbins_big_core[:-1] + log_rbins_big_core[1:])/2.


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
    'core':{
        'scale':'R500',
        'bins':rbins_core,
        'log_bins':log_rbins_core,
        'bin_centres':bin_centres_core,
        'log_bin_centres':log_bin_centres_core,
    },
    'big_core':{
        'scale':'R500',
        'bins':rbins_big_core,
        'log_bins':log_rbins_big_core,
        'bin_centres':bin_centres_big_core,
        'log_bin_centres':log_bin_centres_big_core,
    },
}
final_result['halo_profiles'] = {}
final_result['halo_info'] = {}
# final_result['all_profiles'] = {}
# final_result['all_core_profiles'] = {}
# final_result['all_big_core_profiles'] = {}
halo_ids_to_remove = np.array([])
halo_core_ids = []
halo_big_core_ids = []

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
    
    
    ## For binning
    temp_rbins = rbins * xscale_value
    # temp_radius = pnb.array.SimArray(temp_rbins[-1], units=r_units)
    # temp_radius = f'{temp_rbins[-1]} {r_units}'

    ## For determining CC/NCC classification from tcool of core
    temp_rbins_core = rbins_core * xscale_values['R500']
    # temp_radius_core = pnb.array.SimArray(temp_rbins_core[-1], units=r_units)
    
    temp_rbins_big_core = rbins_big_core * xscale_values['R500']
    # temp_radius_big_core = pnb.array.SimArray(temp_rbins_big_core[-1], units=r_units)
    
    
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
        print('Center :', center, center.units)
    except:
        print('Center :', center)
    
    # ## Extra factor of 1.1 just to make sure all particles are included
    # if args.filter.lower() == 'sphere':
    #     filter_full = pnb.filt.Sphere(radius=1.1*temp_radius, cen=center)
    #     filter_core = pnb.filt.Sphere(radius=1.1*temp_radius_core, cen=center)
    #     filter_big_core = pnb.filt.Sphere(radius=1.1*temp_radius_big_core, cen=center)
    # elif args.filter.lower() == 'cylinder':
    #     ## By default, oriented along the z-axis
    #     filter_full = pnb.filt.Disc(radius=1.1*temp_radius, height=3*xscale_values['R500'], cen=center)
    #     filter_core = pnb.filt.Disc(radius=1.1*temp_radius_core, height=3*xscale_values['R500'], cen=center)
    #     filter_big_core = pnb.filt.Disc(radius=1.1*temp_radius_big_core, height=3*xscale_values['R500'], cen=center)


    # print()
    # print(filter_full)
    # print(filter_full.cen)
    # # print(filter_full.cen.units)
    # print(filter_full.radius)
    # # print(filter_full.radius.units)
    # # print(filter_full.where(s))
    # print(s[filter_full])
    # # print(filter_full.where(igrm))
    # print()

    
    # halo_igrm = igrm[filter_full]
    # # temp_gas = filter_full.where(igrm)
    # print('Length of halo_igrm:', len(halo_igrm))
    
    # halo_igrm_core = igrm[filter_core]
    # print('Length of halo_igrm_core:', len(halo_igrm_core))
    
    # halo_igrm_big_core = igrm[filter_big_core]
    # print('Length of halo_igrm_big_core:', len(halo_igrm_big_core))
    
    # Need halo to not be empty of gas
    if xscale_value>0:
        final_result['halo_profiles'][halo_id] = {}
        full_profile = profile(igrm, center, temp_rbins, xray_bands, 
                               profile_type=profile_type, profile_ndims=args.ndim, weight_by=args.weight_by)
        # full_profile = profile(igrm, center, temp_rbins, xray_bands, 
        #                        profile_type=profile_type, profile_ndims=args.ndim, weight_by=args.weight_by)
        # final_result['all_profiles'][halo_id] = temp_result
        final_result['halo_profiles'][halo_id]['full'] = full_profile
        if True:
            core_profile = profile(igrm, center, temp_rbins_core, xray_bands, 
                                   profile_type=profile_type, profile_ndims=args.ndim, weight_by=args.weight_by)
            # final_result['all_core_profiles'][halo_id] = temp_result_core
            final_result['halo_profiles'][halo_id]['core'] = core_profile
            halo_core_ids.append(halo_id)
        if True:
            big_core_profile = profile(igrm, center, temp_rbins_big_core, xray_bands, 
                                       profile_type=profile_type, profile_ndims=args.ndim, weight_by=args.weight_by)
            # final_result['all_big_core_profiles'][halo_id] = temp_result_big_core
            final_result['halo_profiles'][halo_id]['big_core'] = big_core_profile
            halo_big_core_ids.append(halo_id)
    else:
        print("removing id", halo_id)
        halo_ids_to_remove = np.append(halo_ids_to_remove, halo_id)
        
    print()
print()


# Remove the indices of halos that were empty of gas
halo_ids = np.array([halo_id for halo_id in args.halo_ids if halo_id not in halo_ids_to_remove])
halo_core_ids = np.array(halo_core_ids)
halo_big_core_ids = np.array(halo_big_core_ids)

final_result['halo_ids'] = {
    'full':halo_ids,
    'core':halo_core_ids,
    'big_core':halo_big_core_ids,
}

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