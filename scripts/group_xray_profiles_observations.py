## Load libraries

# import warnings
# warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
# import matplotlib as mpl
# from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import pynbody as pnb
import pandas as pd
pd.set_option('display.max_columns', None)
# import h5py

# import mmappickle
# import klepto

# import periodictable as pt

# import time
# import copy

import os
from pathlib import Path
from glob import glob

# import astropy.constants as c
# import astropy.units as u
import astropy.io.fits as pyfits
# from astropy.table import Table

# from num2tex import num2tex, configure
# configure(exp_format='times', display_singleton=True)

# import oppenheimer21.groups_profiles_all as plot_group_data
# from oppenheimer21.groups_profiles_all import run_plotting_script as opp21_plot
# import oppenheimer21.group_data

# from palettable.cartocolors.sequential import DarkMint_7, BurgYl_7, Purp_7, Sunset_7, agGrnYl_7

# plt.rcParams['figure.figsize'] = (15,8)


# from astro_constants import NA_no_units, NA, kB_no_units, kB, mH, mp_no_units, mp, mu_e, mu, G_no_units, G, R500_OVER_R200, R200_OVER_R500, R2500_OVER_R200, R200_OVER_R2500, R2500_OVER_R500, R500_OVER_R2500
# from solar_abundances import info_Asplund09, info_Lodders09
import scaling_relation_functions as scale
from scaling_relation_functions import h_sun09, h_simba
# from scaling_relation_functions import M500_Tspec_simbac_N1024L100_renier, M500_Tspec_simba_N1024L100_aviv, M500_Tspec_simbac_N1024L100_aviv, M500_T500_sun09, R500_T500_sun09, T500_R500_sun09
import ned_wright_cosmo_calc





## Load observational data



## Metallicity

data_dir = '/project/b/babul/aspadawe/data/groups-and-clusters'

# os.listdir(data_dir)



# Sarkar+22 data (Asplund+09 solar abundances)

auth = 'Sarkar22'
sub_dir = 'shared_abund'
curr_dir = os.path.join(data_dir, auth, sub_dir)
# print(os.listdir(curr_dir))
obs_sarkar22 = {
    'profiles':{
        'metallicity':{},
        'metallicity ratio':{},
#         gal:{
#             prop.stem:np.loadtxt(prop) for prop in [Path(x) for x in glob(os.path.join(curr_dir, gal, '*'))]
#         } for gal in os.listdir(curr_dir)
    },
    'M500':{
        ## M500/Msun
        'literature':{
            'RXJ1159':6e13,
            'Antlia':5.9e13,
            'ESO3060170':1e14,
            'MKW4':6.5e13,
        },
#         'simbac_full_scaling_renier':{
            
#         },
    },
    'T':{
        ## Tx [keV]
        'literature':{
            'RXJ1159':3, # at 0.3Rvir (1 keV at Rvir)
            'Antlia':1.54, # avg x-ray temp outside core (~2 keV in core) -> global temperature from fitting between 0.2-1R500 (Wong+2016)
            'ESO3060170':2.8, # environment (~2.5 keV within 0.8R200=800 kpc, ~1.13 keV beyond 0.8R200, 1.32 keV in cool-core within 10 kpc)
            'MKW4':1.6, # global temperature (1.3 keV at R500, 1.14 keV at R200) Sarkar+2021
        },
    },
#     'elements':{
#         'RXJ1159':['Fe', 'Mg', 'O', 'S', 'Si'],
#         'Antlia':['Fe', 'Mg', 'O', 'S', 'Si', 'Ni'],
#         'ESO3060170':['Fe', 'Mg', 'O', 'S', 'Si'],
#         'MKW4':['Fe', 'Mg', 'O', 'S', 'Si', 'Ni'],
#     },
    'elements':{
        'Fe':['RXJ1159', 'Antlia', 'ESO3060170', 'MKW4'],
        'Mg':['RXJ1159', 'Antlia', 'ESO3060170', 'MKW4'],
        'O':['RXJ1159', 'Antlia', 'ESO3060170', 'MKW4'],
        'S':['RXJ1159', 'Antlia', 'ESO3060170', 'MKW4'],
        'Si':['RXJ1159', 'Antlia', 'ESO3060170', 'MKW4'],
        'Ni':['Antlia', 'MKW4'],
    },
}



for element, gals in obs_sarkar22['elements'].items():
    obs_sarkar22['profiles']['metallicity'][element] = {}
    element_name = element.lower()
    file = element_name + '.dat'
    for gal in gals:
#         file = element_name
#         if gal == 'MKW4':
#             file += '_rad_scaled_2T'
#         file += '.dat'
        obs_sarkar22['profiles']['metallicity'][element][gal] = np.loadtxt(os.path.join(curr_dir, gal, file))
    
    if element.lower() not in ['fe', 'ni']:
        obs_sarkar22['profiles']['metallicity ratio'][element+'/Fe'] = {}
        element_name = element.lower() + '_fe'
        file = element_name + '_ratio'+ '.dat'
        for gal in gals:
#             file = element_name + '_ratio'
#             if gal == 'MKW4':
#                 file += '_rad_scaled'
#             file += '.dat'
            obs_sarkar22['profiles']['metallicity ratio'][element+'/Fe'][gal] = np.loadtxt(os.path.join(curr_dir, gal, file))
            
            
## Calculate all other metallicity ratios
# print('Sarkar+22:')
# print()
for element_num, gals_num in obs_sarkar22['elements'].items():
#     print('numerator:', element_num)
#     print()
#     if element_num.lower() == 'fe':
#         continue
    for element_denom, gals_denom in obs_sarkar22['elements'].items():
#         print('denominator:', element_denom)
#         print()
        element_ratio = element_num + '/' + element_denom
        common_gals = list(filter(lambda gal:gal in gals_num, gals_denom))
        
        if element_ratio not in obs_sarkar22['profiles']['metallicity ratio'].keys():
            obs_sarkar22['profiles']['metallicity ratio'][element_ratio] = {}
            for gal in common_gals:
#                 print('group:', gal)
                
                profile_num = obs_sarkar22['profiles']['metallicity'][element_num][gal]
                profile_denom = obs_sarkar22['profiles']['metallicity'][element_denom][gal]
                
#                 print(np.shape(profile_num))
#                 print(np.shape(profile_denom))
                
                obs_sarkar22['profiles']['metallicity ratio'][element_ratio][gal] = profile_num.copy()
                
                metallicity_num = profile_num[:,3].copy()
                metallicity_err_lo_num = profile_num[:,4].copy()
                metallicity_err_hi_num = profile_num[:,5].copy()

                metallicity_denom = profile_denom[:,3].copy()
                metallicity_err_lo_denom = profile_denom[:,4].copy()
                metallicity_err_hi_denom = profile_denom[:,5].copy()
                
                
                metallicity_ratio = metallicity_num/metallicity_denom  ## Issue -> num and denom can be different shapes 
                                                                        ## b/c of different number of radial bins??
                                                                        ## Ended up being just fe in MKW4 that was missing last bin
#                 print(metallicity_ratio)
                metallicity_ratio_err_lo = metallicity_ratio * np.sqrt((metallicity_err_lo_num/metallicity_num)**2 +
                                                                       (metallicity_err_lo_denom/metallicity_denom)**2)
                metallicity_ratio_err_hi = metallicity_ratio * np.sqrt((metallicity_err_hi_num/metallicity_num)**2 +
                                                                       (metallicity_err_hi_denom/metallicity_denom)**2)
                
                obs_sarkar22['profiles']['metallicity ratio'][element_ratio][gal][:,3] = metallicity_ratio
                obs_sarkar22['profiles']['metallicity ratio'][element_ratio][gal][:,4] = metallicity_ratio_err_lo
                obs_sarkar22['profiles']['metallicity ratio'][element_ratio][gal][:,5] = metallicity_ratio_err_hi
                
#         print()
                
#     print()
            

# print()
# print()

# # for gal in os.listdir(curr_dir):
# #     for file in glob(os.path.join(curr_dir, gal, '*')):
# for gal, elements in obs_sarkar22['elements'].items():        
#     obs_sarkar22['profiles']['metallicity'][gal] = {}
#     obs_sarkar22['profiles']['metallicity ratio'][gal] = {}
#     for element in elements:
# #         for element_type in ['metallicity', 'metallicity ratio']:
        
#         element_name = element.lower()
#         file = element_name
        
#         if gal == 'MKW4':
#             file += '_rad_scaled_2T'
            
#         file += '.dat'
#         obs_sarkar22['profiles']['metallicity'][gal][element] = np.loadtxt(os.path.join(curr_dir, gal, file))
        
#         if element.lower() not in ['fe', 'ni']:
#             element_name = element.lower() + '_fe'
#             file = element_name + '_ratio'

#             if gal == 'MKW4':
#                 file += '_rad_scaled'

#             file += '.dat'
#             obs_sarkar22['profiles']['metallicity ratio'][gal][element+'/Fe'] = np.loadtxt(os.path.join(curr_dir, gal, file))


for dataset, info in obs_sarkar22['M500'].items():
    for group, M500 in info.items():
        obs_sarkar22['M500'][dataset][group] = pnb.array.SimArray(M500, units='Msol')
    
for dataset, info in obs_sarkar22['T'].items():
    for group, T in info.items():
        obs_sarkar22['T'][dataset][group] = pnb.array.SimArray(T, units='keV')

obs_sarkar22['T']['simbac_full_scaling_renier'] = {}
obs_sarkar22['M500']['simbac_full_scaling_renier'] = {}
for group, T in obs_sarkar22['T']['literature'].items():
    obs_sarkar22['T']['simbac_full_scaling_renier'][group] = T
    obs_sarkar22['M500']['simbac_full_scaling_renier'][group] = scale.M500_Tspec_simbac_N1024L100_renier(T)

# obs_sarkar22
# print(obs_sarkar22)






## Fukushima et al (2023)
## Core regions of 14 nearby early type galaxies (ETGs) from CHEERS measured with XMM-Newton/RGS
## Jointly fit first and second order spectra in 0.45-1.75 keV and 0.8-0.75 keV (??) band
## Spectrally fits core region (0"-60") and radial annuli (0-6,6-18,18-60,60-120)"
## Uses Lodders+09 solar abundances
## kT is VEM (volume emission measure)-weighted avg temperature in keV --> seem to be biased low, so values from de Plaa+2017
## R500 in Mpc

obs_fukushima23 = {
    'individual':{
        'NGC1404':{
            'z':0.0065,
            'R500':0.61,
            'kT_VEM':0.49,
            'kT':0.6,
            'metallicity':{
                'core':{
                    'angle':60,
                    'Fe':{
                        'value':[0.34],
                        'err_lo':[0.02],
                        'err_hi':[0.02],
                    },
                    'N/Fe':{
                        'value':[2.1],
                        'err_lo':[0.4],
                        'err_hi':[0.4],
                    },
                    'O/Fe':{
                        'value':[0.58],
                        'err_lo':[0.04],
                        'err_hi':[0.04],
                    },
                    'Ne/Fe':{
                        'value':[1.28],
                        'err_lo':[0.14],
                        'err_hi':[0.14],
                    },
                    'Mg/Fe':{
                        'value':[1.66],
                        'err_lo':[0.15],
                        'err_hi':[0.15],
                    },
                    'Ni/Fe':{
                        'value':[3.4],
                        'err_lo':[0.5],
                        'err_hi':[0.5],
                    },
                },
            },
        },
        'NGC4636':{
            'z':0.0037,
            'R500':0.35,
            'kT_VEM':0.61,
            'kT':0.8,
            'metallicity':{
                'core':{
                    'angle':60,
                    'Fe':{
                        'value':[0.56],
                        'err_lo':[0.02],
                        'err_hi':[0.02],
                    },
                    'N/Fe':{
                        'value':[1.6],
                        'err_lo':[0.3],
                        'err_hi':[0.3],
                    },
                    'O/Fe':{
                        'value':[0.64],
                        'err_lo':[0.04],
                        'err_hi':[0.04],
                    },
                    'Ne/Fe':{
                        'value':[0.89],
                        'err_lo':[0.09],
                        'err_hi':[0.09],
                    },
                    'Mg/Fe':{
                        'value':[1.10],
                        'err_lo':[0.11],
                        'err_hi':[0.11],
                    },
                    'Ni/Fe':{
                        'value':[2.4],
                        'err_lo':[0.3],
                        'err_hi':[0.3],
                    },
                },
            },
        },
        'NGC4649':{
            'z':0.0037,
            'R500':0.53,
            'kT_VEM':0.82,
            'kT':0.8,
            'metallicity':{
                'core':{
                    'angle':60,
                    'Fe':{
                        'value':[0.39],
                        'err_lo':[0.02],
                        'err_hi':[0.08],
                    },
                    'N/Fe':{
                        'value':[2.6],
                        'err_lo':[0.8],
                        'err_hi':[0.8],
                    },
                    'O/Fe':{
                        'value':[0.73],
                        'err_lo':[0.09],
                        'err_hi':[0.09],
                    },
                    'Ne/Fe':{
                        'value':[1.52],
                        'err_lo':[0.12],
                        'err_hi':[0.12],
                    },
                    'Mg/Fe':{
                        'value':[1.31],
                        'err_lo':[0.14],
                        'err_hi':[0.14],
                    },
                    'Ni/Fe':{
                        'value':[1.7],
                        'err_lo':[0.4],
                        'err_hi':[0.4],
                    },
                },
            },
        },
        'NGC5846':{
            'z':0.0061,
            'R500':0.36,
            'kT_VEM':0.56,
            'kT':0.8,
            'metallicity':{
                'core':{
                    'angle':60,
                    'Fe':{
                        'value':[0.35],
                        'err_lo':[0.01],
                        'err_hi':[0.01],
                    },
                    'N/Fe':{
                        'value':[2.5],
                        'err_lo':[0.5],
                        'err_hi':[0.5],
                    },
                    'O/Fe':{
                        'value':[0.74],
                        'err_lo':[0.09],
                        'err_hi':[0.09],
                    },
                    'Ne/Fe':{
                        'value':[1.24],
                        'err_lo':[0.13],
                        'err_hi':[0.13],
                    },
                    'Mg/Fe':{
                        'value':[1.33],
                        'err_lo':[0.16],
                        'err_hi':[0.16],
                    },
                    'Ni/Fe':{
                        'value':[4.2],
                        'err_lo':[0.6],
                        'err_hi':[0.6],
                    },
                },
            },
        },
        'M49':{
            'z':0.0044,
            'R500':0.53,
            'kT_VEM':0.92,
            'kT':1.0,
            'metallicity':{
                'core':{
                    'angle':60,
                    'Fe':{
                        'value':[0.46],
                        'err_lo':[0.03],
                        'err_hi':[0.03],
                    },
                    'N/Fe':{
                        'value':[2.7],
                        'err_lo':[0.6],
                        'err_hi':[0.6],
                    },
                    'O/Fe':{
                        'value':[1.01],
                        'err_lo':[0.09],
                        'err_hi':[0.09],
                    },
                    'Ne/Fe':{
                        'value':[1.45],
                        'err_lo':[0.19],
                        'err_hi':[0.19],
                    },
                    'Mg/Fe':{
                        'value':[1.86],
                        'err_lo':[0.16],
                        'err_hi':[0.16],
                    },
                    'Ni/Fe':{
                        'value':[2.9],
                        'err_lo':[0.4],
                        'err_hi':[0.4],
                    },
                },
            },
        },
        'HCG62':{
            'z':0.0140,
            'R500':0.46,
            'kT_VEM':0.80,
            'kT':1.1,
            'metallicity':{
                'core':{
                    'angle':60,
                    'Fe':{
                        'value':[0.24],
                        'err_lo':[0.02],
                        'err_hi':[0.10],
                    },
                    'N/Fe':{
                        'value':[1.1],
                        'err_lo':[0.8],
                        'err_hi':[0.8],
                    },
                    'O/Fe':{
                        'value':[1.01],
                        'err_lo':[0.07],
                        'err_hi':[0.07],
                    },
                    'Ne/Fe':{
                        'value':[1.44],
                        'err_lo':[0.1],
                        'err_hi':[0.1],
                    },
                    'Mg/Fe':{
                        'value':[2.2],
                        'err_lo':[0.2],
                        'err_hi':[0.2],
                    },
                    'Ni/Fe':{
                        'value':[1.5],
                        'err_lo':[0.7],
                        'err_hi':[0.7],
                    },
                },
            },
        },
        'Fornax':{
            'z':0.0046,
            'R500':0.40,
            'kT_VEM':0.83,
            'kT':1.2,
            'metallicity':{
                'core':{
                    'angle':60,
                    'Fe':{
                        'value':[0.28],
                        'err_lo':[0.02],
                        'err_hi':[0.02],
                    },
                    'N/Fe':{
                        'value':[4.3],
                        'err_lo':[0.9],
                        'err_hi':[0.9],
                    },
                    'O/Fe':{
                        'value':[1.23],
                        'err_lo':[0.09],
                        'err_hi':[0.09],
                    },
                    'Ne/Fe':{
                        'value':[1.52],
                        'err_lo':[0.13],
                        'err_hi':[0.13],
                    },
                    'Mg/Fe':{
                        'value':[1.77],
                        'err_lo':[0.16],
                        'err_hi':[0.16],
                    },
                    'Ni/Fe':{
                        'value':[4.8],
                        'err_lo':[0.6],
                        'err_hi':[0.6],
                    },
                },
            },
        },
        'NGC1550':{
            'z':0.0123,
            'R500':0.62,
            'kT_VEM':1.26,
            'kT':1.4,
            'metallicity':{
                'core':{
                    'angle':60,
                    'Fe':{
                        'value':[0.57],
                        'err_lo':[0.07],
                        'err_hi':[0.07],
                    },
                    'N/Fe':{
                        'value':[3.5],
                        'err_lo':[1.3],
                        'err_hi':[1.3],
                    },
                    'O/Fe':{
                        'value':[1.31],
                        'err_lo':[0.10],
                        'err_hi':[0.10],
                    },
                    'Ne/Fe':{
                        'value':[0.91],
                        'err_lo':[0.16],
                        'err_hi':[0.16],
                    },
                    'Mg/Fe':{
                        'value':[0.61],
                        'err_lo':[0.18],
                        'err_hi':[0.18],
                    },
                    'Ni/Fe':{
                        'value':[1.9],
                        'err_lo':[0.5],
                        'err_hi':[0.5],
                    },
                },
            },
        },
        'M87':{
            'z':0.0042,
            'R500':0.75,
            'kT_VEM':1.5,
            'kT':1.7,
            'metallicity':{
                'core':{
                    'angle':60,
                    'Fe':{
                        'value':[0.67],
                        'err_lo':[0.04],
                        'err_hi':[0.16],
                    },
                    'N/Fe':{
                        'value':[2.5],
                        'err_lo':[0.4],
                        'err_hi':[0.4],
                    },
                    'O/Fe':{
                        'value':[1.09],
                        'err_lo':[0.05],
                        'err_hi':[0.05],
                    },
                    'Ne/Fe':{
                        'value':[0.97],
                        'err_lo':[0.07],
                        'err_hi':[0.07],
                    },
                    'Mg/Fe':{
                        'value':[1.02],
                        'err_lo':[0.08],
                        'err_hi':[0.08],
                    },
                    'Ni/Fe':{
                        'value':[1.9],
                        'err_lo':[0.2],
                        'err_hi':[0.2],
                    },
                },
            },
        },
        'A3581':{
            'z':0.0214,
            'R500':0.72,
            'kT_VEM':1.56,
            'kT':1.8,
            'metallicity':{
                'core':{
                    'angle':60,
                    'Fe':{
                        'value':[0.66],
                        'err_lo':[0.04],
                        'err_hi':[0.04],
                    },
                    'N/Fe':{
                        'value':[1.3],
                        'err_lo':[1.3],
                        'err_hi':[0.],
                    },
                    'O/Fe':{
                        'value':[1.14],
                        'err_lo':[0.10],
                        'err_hi':[0.10],
                    },
                    'Ne/Fe':{
                        'value':[0.96],
                        'err_lo':[0.17],
                        'err_hi':[0.17],
                    },
                    'Mg/Fe':{
                        'value':[1.15],
                        'err_lo':[0.15],
                        'err_hi':[0.15],
                    },
                    'Ni/Fe':{
                        'value':[0.7],
                        'err_lo':[0.3],
                        'err_hi':[0.3],
                    },
                },
            },
        },
        'A262':{
            'z':0.0161,
            'R500':0.74,
            'kT_VEM':1.8,
            'kT':2.2,
            'metallicity':{
                'core':{
                    'angle':60,
                    'Fe':{
                        'value':[1.0],
                        'err_lo':[0.2],
                        'err_hi':[0.2],
                    },
                    'N/Fe':{
                        'value':[3.8],
                        'err_lo':[1.4],
                        'err_hi':[1.4],
                    },
                    'O/Fe':{
                        'value':[1.23],
                        'err_lo':[0.16],
                        'err_hi':[0.16],
                    },
                    'Ne/Fe':{
                        'value':[1.1],
                        'err_lo':[0.3],
                        'err_hi':[0.3],
                    },
                    'Mg/Fe':{
                        'value':[0.7],
                        'err_lo':[0.3],
                        'err_hi':[0.3],
                    },
                    'Ni/Fe':{
                        'value':[1.1],
                        'err_lo':[0.6],
                        'err_hi':[0.6],
                    },
                },
            },
        },
        'AS1101':{
            'z':0.0580,
            'R500':0.98,
            'kT_VEM':2.6,
            'kT':3.0,
            'metallicity':{
                'core':{
                    'angle':60,
                    'Fe':{
                        'value':[0.67],
                        'err_lo':[0.09],
                        'err_hi':[0.09],
                    },
                    'N/Fe':{
                        'value':[0.8],
                        'err_lo':[0.8],
                        'err_hi':[0.0],
                    },
                    'O/Fe':{
                        'value':[1.10],
                        'err_lo':[0.10],
                        'err_hi':[0.10],
                    },
                    'Ne/Fe':{
                        'value':[0.99],
                        'err_lo':[0.14],
                        'err_hi':[0.14],
                    },
                    'Mg/Fe':{
                        'value':[0.76],
                        'err_lo':[0.15],
                        'err_hi':[0.15],
                    },
                    'Ni/Fe':{
                        'value':[1.0],
                        'err_lo':[0.4],
                        'err_hi':[0.4],
                    },
                },
            },
        },
        'AWM7':{
            'z':0.0172,
            'R500':0.86,
            'kT_VEM':2.8,
            'kT':3.3,
            'metallicity':{
                'core':{
                    'angle':60,
                    'Fe':{
                        'value':[2.0],
                        'err_lo':[0.4],
                        'err_hi':[1.4],
                    },
                    'N/Fe':{
                        'value':[0.4],
                        'err_lo':[0.4],
                        'err_hi':[0.0],
                    },
                    'O/Fe':{
                        'value':[1.17],
                        'err_lo':[0.09],
                        'err_hi':[0.09],
                    },
                    'Ne/Fe':{
                        'value':[0.8],
                        'err_lo':[0.2],
                        'err_hi':[0.2],
                    },
                    'Mg/Fe':{
                        'value':[1.33],
                        'err_lo':[0.17],
                        'err_hi':[0.17],
                    },
                    'Ni/Fe':{
                        'value':[1.3],
                        'err_lo':[0.3],
                        'err_hi':[0.3],
                    },
                },
            },
        },
        'Perseus':{
            'z':0.0183,
            'R500':1.29,
            'kT_VEM':2.73,
            'kT':6.8,
            'metallicity':{
                'core':{
                    'angle':60,
                    'Fe':{
                        'value':[0.84],
                        'err_lo':[0.06],
                        'err_hi':[0.06],
                    },
                    'N/Fe':{
                        'value':[0.9],
                        'err_lo':[0.5],
                        'err_hi':[0.5],
                    },
                    'O/Fe':{
                        'value':[1.58],
                        'err_lo':[0.05],
                        'err_hi':[0.05],
                    },
                    'Ne/Fe':{
                        'value':[1.14],
                        'err_lo':[0.07],
                        'err_hi':[0.07],
                    },
                    'Mg/Fe':{
                        'value':[0.73],
                        'err_lo':[0.05],
                        'err_hi':[0.05],
                    },
                    'Ni/Fe':{
                        'value':[1.02],
                        'err_lo':[0.13],
                        'err_hi':[0.13],
                    },
                },
            },
        },
    },
    'average':{
        'cool_subsample':{
            'N/Fe':{
                'value':[3.0],
                'err_lo':[0.2],
                'err_hi':[0.2],
            },
            'O/Fe':{
                'value':[0.91],
                'err_lo':[0.02],
                'err_hi':[0.02],
            },
            'Ne/Fe':{
                'value':[1.70],
                'err_lo':[0.06],
                'err_hi':[0.06],
            },
            'Mg/Fe':{
                'value':[2.03],
                'err_lo':[0.06],
                'err_hi':[0.06],
            },
            'Ni/Fe':{
                'value':[3.31],
                'err_lo':[0.17],
                'err_hi':[0.17],
            },
        },
        'warm+hot_subsample':{
            'N/Fe':{
                'value':[1.0],
                'err_lo':[0.2],
                'err_hi':[0.2],
            },
            'O/Fe':{
                'value':[1.50],
                'err_lo':[0.02],
                'err_hi':[0.02],
            },
            'Ne/Fe':{
                'value':[1.09],
                'err_lo':[0.04],
                'err_hi':[0.04],
            },
            'Mg/Fe':{
                'value':[0.92],
                'err_lo':[0.04],
                'err_hi':[0.04],
            },
            'Ni/Fe':{
                'value':[1.24],
                'err_lo':[0.09],
                'err_hi':[0.09],
            },
        },
        'low_M<60_LB_subsample':{
            'N/Fe':{
                'value':[3.01],
                'err_lo':[0.18],
                'err_hi':[0.18],
            },
            'O/Fe':{
                'value':[0.93],
                'err_lo':[0.02],
                'err_hi':[0.02],
            },
            'Ne/Fe':{
                'value':[1.41],
                'err_lo':[0.04],
                'err_hi':[0.04],
            },
            'Mg/Fe':{
                'value':[1.59],
                'err_lo':[0.05],
                'err_hi':[0.05],
            },
            'Ni/Fe':{
                'value':[2.76],
                'err_lo':[0.13],
                'err_hi':[0.13],
            },
        },
        'high_M<60_LB_subsample':{
            'N/Fe':{
                'value':[0.6],
                'err_lo':[0.2],
                'err_hi':[0.2],
            },
            'O/Fe':{
                'value':[1.54],
                'err_lo':[0.03],
                'err_hi':[0.03],
            },
            'Ne/Fe':{
                'value':[1.16],
                'err_lo':[0.05],
                'err_hi':[0.05],
            },
            'Mg/Fe':{
                'value':[0.92],
                'err_lo':[0.04],
                'err_hi':[0.04],
            },
            'Ni/Fe':{
                'value':[1.11],
                'err_lo':[0.09],
                'err_hi':[0.09],
            },
        },
    },
}


for group, info in obs_fukushima23['individual'].items():
#     print(group)
    for metal, values in info['metallicity']['core'].items():
#         print(metal)
#         print(values)
#         if metal not in ['Fe', 'angle', 'radius', 'R/R500']:

        if metal not in ['angle', 'radius', 'R/R500']:
#             for data in values.values():
#                 values[data] = pnb.array.SimArray(values[data], units='')
            values['value'] = pnb.array.SimArray(values['value'], units='')
            values['err_lo'] = pnb.array.SimArray(values['err_lo'], units='')
            values['err_hi'] = pnb.array.SimArray(values['err_hi'], units='')

#             values['value']

            
    for metal, values in info['metallicity']['core'].copy().items():
        if '/' in metal:
            metal_num = metal.split('/')[0]
            metal_denom = metal.split('/')[1]
#             print(metal_num)
#             print(metal_denom)
            new_value = values['value'] * info['metallicity']['core'][metal_denom]['value']
#             print(new_value)
            new_err_lo = new_value * np.sqrt( (values['err_lo']/values['value'])**2 + 
                                          (info['metallicity']['core'][metal_denom]['err_lo']/
                                           info['metallicity']['core'][metal_denom]['value'])**2 )
            new_err_hi = new_value * np.sqrt( (values['err_hi']/values['value'])**2 + 
                                          (info['metallicity']['core'][metal_denom]['err_hi']/
                                           info['metallicity']['core'][metal_denom]['value'])**2 )
            info['metallicity']['core'][metal_num] = {
                'value':new_value,
                'err_lo':new_err_lo,
                'err_hi':new_err_hi,
            }
            
    for metal, values in info['metallicity']['core'].copy().items():
        if '/' not in metal and metal.lower() != 'o' and metal not in ['angle', 'radius', 'R/R500']:
            new_value = values['value'] / info['metallicity']['core']['O']['value']
            new_err_lo = new_value * np.sqrt( (values['err_lo']/values['value'])**2 + 
                                          (info['metallicity']['core']['O']['err_lo']/
                                           info['metallicity']['core']['O']['value'])**2 )
            new_err_hi = new_value * np.sqrt( (values['err_hi']/values['value'])**2 + 
                                          (info['metallicity']['core']['O']['err_hi']/
                                           info['metallicity']['core']['O']['value'])**2 )
            info['metallicity']['core'][metal+'/O'] = {
                'value':new_value,
                'err_lo':new_err_lo,
                'err_hi':new_err_hi,
            }
    
#             print(info['metallicity']['core'][metal_num])
#         print()
    
#     print(group)
    info['R500'] = pnb.array.SimArray(info['R500'], units='Mpc')
    info['kT_VEM'] = pnb.array.SimArray(info['kT_VEM'], units='keV')
    info['kT'] = pnb.array.SimArray(info['kT'], units='keV')
    info['M500'] = scale.M500_Tspec_simbac_N1024L100_renier(info['kT'])
    info['metallicity']['core']['angle'] = pnb.array.SimArray(info['metallicity']['core']['angle'], units='arcsec')

    cosmo = ned_wright_cosmo_calc.cosmo(H0=pnb.array.SimArray(68, units='km s**-1 Mpc**-1'), Omega_M=0.3, Omega_Vac=0.7, z=info['z'])
    sky_scale = cosmo.kpc_DA
#     print('sky_scale =', sky_scale, sky_scale.units)
    info['metallicity']['core']['radius'] = sky_scale * info['metallicity']['core']['angle']
#     print('radius =', info['metallicity']['core']['radius'], info['metallicity']['core']['radius'].units)
    info['metallicity']['core']['R/R500'] = (info['metallicity']['core']['radius']/info['R500']).in_units('')
#     info['metallicity']['core']['rbins_R500'] = np.array([0, info['metallicity']['core']['R/R500']])
    rbins_R500 = np.array([0, info['metallicity']['core']['R/R500']])
#     print('R/R500 =', info['metallicity']['core']['R/R500'], info['metallicity']['core']['R/R500'].units)
#     print()

    for metal, values in info['metallicity']['core'].copy().items():
        if metal not in ['angle', 'radius', 'R/R500']:
            values['rbins_R500'] = rbins_R500


            
            
            
            
            
## Mao et al (2019)
## Core regions of 8 nearby groups from CHEERS measured with XMM-Newton, both RGS and EPIC/MOS
## RGS spectra from ~3.4' extraction region
## First-order (7-28 A = 0.44-1.77 keV) and second order (7-14 A = 0.89-1.77 keV) spectra
## Uses Lodders+09 solar abundances
## kT in keV from de Plaa+2017
## R500 in Mpc from Mernier+2017

obs_mao19 = {
    'individual':{
        'A3526':{
            'R500':0.83,
            'kT':3.7,
            'metallicity':{
                'core':{
                    'N/O':{
                        'rbins_R500':[0,0.026],
                        'value':[2.7],
                        'err_lo':[0.5],
                        'err_hi':[0.5],
                    },
                    'N/Fe':{
                        'rbins_R500':[0,0.026],
                        'value':[1.5],
                        'err_lo':[0.2],
                        'err_hi':[0.2],
                    },
                    'O/Fe':{
                        'rbins_R500':[0,0.026],
                        'value':[0.54],
                        'err_lo':[0.04],
                        'err_hi':[0.04],
                    },
                    'Ne/Fe':{
                        'rbins_R500':[0,0.026],
                        'value':[0.57],
                        'err_lo':[0.06],
                        'err_hi':[0.06],
                    },
                    'Mg/Fe':{
                        'rbins_R500':[0,0.026],
                        'value':[0.66],
                        'err_lo':[0.07],
                        'err_hi':[0.0],
                    },
                    'Fe':{
                        'rbins_R500':[0,0.026],
                        'value':[1.02],
                        'err_lo':[0.03],
                        'err_hi':[0.03],
                    },
                    'Ni/Fe':{
                        'rbins_R500':[0,0.026],
                        'value':[1.2],
                        'err_lo':[0.1],
                        'err_hi':[0.1],
                    },
                },
            },
        },
        'M49':{
            'R500':0.53,
            'kT':1.0,
            'metallicity':{
                'core':{
                    'N/O':{
                        'rbins_R500':[0,0.018],
                        'value':[2.7],
                        'err_lo':[1.0],
                        'err_hi':[1.0],
                    },
                    'N/Fe':{
                        'rbins_R500':[0,0.018],
                        'value':[1.6],
                        'err_lo':[0.6],
                        'err_hi':[0.6],
                    },
                    'O/Fe':{
                        'rbins_R500':[0,0.018],
                        'value':[0.59],
                        'err_lo':[0.10],
                        'err_hi':[0.10],
                    },
                    'Ne/Fe':{
                        'rbins_R500':[0,0.018],
                        'value':[0.66],
                        'err_lo':[0.17],
                        'err_hi':[0.17],
                    },
                    'Mg/Fe':{
                        'rbins_R500':[0,0.018],
                        'value':[0.79],
                        'err_lo':[0.19],
                        'err_hi':[0.19],
                    },
                    'Fe':{
                        'rbins_R500':[0,0.018],
                        'value':[1.50],
                        'err_lo':[0.12],
                        'err_hi':[0.12],
                    },
                    'Ni/Fe':{
                        'rbins_R500':[0,0.018],
                        'value':[1.8],
                        'err_lo':[0.5],
                        'err_hi':[0.5],
                    },
                },
            },
        },
        'M87':{
            'R500':0.75,
            'kT':1.7,
            'metallicity':{
                'core':{
                    'N/O':{
                        'rbins_R500':[0,0.012],
                        'value':[2.2],
                        'err_lo':[0.3],
                        'err_hi':[0.3],
                    },
                    'N/Fe':{
                        'rbins_R500':[0,0.012],
                        'value':[1.8],
                        'err_lo':[0.2],
                        'err_hi':[0.2],
                    },
                    'O/Fe':{
                        'rbins_R500':[0,0.012],
                        'value':[0.82],
                        'err_lo':[0.03],
                        'err_hi':[0.03],
                    },
                    'Ne/Fe':{
                        'rbins_R500':[0,0.012],
                        'value':[0.55],
                        'err_lo':[0.05],
                        'err_hi':[0.05],
                    },
                    'Mg/Fe':{
                        'rbins_R500':[0,0.012],
                        'value':[0.24],
                        'err_lo':[0.04],
                        'err_hi':[0.04],
                    },
                    'Fe':{
                        'rbins_R500':[0,0.012],
                        'value':[0.55],
                        'err_lo':[0.01],
                        'err_hi':[0.01],
                    },
                    'Ni/Fe':{
                        'rbins_R500':[0,0.012],
                        'value':[0.65],
                        'err_lo':[0.07],
                        'err_hi':[0.07],
                    },
                },
            },
        },
        'NGC4636':{
            'R500':0.35,
            'kT':0.8,
            'metallicity':{
                'core':{
                    'N/O':{
                        'rbins_R500':[0,0.022],
                        'value':[3.3],
                        'err_lo':[1.1],
                        'err_hi':[1.1],
                    },
                    'N/Fe':{
                        'rbins_R500':[0,0.022],
                        'value':[1.9],
                        'err_lo':[0.5],
                        'err_hi':[0.5],
                    },
                    'O/Fe':{
                        'rbins_R500':[0,0.022],
                        'value':[0.59],
                        'err_lo':[0.08],
                        'err_hi':[0.08],
                    },
                    'Ne/Fe':{
                        'rbins_R500':[0,0.022],
                        'value':[0.64],
                        'err_lo':[0.12],
                        'err_hi':[0.12],
                    },
                    'Mg/Fe':{
                        'rbins_R500':[0,0.022],
                        'value':[0.64],
                        'err_lo':[0.13],
                        'err_hi':[0.13],
                    },
                    'Fe':{
                        'rbins_R500':[0,0.022],
                        'value':[0.66],
                        'err_lo':[0.04],
                        'err_hi':[0.04],
                    },
                    'Ni/Fe':{
                        'rbins_R500':[0,0.022],
                        'value':[2.0],
                        'err_lo':[0.4],
                        'err_hi':[0.4],
                    },
                },
            },
        },
        'NGC4649':{
            'R500':0.53,
            'kT':0.8,
            'metallicity':{
                'core':{
                    'N/O':{
                        'rbins_R500':[0,0.015],
                        'value':[2.9],
                        'err_lo':[1.0],
                        'err_hi':[1.0],
                    },
                    'N/Fe':{
                        'rbins_R500':[0,0.015],
                        'value':[2.4],
                        'err_lo':[0.8],
                        'err_hi':[0.8],
                    },
                    'O/Fe':{
                        'rbins_R500':[0,0.015],
                        'value':[0.84],
                        'err_lo':[0.11],
                        'err_hi':[0.11],
                    },
                    'Ne/Fe':{
                        'rbins_R500':[0,0.015],
                        'value':[1.07],
                        'err_lo':[0.19],
                        'err_hi':[0.19],
                    },
                    'Mg/Fe':{
                        'rbins_R500':[0,0.015],
                        'value':[1.40],
                        'err_lo':[0.23],
                        'err_hi':[0.23],
                    },
                    'Fe':{
                        'rbins_R500':[0,0.015],
                        'value':[0.55],
                        'err_lo':[0.03],
                        'err_hi':[0.03],
                    },
                    'Ni/Fe':{
                        'rbins_R500':[0,0.015],
                        'value':[2.5],
                        'err_lo':[0.4],
                        'err_hi':[0.4],
                    },
                },
            },
        },
        'NGC5044':{
            'R500':0.56,
            'kT':1.1,
            'metallicity':{
                'core':{
                    'N/O':{
                        'rbins_R500':[0,0.034],
                        'value':[2.2],
                        'err_lo':[0.5],
                        'err_hi':[0.5],
                    },
                    'N/Fe':{
                        'rbins_R500':[0,0.034],
                        'value':[1.4],
                        'err_lo':[0.3],
                        'err_hi':[0.3],
                    },
                    'O/Fe':{
                        'rbins_R500':[0,0.034],
                        'value':[0.65],
                        'err_lo':[0.05],
                        'err_hi':[0.05],
                    },
                    'Ne/Fe':{
                        'rbins_R500':[0,0.034],
                        'value':[0.68],
                        'err_lo':[0.08],
                        'err_hi':[0.08],
                    },
                    'Mg/Fe':{
                        'rbins_R500':[0,0.034],
                        'value':[0.77],
                        'err_lo':[0.08],
                        'err_hi':[0.08],
                    },
                    'Fe':{
                        'rbins_R500':[0,0.034],
                        'value':[0.78],
                        'err_lo':[0.03],
                        'err_hi':[0.03],
                    },
                    'Ni/Fe':{
                        'rbins_R500':[0,0.034],
                        'value':[1.5],
                        'err_lo':[0.3],
                        'err_hi':[0.3],
                    },
                },
            },
        },
        'NGC5813':{
            'R500':0.44,
            'kT':0.5,
            'metallicity':{
                'core':{
                    'N/O':{
                        'rbins_R500':[0,0.031],
                        'value':[3.2],
                        'err_lo':[0.9],
                        'err_hi':[0.9],
                    },
                    'N/Fe':{
                        'rbins_R500':[0,0.031],
                        'value':[1.9],
                        'err_lo':[0.4],
                        'err_hi':[0.4],
                    },
                    'O/Fe':{
                        'rbins_R500':[0,0.031],
                        'value':[0.58],
                        'err_lo':[0.07],
                        'err_hi':[0.07],
                    },
                    'Ne/Fe':{
                        'rbins_R500':[0,0.031],
                        'value':[0.53],
                        'err_lo':[0.09],
                        'err_hi':[0.09],
                    },
                    'Mg/Fe':{
                        'rbins_R500':[0,0.031],
                        'value':[0.83],
                        'err_lo':[0.11],
                        'err_hi':[0.11],
                    },
                    'Fe':{
                        'rbins_R500':[0,0.031],
                        'value':[0.92],
                        'err_lo':[0.04],
                        'err_hi':[0.04],
                    },
#                     'Ni/Fe':{
#                         'rbins_R500':[0,0.031],
#                         'value':[1.5],
#                         'err_lo':[0.3],
#                         'err_hi':[0.3],
#                     },
                },
            },
        },
        'NGC5846':{
            'R500':0.36,
            'kT':0.8,
            'metallicity':{
                'core':{
                    'N/O':{
                        'rbins_R500':[0,0.036],
                        'value':[2.7],
                        'err_lo':[0.8],
                        'err_hi':[0.8],
                    },
                    'N/Fe':{
                        'rbins_R500':[0,0.036],
                        'value':[2.3],
                        'err_lo':[0.7],
                        'err_hi':[0.7],
                    },
                    'O/Fe':{
                        'rbins_R500':[0,0.036],
                        'value':[0.86],
                        'err_lo':[0.12],
                        'err_hi':[0.12],
                    },
                    'Ne/Fe':{
                        'rbins_R500':[0,0.036],
                        'value':[0.71],
                        'err_lo':[0.14],
                        'err_hi':[0.14],
                    },
                    'Mg/Fe':{
                        'rbins_R500':[0,0.036],
                        'value':[0.66],
                        'err_lo':[0.14],
                        'err_hi':[0.14],
                    },
                    'Fe':{
                        'rbins_R500':[0,0.036],
                        'value':[0.77],
                        'err_lo':[0.05],
                        'err_hi':[0.05],
                    },
                    'Ni/Fe':{
                        'rbins_R500':[0,0.036],
                        'value':[2.0],
                        'err_lo':[0.4],
                        'err_hi':[0.4],
                    },
                },
            },
        },
    },
}


for group, info in obs_mao19['individual'].items():
    for metal, values in info['metallicity']['core'].items():
        if metal not in ['angle', 'radius', 'R/R500']:
            values['rbins_R500'] = pnb.array.SimArray(values['rbins_R500'], units='')
            values['value'] = pnb.array.SimArray(values['value'], units='')
            values['err_lo'] = pnb.array.SimArray(values['err_lo'], units='')
            values['err_hi'] = pnb.array.SimArray(values['err_hi'], units='')
            
    for metal, values in info['metallicity']['core'].copy().items():
        if '/' in metal:
            metal_num = metal.split('/')[0]
            metal_denom = metal.split('/')[1]
            
            if metal_denom.lower() == 'fe':
                new_value = values['value'] * info['metallicity']['core'][metal_denom]['value']
                new_err_lo = new_value * np.sqrt( (values['err_lo']/values['value'])**2 + 
                                              (info['metallicity']['core'][metal_denom]['err_lo']/
                                               info['metallicity']['core'][metal_denom]['value'])**2 )
                new_err_hi = new_value * np.sqrt( (values['err_hi']/values['value'])**2 + 
                                              (info['metallicity']['core'][metal_denom]['err_hi']/
                                               info['metallicity']['core'][metal_denom]['value'])**2 )
                info['metallicity']['core'][metal_num] = {
                    'rbins_R500':values['rbins_R500'],
                    'value':new_value,
                    'err_lo':new_err_lo,
                    'err_hi':new_err_hi,
                }
                
    for metal, values in info['metallicity']['core'].copy().items():
        if '/' not in metal and metal.lower() != 'o':
            new_value = values['value'] / info['metallicity']['core']['O']['value']
            new_err_lo = new_value * np.sqrt( (values['err_lo']/values['value'])**2 + 
                                          (info['metallicity']['core']['O']['err_lo']/
                                           info['metallicity']['core']['O']['value'])**2 )
            new_err_hi = new_value * np.sqrt( (values['err_hi']/values['value'])**2 + 
                                          (info['metallicity']['core']['O']['err_hi']/
                                           info['metallicity']['core']['O']['value'])**2 )
            info['metallicity']['core'][metal+'/O'] = {
                'rbins_R500':values['rbins_R500'],
                'value':new_value,
                'err_lo':new_err_lo,
                'err_hi':new_err_hi,
            }

    info['R500'] = pnb.array.SimArray(info['R500'], units='Mpc')
    info['kT'] = pnb.array.SimArray(info['kT'], units='keV')
    info['M500'] = scale.M500_Tspec_simbac_N1024L100_renier(info['kT'])            
            
            
            
            
            
            
            
            
            
            
            
## Grange et al (2011)
## Core regions of 2 groups
## Includes Carbon abundances
## EPIC and RGS spectral fitting
## RGS spectra from extraction region of cross dispersion width 5' (radius ~2.5' ?)
## EPIC spectra from circular extraction region of radius 3'
## Uses Lodders+03 solar abundances (don't have, need to get!)
## kT in keV from de Plaa+2017
## R500 in Mpc from Mernier+2017

obs_grange11 = {
    'individual':{
        'NGC5044':{
            'z':0.009,
            'R500':0.56,
            'kT':1.1,
            'metallicity':{
                'core':{
                    'C/Fe':{ #RGS
                        'angles':[0,2.5],
                        'value':[1.5],
                        'err_lo':[0.3],
                        'err_hi':[0.3],
                    },
                    'N/Fe':{ #RGS
                        'angles':[0,2.5],
                        'value':[1.5],
                        'err_lo':[0.2],
                        'err_hi':[0.2],
                    },
                    'O/Fe':{ #RGS
                        'angles':[0,2.5],
                        'value':[0.60],
                        'err_lo':[0.03],
                        'err_hi':[0.03],
                    },
                    'Ne/Fe':{ #RGS
                        'angles':[0,2.5],
                        'value':[0.37],
                        'err_lo':[0.10],
                        'err_hi':[0.10],
                    },
                    'Mg/Fe':{ #RGS
                        'angles':[0,2.5],
                        'value':[0.75],
                        'err_lo':[0.05],
                        'err_hi':[0.05],
                    },
                    'Si/Fe':{ #EPIC
                        'angles':[0,3.0],
                        'value':[0.788],
                        'err_lo':[0.012],
                        'err_hi':[0.012],
                    },
                    'S/Fe':{ #EPIC
                        'angles':[0,3.0],
                        'value':[0.843],
                        'err_lo':[0.016],
                        'err_hi':[0.016],
                    },
                    'Ar/Fe':{ #EPIC
                        'angles':[0,3.0],
                        'value':[0.789],
                        'err_lo':[0.013],
                        'err_hi':[0.013],
                    },
                    'Ca/Fe':{ #EPIC
                        'angles':[0,3.0],
                        'value':[1.84],
                        'err_lo':[0.13],
                        'err_hi':[0.13],
                    },
                    'Fe':{ #RGS
                        'angles':[0,2.5],
                        'value':[0.616],
                        'err_lo':[0.016],
                        'err_hi':[0.016],
                    },
                    'Ni/Fe':{ #EPIC
                        'angles':[0,3.0],
                        'value':[1.60],
                        'err_lo':[0.05],
                        'err_hi':[0.05],
                    },
                },
            },
        },
        'NGC5813':{
            'z':0.009,
            'R500':0.44,
            'kT':0.5,
            'metallicity':{
                'core':{
                    'C/Fe':{ #RGS
                        'angles':[0,2.5],
                        'value':[1.7],
                        'err_lo':[0.4],
                        'err_hi':[0.4],
                    },
                    'N/Fe':{ #RGS
                        'angles':[0,2.5],
                        'value':[1.8],
                        'err_lo':[0.3],
                        'err_hi':[0.3],
                    },
                    'O/Fe':{ #RGS
                        'angles':[0,2.5],
                        'value':[0.73],
                        'err_lo':[0.04],
                        'err_hi':[0.04],
                    },
                    'Ne/Fe':{ #RGS
                        'angles':[0,2.5],
                        'value':[0.58],
                        'err_lo':[0.10],
                        'err_hi':[0.10],
                    },
                    'Mg/Fe':{ #RGS
                        'angles':[0,2.5],
                        'value':[0.86],
                        'err_lo':[0.06],
                        'err_hi':[0.06],
                    },
                    'Si/Fe':{ #EPIC
                        'angles':[0,3.0],
                        'value':[0.94],
                        'err_lo':[0.02],
                        'err_hi':[0.02],
                    },
                    'S/Fe':{ #EPIC
                        'angles':[0,3.0],
                        'value':[1.13],
                        'err_lo':[0.05],
                        'err_hi':[0.05],
                    },
                    'Ar/Fe':{ #EPIC
                        'angles':[0,3.0],
                        'value':[1.0],
                        'err_lo':[0.2],
                        'err_hi':[0.2],
                    },
                    'Ca/Fe':{ #EPIC
                        'angles':[0,3.0],
                        'value':[1.9],
                        'err_lo':[0.5],
                        'err_hi':[0.5],
                    },
                    'Fe':{ #RGS
                        'angles':[0,2.5],
                        'value':[0.538],
                        'err_lo':[0.017],
                        'err_hi':[0.017],
                    },
                    'Ni/Fe':{ #EPIC
                        'angles':[0,3.0],
                        'value':[0.54],
                        'err_lo':[0.09],
                        'err_hi':[0.09],
                    },
                },
            },
        },
    },
}


for group, info in obs_grange11['individual'].items():
    info['R500'] = pnb.array.SimArray(info['R500'], units='Mpc')
    info['kT'] = pnb.array.SimArray(info['kT'], units='keV')
    info['M500'] = scale.M500_Tspec_simbac_N1024L100_renier(info['kT'])
    
    cosmo = ned_wright_cosmo_calc.cosmo(H0=pnb.array.SimArray(68, units='km s**-1 Mpc**-1'), Omega_M=0.3, Omega_Vac=0.7, z=info['z'])
    sky_scale = cosmo.kpc_DA#.in_units('Mpc arcmin**-1')
    
    for metal, values in info['metallicity']['core'].items():        
        values['angles'] = pnb.array.SimArray(values['angles'], units='arcmin')
        values['rbins_R500'] = (sky_scale * values['angles']/info['R500']).in_units('')
        
        values['value'] = pnb.array.SimArray(values['value'], units='')
        values['err_lo'] = pnb.array.SimArray(values['err_lo'], units='')
        values['err_hi'] = pnb.array.SimArray(values['err_hi'], units='')
            
    for metal, values in info['metallicity']['core'].copy().items():
        if '/' in metal:
            metal_num = metal.split('/')[0]
            metal_denom = metal.split('/')[1]
            
            if metal_denom.lower() == 'fe':
                new_value = values['value'] * info['metallicity']['core'][metal_denom]['value']
                new_err_lo = new_value * np.sqrt( (values['err_lo']/values['value'])**2 + 
                                              (info['metallicity']['core'][metal_denom]['err_lo']/
                                               info['metallicity']['core'][metal_denom]['value'])**2 )
                new_err_hi = new_value * np.sqrt( (values['err_hi']/values['value'])**2 + 
                                              (info['metallicity']['core'][metal_denom]['err_hi']/
                                               info['metallicity']['core'][metal_denom]['value'])**2 )
                info['metallicity']['core'][metal_num] = {
                    'rbins_R500':values['rbins_R500'],
                    'value':new_value,
                    'err_lo':new_err_lo,
                    'err_hi':new_err_hi,
                }
                
    for metal, values in info['metallicity']['core'].copy().items():
        if '/' not in metal and metal.lower() != 'o':
            new_value = values['value'] / info['metallicity']['core']['O']['value']
            new_err_lo = new_value * np.sqrt( (values['err_lo']/values['value'])**2 + 
                                          (info['metallicity']['core']['O']['err_lo']/
                                           info['metallicity']['core']['O']['value'])**2 )
            new_err_hi = new_value * np.sqrt( (values['err_hi']/values['value'])**2 + 
                                          (info['metallicity']['core']['O']['err_hi']/
                                           info['metallicity']['core']['O']['value'])**2 )
            info['metallicity']['core'][metal+'/O'] = {
                'rbins_R500':values['rbins_R500'],
                'value':new_value,
                'err_lo':new_err_lo,
                'err_hi':new_err_hi,
            }

            
            
            
            
            
            
## Werner et al (2006)
## Core region of M87
## Includes Carbon abundances
## RGS spectral fitting
## RGS spectra from spatially resolved extraction regions
## Uses Lodders+03 solar abundances (don't have, need to get!)
## kT in keV from de Plaa+2017
## R500 in Mpc from Mernier+2017

obs_werner06 = {
    'individual':{
        'M87':{
            'z':0.0042,
            'R500':0.75,
            'kT':1.7,
            'metallicity':{
                'profiles':{
                    'C':{
                        'angles':[0,0.5,1.5,2.5,3.5],
                        'value':[0.63,0.44,0.30,0.18],
                        'err_lo':[0.16,0.13,0.16,0.18],
                        'err_hi':[0.16,0.13,0.16,0.0],
                    },
                    'N':{
                        'angles':[0,0.5,1.5,2.5,3.5],
                        'value':[1.64,0.62,0.67,0.23],
                        'err_lo':[0.24,0.18,0.22,0.23],
                        'err_hi':[0.24,0.18,0.22,0.0],
                    },
                    'O':{
                        'angles':[0,0.5,1.5,2.5,3.5],
                        'value':[0.58,0.48,0.52,0.45],
                        'err_lo':[0.03,0.02,0.03,0.03],
                        'err_hi':[0.03,0.02,0.03,0.03],
                    },
                    'Ne':{
                        'angles':[0,0.5,1.5,2.5,3.5],
                        'value':[1.41,1.17,0.89,0.23],
                        'err_lo':[0.12,0.13,0.17,0.23],
                        'err_hi':[0.12,0.13,0.17,0.0],
                    },
                    'Fe':{
                        'angles':[0,0.5,1.5,2.5,3.5],
                        'value':[0.95,0.78,0.65,0.50],
                        'err_lo':[0.03,0.03,0.03,0.03],
                        'err_hi':[0.03,0.03,0.03,0.03],
                    },
                },
            },
        },
    },
}


for group, info in obs_werner06['individual'].items():
    info['R500'] = pnb.array.SimArray(info['R500'], units='Mpc')
    info['kT'] = pnb.array.SimArray(info['kT'], units='keV')
    info['M500'] = scale.M500_Tspec_simbac_N1024L100_renier(info['kT'])
    
    cosmo = ned_wright_cosmo_calc.cosmo(H0=pnb.array.SimArray(68, units='km s**-1 Mpc**-1'), Omega_M=0.3, Omega_Vac=0.7, z=info['z'])
    sky_scale = cosmo.kpc_DA#.in_units('Mpc arcmin**-1')
    
    for metal, values in info['metallicity']['profiles'].items():        
        values['angles'] = pnb.array.SimArray(values['angles'], units='arcmin')
        values['rbins_R500'] = (sky_scale * values['angles']/info['R500']).in_units('')
        
        values['value'] = pnb.array.SimArray(values['value'], units='')
        values['err_lo'] = pnb.array.SimArray(values['err_lo'], units='')
        values['err_hi'] = pnb.array.SimArray(values['err_hi'], units='')
            
    for metal_num, values in info['metallicity']['profiles'].copy().items():

        if metal_num.lower() != 'fe':
            new_value = values['value'] / info['metallicity']['profiles']['Fe']['value']
            new_err_lo = new_value * np.sqrt( (values['err_lo']/values['value'])**2 + 
                                     (info['metallicity']['profiles']['Fe']['err_lo']/
                                     info['metallicity']['profiles']['Fe']['value'])**2 )
            new_err_hi = new_value * np.sqrt( (values['err_hi']/values['value'])**2 + 
                                     (info['metallicity']['profiles']['Fe']['err_hi']/
                                     info['metallicity']['profiles']['Fe']['value'])**2 )
            
            info['metallicity']['profiles'][metal_num+'/Fe'] = {
                'rbins_R500':values['rbins_R500'],
                'value':new_value,
                'err_lo':new_err_lo,
                'err_hi':new_err_hi,
            }
            
        if metal_num.lower() != 'o':
            new_value = values['value'] / info['metallicity']['profiles']['O']['value']
            new_err_lo = new_value * np.sqrt( (values['err_lo']/values['value'])**2 + 
                                     (info['metallicity']['profiles']['O']['err_lo']/
                                     info['metallicity']['profiles']['O']['value'])**2 )
            new_err_hi = new_value * np.sqrt( (values['err_hi']/values['value'])**2 + 
                                     (info['metallicity']['profiles']['O']['err_hi']/
                                     info['metallicity']['profiles']['O']['value'])**2 )
            
            info['metallicity']['profiles'][metal_num+'/O'] = {
                'rbins_R500':values['rbins_R500'],
                'value':new_value,
                'err_lo':new_err_lo,
                'err_hi':new_err_hi,
            }
            
            
            
            
            
            
            
            
            
            
            

            
## Ghizzardi+2021 X-COP sample
## Anders & Grevesse (1989) solar abundances
## Mean abundance profiles from spectral fitting for total sample, and CC and NCC subsamples
## X-COP = XMM-Newton Cluster Outskirts Project
## EPIC MOS and pn instruments
## 0.5-12 keV for spectral fitting

obs_ghizzardi21 = {
    'average':{
#         'metallicity':{
        'full_sample':{
            'label':'Ghizzardi+21',
            'metallicity':{
                'Fe':{
                    'rbins_R500':[0,0.025,0.050,0.075,0.150,0.225,0.300,0.375,0.450,0.525,0.675,0.875,1.120],
                    'value':[0.578,0.432,0.371,0.317,0.276,0.243,0.236,0.245,0.252,0.250,0.240,0.200],  # Mean abundance
                    'err_lo':[0.193,0.103,0.066,0.052,0.042,0.046,0.042,0.054,0.064,0.053,0.047,0.076], # Total scatter
                    'err_hi':[0.193,0.103,0.066,0.052,0.042,0.046,0.042,0.054,0.064,0.053,0.047,0.076],
                },
            },
        },
        'CC_subsample':{
            'label':'Ghizzardi+21 CC',
            'metallicity':{
                'Fe':{
                    'rbins_R500':[0,0.025,0.050,0.075,0.150,0.225,0.300,0.375,0.450,0.525,0.675,0.875,1.120],
                    'value':[0.648,0.475,0.408,0.342,0.291,0.259,0.236,0.252,0.237,0.222,0.231,0.316],  # Mean abundance
                    'err_lo':[0.158,0.086,0.040,0.037,0.024,0.044,0.044,0.069,0.065,0.028,0.041,0.081], # Total scatter
                    'err_hi':[0.158,0.086,0.040,0.037,0.024,0.044,0.044,0.069,0.065,0.028,0.041,0.081],
                },
            },
        },
        'NCC_subsample':{
            'label':'Ghizzardi+21 NCC',
            'metallicity':{
                'Fe':{
                    'rbins_R500':[0,0.025,0.050,0.075,0.150,0.225,0.300,0.375,0.450,0.525,0.675,0.875,1.120],
                    'value':[0.330,0.325,0.297,0.276,0.258,0.231,0.236,0.242,0.259,0.263,0.243,0.170],  # Mean abundance
                    'err_lo':[0.048,0.048,0.038,0.046,0.051,0.045,0.041,0.046,0.062,0.056,0.049,0.035], # Total scatter
                    'err_hi':[0.048,0.048,0.038,0.046,0.051,0.045,0.041,0.046,0.062,0.056,0.049,0.035],
                },
            },
        },
    },
}



for sample, sample_info in obs_ghizzardi21['average'].copy().items():
    for metal, metal_info in sample_info['metallicity'].items():
        obs_ghizzardi21['average'][sample]['metallicity'][metal]['rbins_R500'] = pnb.array.SimArray(metal_info['rbins_R500'], units='')
        obs_ghizzardi21['average'][sample]['metallicity'][metal]['value'] = pnb.array.SimArray(metal_info['value'], units='')
        obs_ghizzardi21['average'][sample]['metallicity'][metal]['err_lo'] = pnb.array.SimArray(metal_info['err_lo'], units='')
        obs_ghizzardi21['average'][sample]['metallicity'][metal]['err_hi'] = pnb.array.SimArray(metal_info['err_hi'], units='')

            
            
            
            
            
            
            
            
            

         
        
        
## Mernier+2018b CHEERS sample
## Lodders+09 solar abundances
## Mean core abundances within 0.05R500 and 0.2R500 (only for hotter groups for which it could be measured)
## (so I will use 0.05R500, since all groups have it, and in Mernier+2016a, they showed the difference is negligible)
## Same as Mernier+2016a, but with updated spectral code
## Significantly affects Cr/Fe and Ni/Fe metallicities
## Like Mernier+2017, uses XMM-Newton/EPIC MOS 0.5/0.6-10 keV for global fits

obs_mernier18b = {
    'average':{
        'metallicity':{
            'O/Fe':{
                'rbins_R500':[0,0.05],
                'value':[0.817],
                'err_lo':[0.175],
                'err_hi':[0.175],
            },
            'Ne/Fe':{
                'rbins_R500':[0,0.05],
                'value':[0.724],
                'err_lo':[0.133],
                'err_hi':[0.133],
            },
            'Mg/Fe':{
                'rbins_R500':[0,0.05],
                'value':[0.937],
                'err_lo':[0.072],
                'err_hi':[0.072],
            },
            'Si/Fe':{
                'rbins_R500':[0,0.05],
                'value':[0.949],
                'err_lo':[0.061],
                'err_hi':[0.061],
            },
            'S/Fe':{
                'rbins_R500':[0,0.05],
                'value':[1.004],
                'err_lo':[0.021],
                'err_hi':[0.021],
            },
            'Ar/Fe':{
                'rbins_R500':[0,0.05],
                'value':[0.980],
                'err_lo':[0.085],
                'err_hi':[0.085],
            },
            'Ca/Fe':{
                'rbins_R500':[0,0.05],
                'value':[1.272],
                'err_lo':[0.103],
                'err_hi':[0.103],
            },
            'Cr/Fe':{
                'rbins_R500':[0,0.05],
                'value':[0.986],
                'err_lo':[0.188],
                'err_hi':[0.188],
            },
            'Mn/Fe':{
                'rbins_R500':[0,0.05],
                'value':[1.557],
                'err_lo':[0.774],
                'err_hi':[0.774],
            },
            'Ni/Fe':{
                'rbins_R500':[0,0.05],
                'value':[0.959],
                'err_lo':[0.382],
                'err_hi':[0.382],
            },
        },
    },
}



for metal, info in obs_mernier18b['average']['metallicity'].copy().items():
    obs_mernier18b['average']['metallicity'][metal]['rbins_R500'] = pnb.array.SimArray(info['rbins_R500'], units='')
    obs_mernier18b['average']['metallicity'][metal]['value'] = pnb.array.SimArray(info['value'], units='')
    obs_mernier18b['average']['metallicity'][metal]['err_lo'] = pnb.array.SimArray(info['err_lo'], units='')
    obs_mernier18b['average']['metallicity'][metal]['err_hi'] = pnb.array.SimArray(info['err_hi'], units='')
    
    if metal.lower() != 'o/fe':
        metal_num = metal.split('/')[0]
        new_value = info['value'] / obs_mernier18b['average']['metallicity']['O/Fe']['value']
        new_err_lo = new_value * np.sqrt( (info['err_lo']/info['value'])**2 + 
                                      (obs_mernier18b['average']['metallicity']['O/Fe']['err_lo']/
                                       obs_mernier18b['average']['metallicity']['O/Fe']['value'])**2 )
        new_err_hi = new_value * np.sqrt( (info['err_hi']/info['value'])**2 + 
                                      (obs_mernier18b['average']['metallicity']['O/Fe']['err_hi']/
                                       obs_mernier18b['average']['metallicity']['O/Fe']['value'])**2 )
        obs_mernier18b['average']['metallicity'][metal_num+'/O'] = {
            'rbins_R500':info['rbins_R500'],
            'value':new_value,
            'err_lo':new_err_lo,
            'err_hi':new_err_hi,
        }
        
        
        
        
            
            



## Mernier et al (2017) (Lodders+09 solar abundances?)
auth = 'Mernier17'
sub_dirs = {
    'median':'avg_profiles',
    'individual':'CHEERS_indiv_profiles',
}
curr_dirs = {name:os.path.join(data_dir, auth, sub_dir) for name, sub_dir in sub_dirs.items()}
# for name, curr_dir in curr_dirs.items():
# #     print(name)
#     os.listdir(curr_dir)
    
    
obs_mernier17 = {
    'median':{
        prop.stem:np.genfromtxt(prop, comments='#', skip_header=21 if prop.stem=='Mernier_2017_group' else 23, names=None) for prop in [Path(x) for x in glob(os.path.join(curr_dirs['median'], '*'))]
    },
    'individual':{
#         'Abundance':{
#             element=='Fe' and element:{
#                 str(os.path.basename(file)).split('_')[0]:np.genfromtxt(file, skip_header=47 if element=='Fe' else 25, max_rows=9) for file in [Path(x) for x in glob(os.path.join(curr_dirs['individual'], element, '*'))]  # For Fe profile use Global" MOS+pn; for ratios use Local:MOS or Local:pn
#             } for element in os.listdir(curr_dirs['individual'])
#         },
#         'Abundance Ratio':{
#             element+'/Fe' if element!='Fe' else None:{
#                 str(os.path.basename(file)).split('_')[0]:np.genfromtxt(file, skip_header=47 if element=='Fe' else 25, max_rows=9) for file in [Path(x) for x in glob(os.path.join(curr_dirs['individual'], element, '*'))]  # For Fe profile use Global" MOS+pn; for ratios use Local:MOS or Local:pn
#             } for element in os.listdir(curr_dirs['individual'])
#         }
        'metallicity':{},
        'metallicity ratio':{},
    },
    'instruments':{
        'mos':'Local MOS',
        'pn':'Local pn',
        'combined':'Combined MOS+pn',
        'global':'Global MOS+pn',
        'both':'Local MOS and pn',
    },
    'group_properties':{},
}

## Get the abundance and abundance ratio data from the files
for element in os.listdir(curr_dirs['individual']):
    ## XMM-Newton/EPIC metallicities
    ## For Fe profile use Global MOS+pn
    ## For ratios use Local:MOS and Local:pn (ie. average them according to eqn 4 in Mernier+2017)
    
    element_name = element
    element_type = 'metallicity'
    
    if element.lower() != 'fe':
        element_name = element+'/Fe'
        element_type = 'metallicity ratio'
        
    mos_row = 25
    pn_row = 36
    global_row = 47
    if element.lower() == 'o':
#         print(element)
#         print('O')
#         print()
        pn_row = 25
        global_row = 36
        
    for instrument in obs_mernier17['instruments'].keys():
        obs_mernier17['individual'][element_type][element_name+'_'+instrument] = {}
    
    for file in [Path(x) for x in glob(os.path.join(curr_dirs['individual'], element, '*'))]:
        galaxy_group = str(os.path.basename(file)).split('_')[0]
        
        local_mos = np.genfromtxt(file, skip_header=mos_row, max_rows=9)
        local_pn = np.genfromtxt(file, skip_header=pn_row, max_rows=9)
        global_mos_pn = np.genfromtxt(file, skip_header=global_row, max_rows=9)
        
        
        local_mos_mean_sigma = np.mean(np.abs(local_mos[:,3:]), axis=1)
        local_pn_mean_sigma = np.mean(np.abs(local_pn[:,3:]), axis=1)

#         combined_mos_pn = np.zeros(np.shape(local_mos), float)
        combined_mos_pn = local_mos.copy()
        combined_mos_pn[:,2] = ( (local_mos[:,2]/local_mos_mean_sigma**2) + 
                                (local_pn[:,2]/local_pn_mean_sigma**2) )/( local_mos_mean_sigma**(-2) + local_pn_mean_sigma**(-2) )

        ## Calculate error bars of combined mos+pn abundances as encompassing both mos and pn uncertainties separately
        local_mos_upper_bounds = local_mos[:,2] + local_mos[:,3]
        local_pn_upper_bounds = local_pn[:,2] + local_pn[:,3]
        mos_and_pn_upper_bounds = np.array([local_mos_upper_bounds, local_pn_upper_bounds])
        combined_mos_pn_upper_bounds = np.max(mos_and_pn_upper_bounds, axis=0)
        combined_mos_pn[:,3] = np.abs(combined_mos_pn[:,2] - combined_mos_pn_upper_bounds)

        local_mos_lower_bounds = local_mos[:,2] - np.abs(local_mos[:,4])
        local_pn_lower_bounds = local_pn[:,2] - np.abs(local_pn[:,4])
        mos_and_pn_lower_bounds = np.array([local_mos_lower_bounds, local_pn_lower_bounds])
        combined_mos_pn_lower_bounds = np.min(mos_and_pn_lower_bounds, axis=0)
        combined_mos_pn[:,4] = -np.abs(combined_mos_pn[:,2] - combined_mos_pn_lower_bounds)

        # Alternatively, just use error propagation, as the above results in large errorbars(?)
#         combined_mos_pn[:,3] = ( ((local_mos[:,3] * local_pn[:,3])**2)/(local_mos[:,3]**2 + local_pn[:,3]**2) ) * np.sqrt((local_mos[:,2]/(local_mos[:,3]**2))**2 + (local_pn[:,2]/(local_pn[:,3]**2))**2)
#         combined_mos_pn[:,4] = ( ((local_mos[:,4] * local_pn[:,4])**2)/(local_mos[:,4]**2 + local_pn[:,4]**2) ) * np.sqrt((local_mos[:,2]/(local_mos[:,4]**2))**2 + (local_pn[:,2]/(local_pn[:,4]**2))**2)


        obs_mernier17['individual'][element_type][element_name+'_mos'][galaxy_group] = local_mos
        obs_mernier17['individual'][element_type][element_name+'_pn'][galaxy_group] = local_pn
        obs_mernier17['individual'][element_type][element_name+'_combined'][galaxy_group] = combined_mos_pn
        obs_mernier17['individual'][element_type][element_name+'_global'][galaxy_group] = global_mos_pn
        obs_mernier17['individual'][element_type][element_name+'_both'][galaxy_group+'_mos'] = local_mos
        obs_mernier17['individual'][element_type][element_name+'_both'][galaxy_group+'_pn'] = local_pn
        
        
    
#     if element == 'Fe':
#         obs_mernier17['individual']['metallicity'][element] = {}
#         for file in [Path(x) for x in glob(os.path.join(curr_dirs['individual'], element, '*'))]:
#             local_mos = np.genfromtxt(file, skip_header=25, max_rows=9)
#             local_pn = np.genfromtxt(file, skip_header=36, max_rows=9)
#             global_mos_pn = np.genfromtxt(file, skip_header=47, max_rows=9)
#             obs_mernier17['individual']['metallicity'][element][str(os.path.basename(file)).split('_')[0]] = global_mos_pn
    
#     else:
#         ratio = element+'/Fe'
#         obs_mernier17['individual']['metallicity ratio'][ratio] = {}
#         for file in [Path(x) for x in glob(os.path.join(curr_dirs['individual'], element, '*'))]:
#             local_mos = np.genfromtxt(file, skip_header=25, max_rows=9)
#             local_pn = np.genfromtxt(file, skip_header=36, max_rows=9)
            
#             local_mos_mean_sigma = np.mean(np.abs(local_mos[:,3:]), axis=1)
#             local_pn_mean_sigma = np.mean(np.abs(local_pn[:,3:]), axis=1)
            
# #             combined_mos_pn = np.zeros(np.shape(local_mos), float)
#             combined_mos_pn = local_mos.copy()
#             combined_mos_pn[:,2] = ( (local_mos[:,2]/local_mos_mean_sigma**2) + 
#                                     (local_pn[:,2]/local_pn_mean_sigma**2) )/( local_mos_mean_sigma**(-2) + local_pn_mean_sigma**(-2) )
            
#             ## Calculate error bars of combined mos+pn abundances as encompassing both mos and pn uncertainties separately
#             local_mos_upper_bounds = local_mos[:,2] + local_mos[:,3]
#             local_pn_upper_bounds = local_pn[:,2] + local_pn[:,3]
#             mos_and_pn_upper_bounds = np.array([local_mos_upper_bounds, local_pn_upper_bounds])
#             combined_mos_pn_upper_bounds = np.max(mos_and_pn_upper_bounds, axis=0)
#             combined_mos_pn[:,3] = np.abs(combined_mos_pn[:,2] - combined_mos_pn_upper_bounds)
            
#             local_mos_lower_bounds = local_mos[:,2] - np.abs(local_mos[:,4])
#             local_pn_lower_bounds = local_pn[:,2] - np.abs(local_pn[:,4])
#             mos_and_pn_lower_bounds = np.array([local_mos_lower_bounds, local_pn_lower_bounds])
#             combined_mos_pn_lower_bounds = np.min(mos_and_pn_lower_bounds, axis=0)
#             combined_mos_pn[:,4] = -np.abs(combined_mos_pn[:,2] - combined_mos_pn_lower_bounds)
            
#             ## Alternatively, just use error propagation, as the above results in large errorbars(?)
# #             combined_mos_pn[:,3] = ( ((local_mos[:,3] * local_pn[:,3])**2)/(local_mos[:,3]**2 + local_pn[:,3]**2) ) * np.sqrt((local_mos[:,2]/(local_mos[:,3]**2))**2 + (local_pn[:,2]/(local_pn[:,3]**2))**2)
# #             combined_mos_pn[:,4] = ( ((local_mos[:,4] * local_pn[:,4])**2)/(local_mos[:,4]**2 + local_pn[:,4]**2) ) * np.sqrt((local_mos[:,2]/(local_mos[:,4]**2))**2 + (local_pn[:,2]/(local_pn[:,4]**2))**2)
            
    
#             obs_mernier17['individual']['metallicity ratio'][ratio+'_mos'][str(os.path.basename(file)).split('_')[0]] = local_mos
#             obs_mernier17['individual']['metallicity ratio'][ratio+'_pn'][str(os.path.basename(file)).split('_')[0]] = local_pn
#             obs_mernier17['individual']['metallicity ratio'][ratio+'_combined'][str(os.path.basename(file)).split('_')[0]] = combined_mos_pn
    
#     obs_mernier17['individual']['metallicity' if element=='Fe' else 'metallicity ratio'][element if element=='Fe' else element+'/Fe'] = {
#         str(os.path.basename(file)).split('_')[0]:np.genfromtxt(file, skip_header=47 if element=='Fe' else 25, max_rows=9) for file in [Path(x) for x in glob(os.path.join(curr_dirs['individual'], element, '*'))]  # For Fe profile use Global" MOS+pn; for ratios use Local:MOS or Local:pn
#     }

#         element if element!='Fe' else element+'/Fe':{
# #             file.split('_')[0]:Table.read(file, format='ascii.rdb', guess=False, fast_reader=False) for file in [Path(x) for x in glob(os.path.join(curr_dirs['individual'], element, '*'))]  # For Fe profile use Global" MOS+pn; for ratios use Local:MOS or Local:pn
# #             file.split('_')[0]:Table.read(file, format='pandas.csv', table_id=2 if element=='Fe' else 0, names=['R', 'abund' if element=='Fe' else 'abund_ratio']) for file in [Path(x) for x in glob(os.path.join(curr_dirs['individual'], element, '*'))]  # For Fe profile use Global" MOS+pn; for ratios use Local:MOS or Local:pn
# #             file.split('_')[0]:Table.read(file, format='ascii.qdp', table_id=2 if element=='Fe' else 0, names=['R', 'abund' if element=='Fe' else 'abund_ratio'], guess=False, fast_reader=False) for file in [Path(x) for x in glob(os.path.join(curr_dirs['individual'], element, '*'))]  # For Fe profile use Global" MOS+pn; for ratios use Local:MOS or Local:pn
#             str(os.path.basename(file)).split('_')[0]:np.genfromtxt(file, skip_header=47 if element=='Fe' else 25, max_rows=9) for file in [Path(x) for x in glob(os.path.join(curr_dirs['individual'], element, '*'))]  # For Fe profile use Global" MOS+pn; for ratios use Local:MOS or Local:pn
#         } for element in os.listdir(curr_dirs['individual'])

# np.genfromtxt(file, skip_header=47 if element=='Fe' else 25, skip_footer=0 if element=='Fe' else 22)
# np.genfromtxt(file, skip_header=47 if element=='Fe' else 25, max_rows=9)

# file.split('_')[0]:Table.read(file, format='ascii.rdb', guess=False, fast_reader=False) for file in [Path(x) for x in glob(os.path.join(curr_dirs['individual'], element, '*'))]  # For Fe profile use Global" MOS+pn; for ratios use Local:MOS or Local:pn
# file.split('_')[0]:Table.read(file, format='pandas.csv', table_id=2 if element=='Fe' else 0, names=['R', 'abund' if element=='Fe' else 'abund_ratio']) for file in [Path(x) for x in glob(os.path.join(curr_dirs['individual'], element, '*'))]  # For Fe profile use Global" MOS+pn; for ratios use Local:MOS or Local:pn
# file.split('_')[0]:Table.read(file, format='ascii.qdp', table_id=2 if element=='Fe' else 0, names=['R', 'abund' if element=='Fe' else 'abund_ratio'], guess=False, fast_reader=False) for file in [Path(x) for x in glob(os.path.join(curr_dirs['individual'], element, '*'))]  # For Fe profile use Global" MOS+pn; for ratios use Local:MOS or Local:pn

obs_mernier17['median']['group_element_idx'] = {'Fe':2, 'O':4, 'Mg':6, 'Si':8, 'S':10, 'Ar':12, 'Ca':14}
obs_mernier17['median']['cluster_element_idx'] = {'Fe':2, 'O':4, 'Mg':6, 'Si':8, 'S':10, 'Ar':12, 'Ca':14, 'Ni':16}


## Calculate the rest of the metal abundances from the Fe abundance and X/Fe abundance ratios
for element, groups in obs_mernier17['individual']['metallicity ratio'].items():
#     print()
#     print(element)
#     print(groups.keys())
#     print()
    
#     if element != 'Fe':
    element_num = element.split('/')[0]
    element_denom = element.split('_')[0].split('/')[1]
    instrument = element.split('_')[1]
#     print(element_num)
#     print(element_denom)
#     print(instrument)
#     print()
    new_element = element_num + '_' + instrument
#     print(new_element)
    obs_mernier17['individual']['metallicity'][new_element] = {}
#     print()

    for group, vals in groups.items():
#         print(group)
        
        r_centre = vals[:,0]
#         print(r_centre)
        r_err = vals[:,1]
    
        if element_denom.lower() in ['fe']:
            try:
                group_ = group.split('_')[0]
            except:
                group_ = group

#             print(group_)
#             print()
            element_denom_abund = obs_mernier17['individual']['metallicity']['Fe_global'][group_]
        else:
            element_denom_abund = obs_mernier17['individual']['metallicity'][element_denom+'_'+instrument][group]
        
        abund = vals[:,2] * element_denom_abund[:,2]
        abund_err_hi = abund * np.sqrt((vals[:,3]/vals[:,2])**2 + (element_denom_abund[:,3]/element_denom_abund[:,2])**2)
        abund_err_lo = abund * np.sqrt((vals[:,4]/vals[:,2])**2 + (element_denom_abund[:,4]/element_denom_abund[:,2])**2)

#         new_vals = np.hstack((r_centre, r_err, abund, abund_err_hi, abund_err_lo))
        new_vals = np.transpose(np.array([r_centre, r_err, abund, abund_err_hi, abund_err_lo]))

        obs_mernier17['individual']['metallicity'][new_element][group] = new_vals
        
#         obs_mernier17['individual'][element.split('/')[0]] = {
#             group:abund_ratio * obs_mernier17['individual']['Fe'][group] for group, abund_ratio in groups.items()
#         }



# print()
# print()
# # print(obs_mernier17)
# print(obs_mernier17['individual']['metallicity ratio']['Mg/Fe_both'])
# print()
# print(obs_mernier17['individual']['metallicity']['Mg_both'])






## Calculate all other metallicity ratios
# print('CHEERS (Mernier+2017):')
# print()
for element_num_, groups_num in obs_mernier17['individual']['metallicity'].items():
    element_num = element_num_.split('_')[0]
    instrument_num = element_num_.split('_')[1]
#     print('numerator:', element_num)
#     print('numerator instrument:', instrument_num)
#     print()
    for element_denom_, groups_denom in obs_mernier17['individual']['metallicity'].items():
        element_denom = element_denom_.split('_')[0]
        instrument_denom = element_denom_.split('_')[1]
#         print('denominator:', element_denom)
#         print('denominator instrument:', instrument_denom)
#         print()
        
        if instrument_num == instrument_denom:
            instrument_ratio = instrument_num
            element_ratio = element_num + '/' + element_denom
            element_ratio_ = element_ratio + '_' + instrument_ratio
            
            common_groups = list(filter(lambda group:group in groups_num, groups_denom))

            if element_ratio_ not in obs_mernier17['individual']['metallicity ratio'].keys():
                obs_mernier17['individual']['metallicity ratio'][element_ratio_] = {}
                for group in common_groups:
#                     print('group:', group)

                    profile_num = obs_mernier17['individual']['metallicity'][element_num_][group]
                    profile_denom = obs_mernier17['individual']['metallicity'][element_denom_][group]

#                     print(np.shape(profile_num))
#                     print(np.shape(profile_denom))

                    obs_mernier17['individual']['metallicity ratio'][element_ratio_][group] = profile_num.copy()

                    metallicity_num = profile_num[:,2].copy()
                    metallicity_err_hi_num = np.abs(profile_num[:,3].copy())
                    metallicity_err_lo_num = np.abs(profile_num[:,4].copy())

                    metallicity_denom = profile_denom[:,2].copy()
                    metallicity_err_hi_denom = np.abs(profile_denom[:,3].copy())
                    metallicity_err_lo_denom = np.abs(profile_denom[:,4].copy())


                    metallicity_ratio = metallicity_num/metallicity_denom
#                     print(metallicity_ratio)
                    metallicity_ratio_err_hi = metallicity_ratio * np.sqrt((metallicity_err_hi_num/metallicity_num)**2 +
                                                                           (metallicity_err_hi_denom/metallicity_denom)**2)
                    metallicity_ratio_err_lo = metallicity_ratio * np.sqrt((metallicity_err_lo_num/metallicity_num)**2 +
                                                                           (metallicity_err_lo_denom/metallicity_denom)**2)

                    obs_mernier17['individual']['metallicity ratio'][element_ratio_][group][:,2] = metallicity_ratio
                    obs_mernier17['individual']['metallicity ratio'][element_ratio_][group][:,3] = metallicity_ratio_err_hi
                    obs_mernier17['individual']['metallicity ratio'][element_ratio_][group][:,4] = -metallicity_ratio_err_lo

#             print()

#     print()






# Higher the closeness, the more similar it is to how Tspec,corr is measured in Simba/Simba-C
obs_mernier17['group_properties']['obs_samples'] = {
    'dePlaa+2017':{
        'year':2017,
        'closeness':1,
    },
    'RB2002':{
        'year':2002,
        'closeness':0.2,
    },
    'Sun2009':{
        'year':2009,
        'closeness':0.2,
    },
    'ACCEPT':{
        'year':2009,
        'closeness':0.2,
    },
    'Ponman+1996':{
        'year':1996,
        'closeness':0.2,
    },
    'CLoGS':{
        'year':2017,
        'closeness':0.3,
    },
    'Heldson+2002':{
        'year':2002,
        'closeness':0.2,
    },
    'Eckmiller+2011':{
        'year':2011,
        'closeness':0.2,
    },
    'Kettula+2013':{
        'year':2013,
        'closeness':0.05,
    },
    'Lovisari+2015':{
        'year':2015,
        'closeness':0.2,
    },
}


## Quality score of each sample based on how new it is and how similar its method of 
## calculating temperature is to Simba/Simba-C's method of calculating Tspec,corr
## The lower the score, the better
current_year = 2024
def quality_score(year, closeness, current_year):
    year_diff = np.abs(year - current_year)/10.
    closeness_diff = np.abs(closeness - 1.)
    
    return year_diff + closeness_diff


## Normalize year differences by 100; closeness values already between 0 and 1
## year difference of 10 years is equivalent to 0.1 difference in closenesses
def comparison_quality_choice(year1, closeness1, year2, closeness2):
    score1 = 0
    score2 = 0
    
    if year1 > year2:
        score1 += np.abs(year2 - year1)/100.
    elif year1 < year2:
        score2 += np.abs(year2 - year1)/100.
    
    if closeness1 > closeness2:
        score1 += np.abs(closeness2 - closeness1)
    elif closeness1 < closeness2:
        score2 += np.abs(closeness2 - closeness1)
    
    
    if score1 >= score2:
        return 0
    else:
        return 1
    return
        
# print()





## Global properties of groups
## M500
obs_mernier17['group_properties']['M500'] = {
    'RB2002':{
        ## Reiprich & Boehringer (2002)
        ## M500 [10^14 h50^-1], h50=H0/50=1
        'A85':6.84,
        'A119':6.23,
        'A133':2.78,
        'NGC507':0.41,
        'A262':0.90,
        'A400':1.28,
        'A399':10.00,
        'A401':10.27,
        'A3112':5.17,
        'Fornax':0.87,
        '2A0335':2.21,
        'ZwIII54':2.36,
        'A3158':7.00,
        'A478':11.32,
        'NGC1550':0.69,
        'EXO0422':2.89,
        'A3266':14.17,
        'A496':2.76,
        'A3376':6.32,
        'A3391':5.18,
        'A3395s':8.82,
        'A576':5.36,
        'A754':16.37,
        'HydraA':3.76,
        'A1060':2.66,
        'A1367':3.34,
        'MKW4':0.64,
        'ZwCI1215':8.79,
        'NGC4636':0.22,
        'A3526':2.39,
        'A1644':4.10,
        'A1650':9.62,
        'A1651':7.45,
        'Coma':11.99,
        'NGC5044':0.41,
        'A1736':2.19,
        'A3558':5.37,
        'A3562':3.68,
        'A3571':8.33,
        'A1795':9.75,
        'A3581':0.96,
        'MKW8':2.10,
        'A2029':11.82,
        'A2052':1.95,
        'MKW3s':3.06,
        'A2065':13.44,
        'A2063':2.84,
        'A2142':13.29,
        'A2147':2.99,
        'A2163':31.85,
        'A2199':4.21,
        'A2204':8.67,
        'A2244':8.65,
        'A2256':12.83,
        'A2255':10.90,
        'A3667':6.88,
        'AS1101':2.58,
        'A2589':3.14,
        'A2597':4.52,
        'A2634':3.15,
        'A2657':2.83,
        'A4038':2.16,
        'A4059':3.95,
        'A2734':3.49,
        'A2877':2.61,
        'NGC499':0.36,
        'AWM7':3.79,
        'Perseus':6.84,
        'S405':3.91,
        '3C129':5.68,
        'A539':2.33,
        'S540':1.83,
        'A0548w':0.63,
        'A0548e':1.74,
        'A3395n':8.70,
        'UGC03957':2.51,
        'PKS0745':8.88,
        'A644':12.50,
        'S636':0.61,
        'A1413':10.20,
        'M49':0.41,
        'A3528n':2.89,
        'A3528s':1.69,
        'A3530':4.52,
        'A3532':4.77,
        'A1689':15.49,
        'A3560':2.16,
        'A1775':3.61,
        'A1800':4.75,
        'A1914':21.43,
        'NGC5813':0.24,
        'NGC5846':0.33,
        'A2151w':1.52,
        'A3627':5.63,
        'Triangulum':13.42,
        'Ophiuchus':20.25,
        'ZwC11742':6.88,
        'A2319':11.16,
        'A3695':5.57,
        'ZwII108':2.96,
        'A3822':4.97,
        'A3827':16.35,
        'A3888':22.00,
        'A3921':8.46,
        'HCG94':2.28,
        'RXJ2344':6.91,
    },
    'Sun2009':{
        ## Sun et al (2009)
        ## M500 [Msun]
        'A1991':13.4,
        'AS1101':14.1,
        'MKW4':4.85,
        'NGC1550':3.18,
    },
    'CLoGS':{
        ## CLoGS O'Sullivan et al (2017)
        ## M500 [Msun] (estimated from Tsys using Sun+2009 scaling relations for Tier1+2 groups)
        'NGC4261':4.83e13,
        'NGC5044':4.36e13,
        'NGC5846':2.65e13,
    },
    'Eckmiller+2011':{
        ## Eckmiller et al. (2011) Chandra X-ray groups
        ## M500 [1e13 h70^-1 Msun] h70 = 0.7
        'HCG62':2.88,
        'MKW4':9.48,
        'NGC507':2.98,
        'NGC1550':2.52,
        'NGC4325':2.28,
    },
    'Lovisari+2015':{
        ## Lovisar et al. (2015) (XMM-Newton)
        ## M500 [1e13 h70^-1 Msun] h70 = 1
        'HCG62':2.39,
        'NGC4325':2.34,
    }
}

h50_rb2002 = 1
h70_eckmiller2011 = 0.7
h70_lovisari2015 = 1
for group, M500 in obs_mernier17['group_properties']['M500']['RB2002'].items():
    obs_mernier17['group_properties']['M500']['RB2002'][group] = pnb.array.SimArray(M500 * 1e14 * h50_rb2002, units='Msol')
#     print(pnb.array.SimArray(M500 * 1e14 * h50_rb2002, units='Msol'))
#     print(obs_mernier17['individual']['M500']['RB2002'][group])

for group, M500 in obs_mernier17['group_properties']['M500']['Sun2009'].items():
    obs_mernier17['group_properties']['M500']['Sun2009'][group] = pnb.array.SimArray(M500 * 1e13, units='Msol')
    
for group, M500 in obs_mernier17['group_properties']['M500']['CLoGS'].items():
    obs_mernier17['group_properties']['M500']['CLoGS'][group] = pnb.array.SimArray(M500, units='Msol')
    
for group, M500 in obs_mernier17['group_properties']['M500']['Eckmiller+2011'].items():
    obs_mernier17['group_properties']['M500']['Eckmiller+2011'][group] = pnb.array.SimArray(M500 * 1e13 * h70_eckmiller2011, units='Msol')
    
for group, M500 in obs_mernier17['group_properties']['M500']['Lovisari+2015'].items():
    obs_mernier17['group_properties']['M500']['Lovisari+2015'][group] = pnb.array.SimArray(M500 * 1e13 * h70_lovisari2015, units='Msol')

    
    

## R500
obs_mernier17['group_properties']['R500'] = {
    'Mernier+2017':{
        ## Mernier et al (2017) [Mpc]
        '2A0335':1.05,
        'A85':1.21,
        'A133':0.94,
        'A189':0.50,
        'A262':0.74,
        'A496':1.00,
        'A1795':1.22,
        'A1991':0.82,
        'A2029':1.33,
        'A2052':0.95,
        'A2199':1.00,
        'A2597':1.11,
        'A2626':0.84,
        'A3112':1.13,
        'A3526':0.83,
        'A3581':0.72,
        'A4038':0.89,
        'A4059':0.96,
        'AS1101':0.98,
        'AWM7':0.86,
        'EXO0422':0.89,
        'Fornax':0.40,
        'HCG62':0.46,
        'HydraA':1.07,
        'M49':0.53,
        'NGC4649':0.53,
        'M84':0.46,
        'M86':0.49,
        'M87':0.75,
        'M89':0.44,
        'MKW3s':0.95,
        'MKW4':0.62,
        'NGC507':0.60,
        'NGC1316':0.46,
        'NGC1404':0.61,
        'NGC1550':0.62,
        'NGC3411':0.47,
        'NGC4261':0.45,
        'NGC4325':0.58,
        'NGC4636':0.35,
        'NGC5044':0.56,
        'NGC5813':0.44,
        'NGC5846':0.36,
        'Perseus':1.29,
    },
    'CLoGS':{
        ## CLoGS O'Sullivan et al (2017)
        ## R500 [kpc] (estimated from Tsys using Sun+2009 scaling relations for Tier1+2 groups)
        'NGC4261':552,
        'NGC5044':533,
        'NGC5846':452,
    },
}

for group, R500 in obs_mernier17['group_properties']['R500']['Mernier+2017'].items():
    obs_mernier17['group_properties']['R500']['Mernier+2017'][group] = pnb.array.SimArray(R500, units='Mpc').in_units('kpc')
#     print(obs_mernier17['individual']['R500']['M+2017'][group])
    
#     obs_mernier17['individual']['M500']['M+2017/Simba'][group] = 
#     obs_mernier17['individual']['M500']['M+2017/Simba-C'][group] = 
#     obs_mernier17['individual']['M500']['Mernier+2017/Sun2009'][group] = M500_T500_sun09(T500_R500_sun09(obs_mernier17['individual']['R500']['Mernier+2017'][group], h_simba), h_simba)
#     print(obs_mernier17['individual']['M500']['M+2017/Sun2009'][group].units)

# print(obs_mernier17['individual']['M500']['M+2017/Sun2009'])

for group, R500 in obs_mernier17['group_properties']['R500']['CLoGS'].items():
    obs_mernier17['group_properties']['R500']['CLoGS'][group] = pnb.array.SimArray(R500, units='kpc')
    
    



## Compile dictionary of all best 'M500' values directly from literature
# print('All best M500 value directly from literature')
# print()
score_dict = {}
M500_samples = obs_mernier17['group_properties']['M500'].copy().items()
# print(M500_samples.keys())
obs_mernier17['group_properties']['M500']['full_literature'] = {}
# print(M500_samples.keys())

for group in obs_mernier17['group_properties']['R500']['Mernier+2017'].keys():
#     print(group)
    score_dict[group] = {}
    
    ## Calculate quality score for each sample that has this group
    for sample, sample_groups in M500_samples:
        sample_info = obs_mernier17['group_properties']['obs_samples'][sample]
        if group in sample_groups.keys():
            score_dict[group][sample] = quality_score(sample_info['year'], sample_info['closeness'], current_year)
#             print(sample,':','score=',score_dict[group][sample],':',
#                   'M500=',sample_groups[group],':')#,
# #                   'T=',sample_groups[group])

     ## If there were samples for this group, determine best sample (minimum quality score)
    if len(score_dict[group].keys())>0:
#         best_sample = list(score_dict[group].keys())[np.argmin(score_dict[group].values())]
        best_sample = min(score_dict[group], key=score_dict[group].get)
#         del score_dict[best_sample]
#         print('best sample:',best_sample)
        
        obs_mernier17['group_properties']['M500']['full_literature'][group] = obs_mernier17['group_properties']['M500'][best_sample][group]
    
#     print()
# print()
# print()

    



## Temperature
obs_mernier17['group_properties']['T'] = {
    'dePlaa+2017':{
        ## de Plaa et al., 2017 (original CHEERS paper)
        ## Tx [keV] -->  Adapted from Chen et al (2007) and Snowden et al (2008)
        ## Chen+2007: Tm is emission-measure weighted temp, derived from single temp fit to global X-ray spectrum
        ##            Th is hotter bulk component of a two-temp model fitted to the spectrum (allow for low-temp phase in cool core)
        ##            (Th is usually slightly higher than Tm, and Th is expected to provide good measure of grav depth and total mass)
        ##            don't know if dePlaa+2017 used Tm or Th
        ## Snowden+2008: T is average fitted value for cluster temperature in 1'-4' annulus (normalized to values in range 5-30% of R500?)
        '2A0335':3.0,
        'A85':6.1,
        'A133':3.8,
        'A189':1.3,
        'A262':2.2,
        'A496':4.1,
        'A1795':6.0,
        'A1991':2.7,
        'A2029':8.7,
        'A2052':3.0,
        'A2199':4.1,
        'A2597':3.6,
        'A2626':3.1,
        'A3112':4.7,
        'A3526':3.7,
        'A3581':1.8,
        'A4038':3.2,
        'A4059':4.1,
        'AS1101':3.0,
        'AWM7':3.3,
        'EXO0422':3.0,
        'Fornax':1.2,
        'HCG62':1.1,
        'HydraA':3.8,
        'M49':1.0,
        'M84':0.6,
        'M86':0.7,
        'M87':1.7,
        'M89':0.6,
        'MKW3s':3.5,
        'MKW4':1.7,
        'NGC507':1.3,
        'NGC1316':0.6,
        'NGC1404':0.6,
        'NGC1550':1.4,
        'NGC3411':0.8,
        'NGC4261':0.7,
        'NGC4325':1.0,
        'NGC4636':0.8,
        'NGC4649':0.8,
        'NGC5044':1.1,
        'NGC5813':0.5,
        'NGC5846':0.8,
        'Perseus':6.8,
    },
    'RB2002':{
        ## Reiprich & Boehringer (2002)
        ## Tx [keV]
        'A85':6.90,
        'A119':5.60,
        'A133':3.80,
        'NGC507':1.26,
        'A262':2.15,
        'A400':2.31,
        'A399':7.00,
        'A401':8.00,
        'A3112':5.30,
        'Fornax':1.20,
        '2A0335':3.01,
        'ZwIII54':2.16,
        'A3158':5.77,
        'A478':8.40,
        'NGC1550':1.43,
        'EXO0422':2.90,
        'A3266':8.00,
        'A496':4.13,
        'A3376':4.00,
        'A3391':5.40,
        'A3395s':5.00,
        'A576':4.02,
        'A754':9.50,
        'HydraA':4.30,
        'A1060':3.24,
        'A1367':3.55,
        'MKW4':1.71,
        'ZwCI1215':5.58,
        'NGC4636':0.76,
        'A3526':3.68,
        'A1644':4.70,
        'A1650':6.70,
        'A1651':6.10,
        'Coma':8.38,
        'NGC5044':1.07,
        'A1736':3.50,
        'A3558':5.50,
        'A3562':5.16,
        'A3571':6.90,
        'A1795':7.80,
        'A3581':1.83,
        'MKW8':3.29,
        'A2029':9.10,
        'A2052':3.03,
        'MKW3s':3.70,
        'A2065':5.50,
        'A2063':3.68,
        'A2142':9.70,
        'A2147':4.91,
        'A2163':13.29,
        'A2199':4.10,
        'A2204':7.21,
        'A2244':7.10,
        'A2256':6.60,
        'A2255':6.87,
        'A3667':7.00,
        'AS1101':3.00,
        'A2589':3.70,
        'A2597':4.40,
        'A2634':3.70,
        'A2657':3.70,
        'A4038':3.15,
        'A4059':4.40,
        'A2734':3.85,
        'A2877':3.50,
        'NGC499':0.72,
        'AWM7':3.75,
        'Perseus':6.79,
        'S405':4.21,
        '3C129':5.60,
        'A0539':3.24,
        'S540':2.40,
        'A0548w':1.20,
        'A0548e':3.10,
        'A3395n':5.00,
        'UGC03957':2.58,
        'PKS0745':7.21,
        'A644':7.90,
        'S636':1.18,
        'A1413':7.32,
        'M49':0.95,
        'A3528n':3.40,
        'A3528s':3.15,
        'A3530':3.89,
        'A3532':4.58,
        'A1689':9.23,
        'A3560':3.16,
        'A1775':3.69,
        'A1800':4.02,
        'A1914':10.53,
        'NGC5813':0.52,
        'NGC5846':0.82,
        'A2151w':2.40,
        'A3627':6.02,
        'Triangulum':9.60,
        'Ophiuchus':10.26,
        'ZwC11742':5.23,
        'A2319':8.80,
        'A3695':5.29,
        'ZwII108':3.44,
        'A3822':4.90,
        'A3827':7.08,
        'A3888':8.84,
        'A3921':5.73,
        'HCG94':3.45,
        'RXJ2344':4.73,
    },
    'Sun2009':{
        ## Sun et al (2009)
        ## T500 [keV]
        'A262':1.94,
        'A1991':2.68,
        'A3581':1.68,
        'AS1101':2.57,
        'MKW4':1.58,
        'NGC1550':1.06,
        'NGC4325':0.89,
    },
    'ACCEPT':{
        ## ACCEPTCAT - Archive of Chandra Cluster Entropy Profile Tables (ACCEPT) Catalog
        ## "Intracluster Medium Entropy Profiles for a Chandra Archival Sample of Galaxy Clusters"
        ## Cavagnolo, Kenneth W.; Donahue, Megan; Voit, G. Mark; Sun, Ming
        ## Cavognolo et al (2009)
        ## Tx [keV]
        '2A0335':2.88,
        'A85':6.90,
        'A133':3.71,
#         'A189':None,
        'A262':2.18,
        'A496':3.89,
        'A1795':7.80,
        'A1991':5.40,
        'A2029':7.38,
        'A2052':2.98,
        'A2199':4.14,
        'A2597':3.58,
        'A2626':2.90,
        'A3112':4.28,
        'A3526':3.96,
        'A3581':2.10,
        'A4038':3.30,
        'A4059':4.69,
        'AS1101':2.65,
        'AWM7':3.71,
        'EXO0422':3.28,
#         'Fornax':None,
        'HCG62':1.10,
        'HydraA':4.30,
        'M49':1.33,
#         'NGC4649':None,
#         'M84':None,
#         'M86':None,
        'M87':2.50,
#         'M89':None,
        'MKW3s':3.50,
        'MKW4':2.16,
        'NGC507':1.40,
#         'NGC1316':None,
#         'NGC1404':None,
#         'NGC1550':None,
#         'NGC3411':None,
#         'NGC4261':None,
#         'NGC4325':None,
        'NGC4636':0.66,
        'NGC5044':1.22,
        'NGC5813':0.76,
        'NGC5846':0.64,
#         'Perseus':None,
    },
    'Ponman+1996':{
        ## Ponman et al (1996)
        ## Tx [keV]
        'HCG62':0.96,
        ## Cluster data in Ponman+1996 from Edge & Stewart (1991) and Yamashita (1992)
        'Perseus':6.08,
        'A3526':3.54,
        'A1795':5.34,
    },
    'CLoGS':{
        ## CLoGS O'Sullivan et al (2017)
        ## Tsys [keV] (spectral fitting --> not core temp!)
        'NGC4261':1.36,
        'NGC5044':1.28,
        'NGC5846':0.95,
    },
    'Heldson+2002':{
        ## Heldson et al (2002) (H0=50 km/s/Mpc)
        ## Tx [keV]
        'NGC4261':0.94,
        'NGC4325':0.86,
        'NGC4636':0.72,
        'NGC5846':0.70,
    },
    'Eckmiller+2011':{
        ## Eckmiller et al. (2011) Chandra X-ray groups
        ## Tx [keV]
        'HCG62':1.31,
        'MKW4':1.86,
        'NGC507':1.32,
        'NGC1550':1.33,
        'NGC4325':0.98,
    },
#     'Kettula+2013':{
#         ## Kettula et al. (2013)
#         ## Tx [keV] --> hard energy band more reliable?
#         'A1795':4.2,
#         'A262':2.1,
#         'A3112':3.9,
#         'A496':3.4,
#         'AWM7':4.0,
#         'A3526':3.7,
#     },
    'Lovisari+2015':{
        ## Lovisar et al. (2015) (XMM-Newton)
        ## Tx [keV]
        'HCG62':1.05,
        'NGC4325':1.00,
    }
}



for sample, groups in obs_mernier17['group_properties']['T'].items():
    obs_mernier17['group_properties']['M500'][sample+'_T/Sun2009_scaling'] = {}
    obs_mernier17['group_properties']['M500'][sample+'_T/simba_scaling_aviv'] = {}
    obs_mernier17['group_properties']['M500'][sample+'_T/simbac_scaling_aviv'] = {}
    obs_mernier17['group_properties']['M500'][sample+'_T/simbac_scaling_renier'] = {}
    
    for group, temp in groups.items():
        T = pnb.array.SimArray(temp, units='keV')
#         print(T)
        obs_mernier17['group_properties']['T'][sample][group] = T
        
        obs_mernier17['group_properties']['M500'][sample+'_T/Sun2009_scaling'][group] = scale.M500_T500_sun09(T, h_simba)
        obs_mernier17['group_properties']['M500'][sample+'_T/simba_scaling_aviv'][group] = scale.M500_Tspec_simba_N1024L100_aviv(T)
        obs_mernier17['group_properties']['M500'][sample+'_T/simbac_scaling_aviv'][group] = scale.M500_Tspec_simbac_N1024L100_aviv(T)
        obs_mernier17['group_properties']['M500'][sample+'_T/simbac_scaling_renier'][group] = scale.M500_Tspec_simbac_N1024L100_renier(T)





## Compile dictionary of all M500 (and T) values calculated from temperature measurements, one per group
## choosing the value - if there are multiple for a given group, based on how closely the measured
## temperature matches the way Tspec,corr is calculated and how new the paper is
# print('Best M500 values calculated from M500-T scaling relations')
# print()
score_dict = {}
obs_mernier17['group_properties']['T']['best'] = {}  ## Best T values from literature
obs_mernier17['group_properties']['M500']['simbac_full_scaling_aviv'] = {}  ## Best M500 values calculated from M500-T scaling relation
obs_mernier17['group_properties']['M500']['simba_full_scaling_aviv'] = {}
obs_mernier17['group_properties']['M500']['simbac_full_scaling_renier'] = {}  ## Best M500 values calculated from M500-T scaling relation
obs_mernier17['group_properties']['M500']['Sun2009_full_scaling'] = {}
# obs_mernier17['individual']['M500']['simbac_full_literature'] = {}  ## All M500 values directly from literature that match 'simbac_full' (may not have all of them)
# obs_mernier17['individual']['M500']['full_literature'] = {}  ## All M500 values directly from literature for which I have T values (may not have all of them)
obs_mernier17['group_properties']['M500']['full_literature_match_T'] = {}  ## All M500 values directly from literature for which I have T values (may not have all of them)

for group in obs_mernier17['group_properties']['R500']['Mernier+2017'].keys():
#     print(group)
    score_dict[group] = {}
    
    ## Calculate quality score for each sample that has this group
    for sample, sample_groups in obs_mernier17['group_properties']['T'].items():
        if sample == 'best':
            continue
        sample_info = obs_mernier17['group_properties']['obs_samples'][sample]
        if group in sample_groups.keys():
            score_dict[group][sample] = quality_score(sample_info['year'], sample_info['closeness'], current_year)
#             print(sample,':','score=',score_dict[group][sample],':',
#                   'Simba-C (Aviv) M500=',obs_mernier17['individual']['M500'][sample+'_T/simbac_scaling_aviv'][group],':',
#                   'Simba-C (Renier) M500=',obs_mernier17['individual']['M500'][sample+'_T/simbac_scaling_renier'][group],':',
#                   'T=',sample_groups[group])
            
    ## If there were samples for this group, determine best sample (minimum quality score)
    if len(score_dict[group].keys())>0:
#         best_sample = list(score_dict[group].keys())[np.argmin(score_dict[group].values())]
        best_sample = min(score_dict[group], key=score_dict[group].get)
#         del score_dict[best_sample]
#         print('best sample:',best_sample)
    
        
        ## Set 'simbac_full' for that group to T value of group from best sample
        obs_mernier17['group_properties']['T']['best'][group] = obs_mernier17['group_properties']['T'][best_sample][group]
        
        ## Set 'simbac_full' for that group to M500 value of group from best sample
        obs_mernier17['group_properties']['M500']['simbac_full_scaling_aviv'][group] = obs_mernier17['group_properties']['M500'][best_sample+'_T/simbac_scaling_aviv'][group]
        obs_mernier17['group_properties']['M500']['simba_full_scaling_aviv'][group] = obs_mernier17['group_properties']['M500'][best_sample+'_T/simba_scaling_aviv'][group]
        obs_mernier17['group_properties']['M500']['simbac_full_scaling_renier'][group] = obs_mernier17['group_properties']['M500'][best_sample+'_T/simbac_scaling_renier'][group]
        
        obs_mernier17['group_properties']['M500']['Sun2009_full_scaling'][group] = obs_mernier17['group_properties']['M500'][best_sample+'_T/Sun2009_scaling'][group]
        
        
        ## If available
#         if best_sample in obs_mernier17['individual']['M500'].keys():
#             obs_mernier17['individual']['M500']['simbac_full_match'][group] = obs_mernier17['individual']['M500'][best_sample][group]
#         else:
        while len(score_dict[group].keys())>0:
            best_sample = min(score_dict[group], key=score_dict[group].get)
#             print(best_sample)
            
            if best_sample in obs_mernier17['group_properties']['M500'].keys():
                if group in obs_mernier17['group_properties']['M500'][best_sample].keys():
                    obs_mernier17['group_properties']['M500']['full_literature_match_T'][group] = obs_mernier17['group_properties']['M500'][best_sample][group]
                    break
#                 else:
#                     print('None')
#                     break
#             else:
#                 print('None')
#                 break

            del score_dict[group][best_sample]
                
#     print()
    
# print()


## Find only groups that match
obs_mernier17['group_properties']['M500']['simbac_full_scaling_aviv_match'] = {}  ## Best M500 values from M500-T scaling relation that are also present in 'simbac_full_literature'
obs_mernier17['group_properties']['M500']['simba_full_scaling_aviv_match'] = {}
# obs_mernier17['individual']['M500']['simbac_aviv_full_literature_match'] = {}  ## Best M500 values from M500-T scaling relation that are also present in 'simbac_full_literature'
obs_mernier17['group_properties']['M500']['simbac_full_scaling_renier_match'] = {}  ## Best M500 values from M500-T scaling relation that are also present in 'simbac_full_literature'
obs_mernier17['group_properties']['M500']['Sun2009_full_scaling_match'] = {}
# obs_mernier17['individual']['M500']['simbac_renier_full_literature_match'] = {}  ## Best M500 values from M500-T scaling relation that are also present in 'simbac_full_literature'
# obs_mernier17['individual']['M500']['full_literature_match'] = {}  ## Best M500 values from M500-T scaling relation that are also present in 'simbac_full_literature'

# common_groups = obs_mernier17['individual']['M500']['simbac_full_scaling'].keys() & obs_mernier17['individual']['M500']['simbac_full_literature'].keys()
common_groups = obs_mernier17['group_properties']['M500']['simbac_full_scaling_aviv'].keys() & obs_mernier17['group_properties']['M500']['full_literature_match_T'].keys()
# print(common_groups)
# print(len(common_groups))
for group in common_groups:
    obs_mernier17['group_properties']['M500']['simbac_full_scaling_aviv_match'][group] = obs_mernier17['group_properties']['M500']['simbac_full_scaling_aviv'][group]
    obs_mernier17['group_properties']['M500']['simba_full_scaling_aviv_match'][group] = obs_mernier17['group_properties']['M500']['simba_full_scaling_aviv'][group]
    obs_mernier17['group_properties']['M500']['simbac_full_scaling_renier_match'][group] = obs_mernier17['group_properties']['M500']['simbac_full_scaling_renier'][group]
    obs_mernier17['group_properties']['M500']['Sun2009_full_scaling_match'][group] = obs_mernier17['group_properties']['M500']['Sun2009_full_scaling'][group]
#     obs_mernier17['individual']['M500']['simbac_full_literature_match'][group] = obs_mernier17['individual']['M500']['simbac_full_literature'][group]

# print(score_dict)
# print(obs_mernier17['individual']['M500']['simbac_full'])







## Gastaldello et al (2021) (Compilation, with unknown solar abundance normalizations for each?)
auth = 'Gastaldello21'
sub_dir = 'data/Fe'
curr_dir = os.path.join(data_dir, auth, sub_dir)
# os.listdir(curr_dir)
#file = 'data_fig4.dat'

obs_gastaldello21 = {
    prop.stem:np.genfromtxt(prop, comments='#', skip_header=1, names=None) for prop in [Path(x) for x in glob(os.path.join(curr_dir, '*'))]
}

#obs_gastaldello21 = np.genfromtxt(os.path.join(curr_dir, file))
#obs_gastaldello21 = pd.read_csv(os.path.join(curr_dir, file))#, delimiter='\s+')

# obs_gastaldello21




## Other Properties


# Oppenheimer+21 simulation profiles

auth = 'Oppenheimer+21'
sub_dir = 'simulations'
curr_dir = os.path.join(data_dir, auth, sub_dir)
props_ = ['density', 'entropy', 'pressure', 'temperature']
sims_ = ['EAGLE', 'SIMBA', 'TNG100']
# print(os.listdir(curr_dir))
obs_oppenheimer21 = {
    prop_:{
        sim_:[
            np.loadtxt(os.path.join(curr_dir, file_)) for file_ in os.listdir(curr_dir) if (prop_ in file_ and sim_ in file_ and file_.endswith('dat'))
        ] for sim_ in sims_
    } for prop_ in props_
}

# obs_oppenheimer21



# CLoGS profiles from O'Sullivan+17

auth = 'CLoGS_OSullivan17'
sub_dir = 'data'
curr_dir = os.path.join(data_dir, auth, sub_dir)
props_ = ['entropy', 'kT']
# print(os.listdir(curr_dir))
obs_osullivan17 = {
    prop_:{
        file_:np.loadtxt(os.path.join(curr_dir, prop_+'_profs', file_)) for file_ in os.listdir(os.path.join(curr_dir, prop_+'_profs'))
    } for prop_ in props_
}

# kpc
obs_osullivan17['R500'] = {
    'LGG9':432,
    'LGG18':458,
    'LGG31':406,
    'LGG42':434,
    'LGG58':282,
    'LGG66':312,
    'LGG72':468,
    'LGG103':392,
    'LGG117':267,
    'LGG158':525,
    'LGG185':294,
    'LGG262':336,
    'LGG276':384,
    'LGG278':552,
    'LGG338':533,
    'LGG363':387,
    'LGG393':452,
    'LGG402':346,
    'LGG421':233,
    'LGG473':464,
}

# 10^13 Msun
obs_osullivan17['M500'] = {
    'LGG9':1,
    'LGG18':458,
    'LGG31':406,
    'LGG42':434,
    'LGG58':282,
    'LGG66':312,
    'LGG72':468,
    'LGG103':392,
    'LGG117':267,
    'LGG158':525,
    'LGG185':294,
    'LGG262':336,
    'LGG276':384,
    'LGG278':552,
    'LGG338':533,
    'LGG363':387,
    'LGG393':452,
    'LGG402':346,
    'LGG421':233,
    'LGG473':464,
}

# obs_osullivan17



def fits_columns_to_dataframe(file_path, column_names=None, column_units=None):
#     print(file_path)
    
    # Read FITS file
    with pyfits.open(file_path) as hdul:
        # Extract data from the FITS file
        data = hdul[1].data

        # Get the actual column names from the FITS file
        actual_column_names = data.columns.names
        actual_column_units = data.columns.units

        # If specific column names are provided, use them; otherwise, use all columns
        selected_columns = column_names if column_names else actual_column_names
        selected_units = column_units if column_units else actual_column_units

        # Create a DataFrame with the selected columns
        df_data = {}

        # Ensure all arrays have the same length
        max_length = max(len(data[col]) for col in selected_columns)
        for col, unit in zip(selected_columns, selected_units):
            array = data[col].byteswap().newbyteorder().flatten()
            padded_array = np.pad(array, (0, max_length - len(array)), constant_values=np.nan)
            df_data[col] = {
                'values':padded_array,
                'units':unit,
            }
            

        # Create the DataFrame
        df = pd.DataFrame(df_data)

        # Print the FITS header
#         print("FITS Header:")
        fits_header = hdul[1].header
#         print(fits_header)

#         # Print the first 4 lines of the DataFrame
#         print("\nFirst 4 lines of DataFrame:")
# #         print(df.to_html)
# #         df.style.set_sticky()
# #         df.style.set_sticky(axis="index")
#         print(df.head(4))
# #         print(df.head)

# #         print(df_data)
    
#         print()
#         print()

    return df
#     return df_data

# X-GAP observational data
if False:
    auth = 'X-GAP'
    sub_dir = ''
    # sub_dir = 'transfer_274096_files_fd2d28f7/x-gap/x-gap'
    curr_dir = os.path.join(data_dir, auth, sub_dir)
    # file = 'cosmos_galaxy_groups_catalog.fits'
    # file = 'vdisp_39systems.fits'
    # file = 'xgap_m500_forecast.fits'
    file = 'NP_thermo_profiles.fits'
    # file = 'xgap_master_v1.1.fits'
    # print(os.listdir(curr_dir))

    # obs_xgap = pyfits.open(os.path.join(curr_dir, file))
    obs_xgap = {
        'NP_thermo_profiles':fits_columns_to_dataframe(os.path.join(curr_dir, 'NP_thermo_profiles.fits'), column_names=None),
        'cosmos_galaxy_groups_catalog':fits_columns_to_dataframe(os.path.join(curr_dir, 'cosmos_galaxy_groups_catalog.fits'), column_names=None),
        'vdisp_39systems':fits_columns_to_dataframe(os.path.join(curr_dir, 'vdisp_39systems.fits'), column_names=None),
        'xgap_m500_forecast':fits_columns_to_dataframe(os.path.join(curr_dir, 'xgap_m500_forecast.fits'), column_names=None),
        'xgap_master_v1':fits_columns_to_dataframe(os.path.join(curr_dir, 'xgap_master_v1.1.fits'), column_names=None),
    }


    # obs_xgap = {
    #     prop_:{
    #         sim_:[
    #             np.loadtxt(os.path.join(curr_dir, file_)) for file_ in os.listdir(curr_dir) if (prop_ in file_ and sim_ in file_ and file_.endswith('dat'))
    #         ] for sim_ in sims_
    #     } for prop_ in props_
    # }