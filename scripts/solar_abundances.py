## Load libraries

import numpy as np
import periodictable as pt



## Elemental information

pt_info = {}
for element in pt.elements:
#    print(element.symbol)
    pt_info[element.number] = {
        'symbol':element, 
        'mass':element.mass,
        'mass units':element.mass_units,
        'density':element.density, 
        'density units':element.density_units
    }


# print(pt_info)
    

# ## Total solar metallicity
Z_tot_Asplund09 = 0.0134  # Asplund+09 (photospheric)
Z_tot_Lodders09 = 0.0141  # Lodders+09 (photospheric)
# Z_tot_Asplund09 = 0.0142  # Asplund+09 (proto-solar)
# Z_tot_Lodders09 = 0.0153  # Lodders+09 (proto-solar)



# Photospheric Solar Abundances (from SIMBA stuff from Renier)

Z_sol_Asplund09_v1 = {'All': 0.02, 'He': 0.28, 'Li': None, 'Be': None, 'B': None, 'C': 3.26e-3, 'N': 1.32e-3, 'O': 8.65e-3, 'F': None,
                   'Ne': 2.22e-3, 'Na': None, 'Mg': 9.31e-4, 'Al':None, 'Si': 1.08e-3, 'P': None, 'S': 6.44e-4, 'Cl': None, 'Ar': None,
                   'K': None,'Ca': 1.01e-4, 'Sc': None, 'Ti': None, 'V': None, 'Cr': None, 'Mn': None, 'Fe': 1.73e-3, 'Co': None, 
                      'Ni':None , 'Cu': None, 'Zn': None}

Z_sol_Asplund09_v2 = {'All': 0.0134, 'H': 0.7381, 'He': 0.2485, 'Li': None, 'Be': None, 'B': None, 'C': 2.38e-3, 'N': 0.7e-3, 
                      'O': 5.79e-3, 'F': None,
                   'Ne': 1.26e-3, 'Na': None, 'Mg': 7.14e-4, 'Al':None, 'Si': 6.71e-4, 'P': None, 'S': 3.12e-4, 'Cl': None, 'Ar': None,
                   'K': None,'Ca': 0.65e-4, 'Sc': None, 'Ti': None, 'V': None, 'Cr': None, 'Mn': None, 'Fe': 1.31e-3, 'Co': None,
                      'Ni': None, 'Cu': None, 'Zn': None}

Z_sol_AG89 = {'All': 0.0201, 'H': 0.7314, 'He': 0.2485, 'Li': None, 'Be': None, 'B': None, 'C': 3.18e-3, 'N': 1.15e-3, 'O': 9.97e-3, 'F': None,
                   'Ne': 1.72e-3, 'Na': None, 'Mg': 6.75e-4, 'Al':None, 'Si': 7.30e-4, 'P': None, 'S': 3.80e-4, 'Cl': None, 'Ar': None,
                   'K': None,'Ca': 0.67e-4, 'Sc': None, 'Ti': None, 'V': None, 'Cr': None, 'Mn': None, 'Fe': 1.92e-3, 'Co': None, 'Ni': None, 'Cu': None, 'Zn': None}



## Photospheric solar abundances from Asplund et al, 2009 paper
elem_numbers_Asplund09 = list(range(1, 43)) + list(range(44, 61)) + list(range(62, 84)) + [90, 92]
# print(elem_numbers_Asplund09)
log_epsilon_Asplund09 = [12.00, 10.93, 1.05, 1.38, 2.70, 8.43, 7.83, 8.69, 4.56, 7.93, 6.24,7.60,6.45,7.51,5.41,7.12,5.50,
                     6.40,5.03,6.34,3.15,4.95,3.93,5.64,5.43,7.50,4.99,6.22,4.19,4.56,3.04,3.65,2.30,3.34,2.54,3.25,
                     2.52,2.87,2.21,2.58,1.46,1.88,1.75,0.91,1.57,0.94,1.71,0.80,2.04,1.01,2.18,1.55,2.24,1.08,2.18,
                     1.10,1.58,0.72,1.42,0.96,0.52,1.07,0.30,1.10,0.48,0.92,0.10,0.84,0.10,0.85,-0.12,0.85,0.26,1.40,
                     1.38,1.62,0.92,1.17,0.90,1.75,0.65,0.02,-0.54]
# print(log_epsilon_Asplund09)
#elems_Asplund09 = [] #['H', 'He', '']


## Proto-solar abundances from Lodders et al, 2009 paper
elem_numbers_Lodders09 = list(range(1, 43)) + list(range(44, 61)) + list(range(62, 84)) + [90, 92]
# print(elem_numbers_Lodders09)
log_epsilon_Lodders09 = [12.00, 10.93, 3.28, 1.32, 2.81, 8.39, 7.86, 8.73, 4.44, 8.05, 6.29,7.54,6.46,7.53,5.45,7.16,5.25,
                     6.50,5.11,6.31,3.07,4.93,3.99,5.65,5.50,7.46,4.90,6.22,4.27,4.65,3.10,3.59,2.32,3.36,2.56,3.28,
                     2.38,2.90,2.20,2.57,1.42,1.94,1.78,1.10,1.67,1.22,1.73,0.78,2.09,1.03,2.20,1.57,2.27,1.10,2.18,
                     1.19,1.60,0.77,1.47,0.96,0.53,1.09,0.34,1.14,0.49,0.95,0.14,0.94,0.11,0.73,-0.14,0.67,0.28,1.37,
                     1.36,1.63,0.82,1.19,0.79,2.06,0.67,0.08,-0.52]
# print(log_epsilon_Lodders09)

## Proto-solar abundances from Lodders et al (2003), Table 2
elem_numbers_Lodders03 = list(range(1, 43)) + list(range(44, 61)) + list(range(62, 84)) + [90, 92]
# print(elem_numbers_Lodders09)
log_epsilon_Lodders03 = [12.00, 10.984, 3.35, 1.48, 2.85, 8.46, 7.90, 8.76, 4.53, 7.95, 6.37,7.62,6.54,7.61,5.54,7.26,5.33,
                     6.62,5.18,6.41,3.15,5.00,4.07,5.72,5.58,7.54,4.98,6.29,4.34,4.70,3.17,3.70,2.40,3.43,2.67,3.36,
                     2.43,2.99,2.28,2.67,1.49,2.03,1.89,1.18,1.77,1.30,1.81,0.87,2.19,1.14,2.30,1.61,2.35,1.18,2.25,
                     1.25,1.68,0.85,1.54,1.02,0.60,1.13,0.38,1.21,0.56,1.02,0.18,1.01,0.16,0.84,-0.06,0.72,0.33,1.44,
                     1.42,1.75,0.91,1.23,0.88,2.13,0.76,0.16,-0.42]

## Photospheric solar abundances from Anders & Grevesse (1989) Table 2
elem_numbers_AG89 = list(range(1, 43)) + list(range(44, 61)) + list(range(62, 84)) + [90, 92]
# print(elem_numbers_Asplund09)
log_epsilon_AG89 = [12.00, 10.99, 1.16, 1.15, 2.6, 8.56, 8.05, 8.93, 4.56, 8.09, 6.33,7.58,6.47,7.55,5.45,7.21,5.5,
                     6.56,5.12,6.36,3.10,4.99,4.00,5.67,5.39,7.67,4.92,6.25,4.21,4.60,2.88,3.41,2.37,3.35,2.63,3.23,
                     2.60,2.90,2.24,2.60,1.42,1.92,1.84,1.12,1.69,0.94,1.86,1.66,2.0,1.0,2.24,1.51,2.23,1.12,2.13,
                     1.22,1.55,0.71,1.50,1.00,0.51,1.12,-0.1,1.1,0.26,0.93,0.00,1.08,0.76,0.88,0.13,1.11,0.27,1.45,
                     1.35,1.8,1.01,1.09,0.9,1.85,0.71,0.12,-0.47]



info_Asplund09 = {}
for ii in range(len(elem_numbers_Asplund09)):
    number = elem_numbers_Asplund09[ii]
    info_Asplund09[str(pt_info[number]['symbol'])] = {
        'number':number, 
        'mass':pt_info[number]['mass'], 
        'mass units':pt_info[number]['mass units'], 
        'density':pt_info[number]['density'], 
        'density units':pt_info[number]['density units'],
        'log_epsilon':log_epsilon_Asplund09[ii]
    }
    
info_Lodders09 = {}
for ii in range(len(elem_numbers_Lodders09)):
    number = elem_numbers_Lodders09[ii]
    info_Lodders09[str(pt_info[number]['symbol'])] = {
        'number':number, 
        'mass':pt_info[number]['mass'], 
        'mass units':pt_info[number]['mass units'], 
        'density':pt_info[number]['density'], 
        'density units':pt_info[number]['density units'], 
        'log_epsilon':log_epsilon_Lodders09[ii]
    }
    
info_Lodders03 = {}
for ii in range(len(elem_numbers_Lodders03)):
    number = elem_numbers_Lodders03[ii]
    info_Lodders03[str(pt_info[number]['symbol'])] = {
        'number':number, 
        'mass':pt_info[number]['mass'], 
        'mass units':pt_info[number]['mass units'], 
        'density':pt_info[number]['density'], 
        'density units':pt_info[number]['density units'], 
        'log_epsilon':log_epsilon_Lodders03[ii]
    }
    
info_AG89 = {}
for ii in range(len(elem_numbers_AG89)):
    number = elem_numbers_AG89[ii]
    info_AG89[str(pt_info[number]['symbol'])] = {
        'number':number, 
        'mass':pt_info[number]['mass'], 
        'mass units':pt_info[number]['mass units'], 
        'density':pt_info[number]['density'], 
        'density units':pt_info[number]['density units'], 
        'log_epsilon':log_epsilon_AG89[ii]
    }


## info_Asplund09
gas_mass = 0    
for element in info_Asplund09.keys():
    NX_over_NH = 10**(info_Asplund09[element]['log_epsilon'] - 12.)
    info_Asplund09[element]['NX_over_NH'] = NX_over_NH
    
    if not np.isnan(NX_over_NH):
        gas_mass += NX_over_NH * info_Asplund09[element]['mass']

# print(gas_mass)
for element in info_Asplund09.keys():
#    print(type(element))
    ZX = info_Asplund09[element]['NX_over_NH'] * info_Asplund09[element]['mass'] / gas_mass
    info_Asplund09[element]['ZX'] = ZX
    


## info_Lodders09
gas_mass = 0
for element in info_Lodders09.keys():
    NX_over_NH = 10**(info_Lodders09[element]['log_epsilon'] - 12)
    info_Lodders09[element]['NX_over_NH'] = NX_over_NH
    
    if not np.isnan(NX_over_NH):
        gas_mass += NX_over_NH * info_Lodders09[element]['mass']

# print(gas_mass)        
for element in info_Lodders09.keys():
#    print(type(element))
    ZX = info_Lodders09[element]['NX_over_NH'] * info_Lodders09[element]['mass'] / gas_mass
    info_Lodders09[element]['ZX'] = ZX
    
    
    
## info_Lodders03
gas_mass = 0
for element in info_Lodders03.keys():
    NX_over_NH = 10**(info_Lodders03[element]['log_epsilon'] - 12)
    info_Lodders03[element]['NX_over_NH'] = NX_over_NH
    
    if not np.isnan(NX_over_NH):
        gas_mass += NX_over_NH * info_Lodders03[element]['mass']

# print(gas_mass)        
for element in info_Lodders03.keys():
#    print(type(element))
    ZX = info_Lodders03[element]['NX_over_NH'] * info_Lodders03[element]['mass'] / gas_mass
    info_Lodders03[element]['ZX'] = ZX
    
    
    
## info_AG89
gas_mass = 0
for element in info_AG89.keys():
    NX_over_NH = 10**(info_AG89[element]['log_epsilon'] - 12)
    info_AG89[element]['NX_over_NH'] = NX_over_NH
    
    if not np.isnan(NX_over_NH):
        gas_mass += NX_over_NH * info_AG89[element]['mass']

# print(gas_mass)        
for element in info_AG89.keys():
#    print(type(element))
    ZX = info_AG89[element]['NX_over_NH'] * info_AG89[element]['mass'] / gas_mass
    info_AG89[element]['ZX'] = ZX