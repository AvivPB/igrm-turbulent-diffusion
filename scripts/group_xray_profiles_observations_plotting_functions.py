## Load libraries

import matplotlib.pyplot as plt

import numpy as np
import pynbody as pnb
import pandas as pd
pd.set_option('display.max_columns', None)
import copy

import os
from pathlib import Path
from glob import glob
import astropy.io.fits as pyfits
from oppenheimer21.groups_profiles_all import run_plotting_script as opp21_plot


from gen_sim_data import weighted_mean_of_profiles, weighted_quantiles_of_profiles


from astro_constants import NA_no_units, NA, kB_no_units, kB, mH, mp_no_units, mp, mu_e, mu, G_no_units, G, Rxxx_conversions
from solar_abundances import info_Asplund09, info_Lodders09, info_Lodders03, info_AG89

from group_xray_profiles_observations import obs_sarkar22, obs_fukushima23, obs_mao19, obs_mernier17, obs_mernier18b, obs_ghizzardi21, obs_grange11, obs_werner06, obs_gastaldello21, obs_oppenheimer21, obs_osullivan17#, obs_xgap






# def calc_stack_generic(bins, obs, calc_kwargs=None):
    
#     averaged_profile = np.zeros(len(bins)-1)
#     averaged_scatter = np.zeros(len(bins)-1)

#     for ref_bin in range(len(averaged_profile)):
#         profile_numerator = 0
#         profile_denominator = 0

#         ref_bin_lo = bins[ref_bin]
#         ref_bin_hi = bins[ref_bin+1]

#         for group, info in obs.items():
#             rbins = info['rbins'].copy()
#             vals = info['values'].copy()
#             errs = info['errs'].copy()
            
# #             vals_err_lo = np.abs(info['values_err_lo'].copy())
# #             vals_err_hi = np.abs(info['values_err_hi'].copy())

# #             mean_vals_err = np.mean(np.array([vals_err_lo, vals_err_hi]), axis=0)
            

#             for r_idx in range(len(rbins)-1):

#                 if not np.isfinite(vals[r_idx]) or not np.isfinite(errs[r_idx]):
#                     ## This is wrong, something else should be done with the nan or inf vals
#                     ## They come from taking the log of neg value in the individual profiles
#                     ## but then removing them from the averaging biases the avgd profiles high
#                     ## what to set them to? 0 -> no, this is just 1, which is not right
#                     ## Some large neg number?
#                     continue

#                 if calc_kwargs['exclude_negatives']:
#                     if vals[r_idx] < 0:
#                         continue

#                 ## Weight factor for geometric overlap of radial bin of observed halo with reference radial bin
#                 if rbins[r_idx] >= ref_bin_lo and rbins[r_idx+1] <= ref_bin_hi:
#                     ## Radial bin fully inside reference radial bin
#                     w = 1.
# #                 elif rbins[r_idx] < ref_bin_lo or rbins[r_idx+1] > ref_bin_hi:
#                 elif rbins[r_idx+1] < ref_bin_lo or rbins[r_idx] > ref_bin_hi:
#                     ## Radial bin fully outside reference radial bin
#                     w = 0.
#                 else:
#                     ## Radial bin partly in reference radial bin
#                     w = np.abs(min(rbins[r_idx+1], ref_bin_hi) - max(rbins[r_idx], ref_bin_lo)) / np.abs(rbins[r_idx+1] - rbins[r_idx])

#                 profile_numerator += w * vals[r_idx]/(errs[r_idx]**2)
#                 profile_denominator += w/(errs[r_idx]**2)
        
#         averaged_profile[ref_bin] = profile_numerator/profile_denominator


#     for ref_bin in range(len(averaged_profile)):
#         scatter_numerator = 0
#         scatter_denominator = 0

#         ref_bin_lo = bins[ref_bin]
#         ref_bin_hi = bins[ref_bin+1]
#         ref_bin_avg = averaged_profile[ref_bin]
        
#         for group, info in obs.items():
#             rbins = info['rbins'].copy()
#             vals = info['values'].copy()
#             errs = info['errs'].copy()
            
# #             vals_err_lo = np.abs(info['values_err_lo'].copy())
# #             vals_err_hi = np.abs(info['values_err_hi'].copy())

# #             mean_vals_err = np.mean(np.array([vals_err_lo, vals_err_hi]), axis=0)
            
# #             ## determine if the upper or lower error should be used
# #             directional_vals_err = np.where(vals >= ref_bin_avg, vals_err_lo, vals_err_hi)


#             for r_idx in range(len(rbins)-1):
                
#                 if not np.isfinite(vals[r_idx]) or not np.isfinite(errs[r_idx]):
#                     continue
                
#                 if calc_kwargs['exclude_negatives']:
#                     if vals[r_idx] < 0:
#                         continue

#                 ## Weight factor for geometric overlap of radial bin of observed halo with reference radial bin
#                 if rbins[r_idx] >= ref_bin_lo and rbins[r_idx+1] <= ref_bin_hi:
#                     ## Radial bin fully inside reference radial bin
#                     w = 1.
#                 elif rbins[r_idx+1] < ref_bin_lo or rbins[r_idx] > ref_bin_hi:
#                     ## Radial bin fully outside reference radial bin
#                     w = 0.
#                 else:
#                     ## Radial bin partly in reference radial bin
#                     w = np.abs(min(rbins[r_idx+1], ref_bin_hi) - max(rbins[r_idx], ref_bin_lo)) / np.abs(rbins[r_idx+1] - rbinss[r_idx])

#                 scatter_numerator += w * ((vals[r_idx] - ref_bin_avg)/errs[r_idx])**2
#                 scatter_denominator += w/(errs[r_idx]**2)

#         averaged_scatter[ref_bin] = np.sqrt(scatter_numerator/scatter_denominator)
        
        
#     return averaged_profile, averaged_scatter






## Function for checking if group should be included in given bin
def include_in_bin(bin_prop_val, lower_lim, upper_lim, extra=0.1, use_frac=False, frac=0.1, transform_func=lambda x:x):

    bin_prop_val_ = transform_func(bin_prop_val)
    lower_lim_ = transform_func(lower_lim)
    upper_lim_ = transform_func(upper_lim)
    
    if use_frac:
        bin_width = np.abs(upper_lim_ - lower_lim_)
        extra = frac * bin_width
    
    if bin_prop_val_ >= lower_lim_ - extra and bin_prop_val_ <= upper_lim_ + extra:
        return True
    else:
        return False
    
    
    
    
    

###### Extra functions for Mernier+2017 #######################################################################

def correct_group_name_mernier17(group):
    try:
        return group.split('_')[0]
    except:
        return group
    

def remove_bad_bins_mernier17(element=None, group=None, vals=None, remove_or_set_to_nan='remove'):
    ## 8 concentric angular annuli are defined as (0-0.5, 0.5-1, 1-2, 2-3, 3-4, 4-6, 6-9, 9-12) arcsec
    ## But all the data Francois gave me have 9 annuli??? --> yes (12-15"), last one is discarded from all analyses bc not reliable
    
    ## Why do I have this??
    ## Removing the last element?
#     for idx in range(len(vals)):
#         print('vals[idx]:',vals[idx])
#         vals[idx] = vals[idx][:-1]
#         print('vals[idx]:',vals[idx])
    
#     group_ = group.split('_')[0]
    group_ = correct_group_name_mernier17(group)

    if group_ in ['2A0335', 'HydraA', 'M86']:
        ## Remove >= 6'
        keep_idx_lo = 0
        keep_idx_hi = 6
    elif group_ in ['A4038', 'NGC5044']:
        ## Remove >= 9'
        keep_idx_lo = 0
        keep_idx_hi = 7
    elif group_ in ['A3526'] and element.lower() == 'mg':
        ## Remove >= 9'
        keep_idx_lo = 0
        keep_idx_hi = 7
    elif group_ in ['M84', 'M87', 'NGC4261']:
        ## Remove <= 0.5'
        keep_idx_lo = 1
        keep_idx_hi = len(vals[0])
    elif group_ in ['M89'] and element.lower() in ['mg', 's', 'ar', 'ca', 'ni']:
        ## Remove all
        keep_idx_lo = 0
        keep_idx_hi = 0
    elif group_ in ['M89'] and element.lower() in ['fe', 'si']:
        ## Remove <= 0.5'
        keep_idx_lo = 1
        keep_idx_hi = len(vals[0])
    elif group_ in ['NGC5813'] and element.lower() != 'mg':
        ## Remove <= 0.5'
        keep_idx_lo = 1
        keep_idx_hi = len(vals[0])
    elif group_ in ['NGC5813', 'NGC5846'] and element.lower() == 'mg':
        ## Remove <= 6'
        keep_idx_lo = 6
        keep_idx_hi = len(vals[0])
    else:
        ## Remove none
        keep_idx_lo = 0
        keep_idx_hi = len(vals[0])


    for idx in range(len(vals)):
        if remove_or_set_to_nan.lower() == 'remove':
            vals[idx] = vals[idx][keep_idx_lo:keep_idx_hi]
        else:
            vals[idx][:keep_idx_lo] = np.nan
            vals[idx][keep_idx_hi+1:] = np.nan
    
    return vals


def remove_too_good_obs_mernier17(element=None, group=None):
    ## 8 concentric angular annuli are defined as (0-0.5, 0.5-1, 1-2, 2-3, 3-4, 4-6, 6-9, 9-12) arcsec
    ## Remove observations with excellent data quality, as they have smaller uncertainties, and so are heavily weighted
    ## in the stacking procedure
    
    remove = False
    
#     group_ = group.split('_')[0]
    group_ = correct_group_name_mernier17(group)

    if group_ in ['A3526', 'M87', 'Perseus'] and element.lower() == 'fe':
        remove = True
    elif group_ in ['M49', 'M60', 'NGC4636'] and element.lower() == 'o':
        remove = True
    elif group_ in ['Perseus'] and element.lower() == 'mg':
        remove = True
    elif group_ in ['NGC1550', 'Perseus'] and element.lower() == 's':
        remove = True

    return remove







####### Functions for plotting observational data ################################################################



def plot_ind_metallicity_werner06(type_='metallicity', axes=plt, bin_=None,
                                  bin_low=None, bin_high=None,
                                  element=None, element_num=None, element_denom=None,
                                  plot_kwargs=None, fig_kwargs=None):
    
    ## Werner et al (2006)
    ## Core region of M87
    ## Includes Carbon abundances
    ## RGS spectral fitting
    ## RGS spectra from spatially resolved extraction regions
    ## Uses Lodders+03 solar abundances (don't have, need to get!)
    
    if not plot_kwargs['plot']:
        if plot_kwargs['verbose']:
            print('Not plotting Werner+2006 average core abundances')
        return
    
    if type_.lower() == 'metallicity':
        element_name = element
    elif type_.lower() == 'metallicity ratio':
        element_name = element_num + '/' + element_denom
    else:
        raise Exception("type_ needs to be 'metallicity' or 'metallicity ratio'")
        
    if plot_kwargs['print_labels']:
        if plot_kwargs['include_citation']:
            labels = {
                group:'#data_werner06##%s (Werner+06)' % group for group in obs_werner06['individual'].keys()
            }
        else:
            labels = {
                group:'#data_werner06##%s' % group for group in obs_werner06['individual'].keys()
            }
    else:
        labels = {
            group:None for group in obs_werner06['individual'].keys()
        }
        
        
    data_bin_prop = None
    use_data_bin_prop = plot_kwargs['use_bin_prop']
    if fig_kwargs['binning']['prop'][0] == 'M' and fig_kwargs['binning']['prop'][1] == '500':
        data_bin_prop = 'M500'
    elif bin_prop[0] == 'T':
        data_bin_prop = 'kT'
    else:
        use_data_bin_prop = False
        
        
    ## Get exact observations to plot
    plot_obs = {}
    for group, info in obs_werner06['individual'].items():
        try:
            obs = info['metallicity'][plot_kwargs['region']][element_name]
        except:
            if plot_kwargs['verbose']:
                print(element_name + ' not available from Werner+2006, group', group)
            continue

        if use_data_bin_prop:
            ## Remove groups that do not have bin_prop (eg. M500, kT) values
            if data_bin_prop not in info.keys():
                continue
            ## Remove groups outside of bin_prop bin
            if not include_in_bin(info[data_bin_prop], bin_low, bin_high,
                                  extra=plot_kwargs['limit_extra'], use_frac=plot_kwargs['use_limit_frac'],
                                  frac=plot_kwargs['limit_frac'], transform_func=lambda x:np.log10(x)):
                continue
                
        rbins_R500 = obs['rbins_R500'].copy()
        abun = obs['value'].copy()
        abun_err_lo = np.abs(obs['err_lo'].copy())
        abun_err_hi = np.abs(obs['err_hi'].copy())

        if plot_kwargs['renormalize_abundances']:
            if plot_kwargs['normalizations'] is None:
                raise Exception('Need to provide abundances to normalize by!')

            if type_.lower() == 'metallicity':
                abun *= info_Lodders03[element]['ZX']/plot_kwargs['normalizations'][element]['ZX']
                abun_err_lo *= info_Lodders03[element]['ZX']/plot_kwargs['normalizations'][element]['ZX']
                abun_err_hi *= info_Lodders03[element]['ZX']/plot_kwargs['normalizations'][element]['ZX']
            elif type_.lower() == 'metallicity ratio':
                abun *= (info_Lodders03[element_num]['ZX']/plot_kwargs['normalizations'][element_num]['ZX']) / (info_Lodders03[element_denom]['ZX']/plot_kwargs['normalizations'][element_denom]['ZX'])
                abun_err_lo *= (info_Lodders03[element_num]['ZX']/plot_kwargs['normalizations'][element_num]['ZX']) / (info_Lodders03[element_denom]['ZX']/plot_kwargs['normalizations'][element_denom]['ZX'])
                abun_err_hi *= (info_Lodders03[element_num]['ZX']/plot_kwargs['normalizations'][element_num]['ZX']) / (info_Lodders03[element_denom]['ZX']/plot_kwargs['normalizations'][element_denom]['ZX'])

        if 'r200' in fig_kwargs['xaxis']['units'].lower():
#             rbins_R500 *= Rxxx_conversions['R500/R200']
            rbins_R500 *= Rxxx_conversions[plot_kwargs['sim_for_R_conversion']][plot_kwargs['type_for_R_conversion']]['bin_'+str(bin_)]['R500/R200']
        elif 'r2500' in fig_kwargs['xaxis']['units'].lower():
#             rbins_R500 *= Rxxx_conversions['R500/R2500']
            rbins_R500 *= Rxxx_conversions[plot_kwargs['sim_for_R_conversion']][plot_kwargs['type_for_R_conversion']]['bin_'+str(bin_)]['R500/R2500']


        if fig_kwargs['yaxis']['is_log']:
            # Change to derivative method?
            abun_err_lo = np.abs(np.log10(abun) - np.log10(abun - abun_err_lo))
            abun_err_hi = np.abs(np.log10(abun) - np.log10(abun + abun_err_hi))
            abun = np.log10(abun)
#             abun_err_lo = np.log10(abun_err_lo)
#             abun_err_hi = np.log10(abun_err_hi)

        plot_obs[group] = {
            'xbins':rbins_R500,
            'values':abun,
            'values_err_lo':abun_err_lo,
            'values_err_hi':abun_err_hi,
        }
        
    for group, info in plot_obs.items():
        rbin_centres = 0.5 * (info['xbins'][:-1] + info['xbins'][1:])
        rbin_halfwidths = 0.5 * np.abs(info['xbins'][:-1] - info['xbins'][1:])
    
        xerr = None
        yerr = None
        if plot_kwargs['plot_xerror']:
            xerr = [rbin_halfwidths, rbin_halfwidths]
        if plot_kwargs['plot_yerror']:
            yerr = [info['values_err_lo'], info['values_err_hi']]
        axes.errorbar(rbin_centres, info['values'], 
                      yerr=yerr, xerr=xerr, 
                      marker=plot_kwargs['marker'], 
                      ms=plot_kwargs['ms'], 
                      mec=plot_kwargs['mec'], 
                      mew=plot_kwargs['mew'], 
                      lw=plot_kwargs['lw'], 
                      color=plot_kwargs['colors'][group],
                      ecolor=plot_kwargs['ecolor'],
                      label=labels[group],
                      zorder=plot_kwargs['zorder'])


    if plot_kwargs['show_extra_label']:
        axes.errorbar([], [], 
                      yerr=[], xerr=[], 
                      marker=plot_kwargs['marker'], 
                      ms=plot_kwargs['ms'], 
                      mec=plot_kwargs['mec'], 
                      mew=plot_kwargs['mew'], 
                      lw=plot_kwargs['lw'], 
                      color=plot_kwargs['color'],
                      label=plot_kwargs['extra_label'],
                      zorder=plot_kwargs['zorder'])
    return





def plot_ind_metallicity_grange11(type_='metallicity', axes=plt, bin_=None,
                                  bin_low=None, bin_high=None,
                                  element=None, element_num=None, element_denom=None,
                                  plot_kwargs=None, fig_kwargs=None):
    
    ## Grange et al (2011)
    ## Core regions of 2 groups
    ## Includes Carbon abundances
    ## EPIC and RGS spectral fitting
    ## RGS spectra from extraction region of cross dispersion width 5' (radius ~2.5' ?)
    ## EPIC spectra from circular extraction region of radius 3'
    ## Uses Lodders+03 solar abundances (don't have, need to get!)
    
    if not plot_kwargs['plot']:
        if plot_kwargs['verbose']:
            print('Not plotting Grange+2011 average core abundances')
        return
    
    if type_.lower() == 'metallicity':
        element_name = element
    elif type_.lower() == 'metallicity ratio':
        element_name = element_num + '/' + element_denom
    else:
        raise Exception("type_ needs to be 'metallicity' or 'metallicity ratio'")
        
    if plot_kwargs['print_labels']:
        if plot_kwargs['include_citation']:
            labels = {
                group:'#data_grange11##%s (Grange+11)' % group for group in obs_grange11['individual'].keys()
            }
        else:
            labels = {
                group:'#data_grange11##%s' % group for group in obs_grange11['individual'].keys()
            }
    else:
        labels = {
            group:None for group in obs_grange11['individual'].keys()
        }
        
        
    data_bin_prop = None
    use_data_bin_prop = plot_kwargs['use_bin_prop']
    if fig_kwargs['binning']['prop'][0] == 'M' and fig_kwargs['binning']['prop'][1] == '500':
        data_bin_prop = 'M500'
    elif bin_prop[0] == 'T':
        data_bin_prop = 'kT'
    else:
        use_data_bin_prop = False
        
        
    ## Get exact observations to plot
    plot_obs = {}
    for group, info in obs_grange11['individual'].items():
        try:
            obs = info['metallicity'][plot_kwargs['region']][element_name]
        except:
            if plot_kwargs['verbose']:
                print(element_name + ' not available from Grange+2011, group', group)
            continue

            
        if use_data_bin_prop:
            ## Remove groups that do not have bin_prop (eg. M500, kT) values
            if data_bin_prop not in info.keys():
                continue
            ## Remove groups outside of bin_prop bin
            if group != 'NGC5813':
                if not include_in_bin(info[data_bin_prop], bin_low, bin_high, 
                                      extra=plot_kwargs['limit_extra'], use_frac=plot_kwargs['use_limit_frac'],
                                      frac=plot_kwargs['limit_frac'], transform_func=lambda x:np.log10(x)):
                    continue
            else:
                if bin_high > 10**13.5:
                    continue

                
        rbins_R500 = obs['rbins_R500'].copy()
        abun = obs['value'].copy()
        abun_err_lo = np.abs(obs['err_lo'].copy())
        abun_err_hi = np.abs(obs['err_hi'].copy())

        if plot_kwargs['renormalize_abundances']:
            if plot_kwargs['normalizations'] is None:
                raise Exception('Need to provide abundances to normalize by!')

            if type_.lower() == 'metallicity':
                abun *= info_Lodders03[element]['ZX']/plot_kwargs['normalizations'][element]['ZX']
                abun_err_lo *= info_Lodders03[element]['ZX']/plot_kwargs['normalizations'][element]['ZX']
                abun_err_hi *= info_Lodders03[element]['ZX']/plot_kwargs['normalizations'][element]['ZX']
            elif type_.lower() == 'metallicity ratio':
                abun *= (info_Lodders03[element_num]['ZX']/plot_kwargs['normalizations'][element_num]['ZX']) / (info_Lodders03[element_denom]['ZX']/plot_kwargs['normalizations'][element_denom]['ZX'])
                abun_err_lo *= (info_Lodders03[element_num]['ZX']/plot_kwargs['normalizations'][element_num]['ZX']) / (info_Lodders03[element_denom]['ZX']/plot_kwargs['normalizations'][element_denom]['ZX'])
                abun_err_hi *= (info_Lodders03[element_num]['ZX']/plot_kwargs['normalizations'][element_num]['ZX']) / (info_Lodders03[element_denom]['ZX']/plot_kwargs['normalizations'][element_denom]['ZX'])

        if 'r200' in fig_kwargs['xaxis']['units'].lower():
#             rbins_R500 *= Rxxx_conversions['R500/R200']
            rbins_R500 *= Rxxx_conversions[plot_kwargs['sim_for_R_conversion']][plot_kwargs['type_for_R_conversion']]['bin_'+str(bin_)]['R500/R200']
        elif 'r2500' in fig_kwargs['xaxis']['units'].lower():
#             rbins_R500 *= Rxxx_conversions['R500/R2500']
            rbins_R500 *= Rxxx_conversions[plot_kwargs['sim_for_R_conversion']][plot_kwargs['type_for_R_conversion']]['bin_'+str(bin_)]['R500/R2500']


        if fig_kwargs['yaxis']['is_log']:
            # Change to derivative method?
            abun_err_lo = np.abs(np.log10(abun) - np.log10(abun - abun_err_lo))
            abun_err_hi = np.abs(np.log10(abun) - np.log10(abun + abun_err_hi))
            abun = np.log10(abun)
#             abun_err_lo = np.log10(abun_err_lo)
#             abun_err_hi = np.log10(abun_err_hi)

        
        plot_obs[group] = {
            'xbins':rbins_R500,
            'values':abun,
            'values_err_lo':abun_err_lo,
            'values_err_hi':abun_err_hi,
        }
        
        
    for group, info in plot_obs.items():
        rbin_centres = 0.5 * (info['xbins'][:-1] + info['xbins'][1:])
        rbin_halfwidths = 0.5 * np.abs(info['xbins'][:-1] - info['xbins'][1:])
    
        xerr = None
        yerr = None
        if plot_kwargs['plot_xerror']:
            xerr = [rbin_halfwidths, rbin_halfwidths]
        if plot_kwargs['plot_yerror']:
            yerr = [info['values_err_lo'], info['values_err_hi']]
        axes.errorbar(rbin_centres, info['values'], 
                      yerr=yerr, xerr=xerr, 
                      marker=plot_kwargs['marker'], 
                      ms=plot_kwargs['ms'], 
                      mec=plot_kwargs['mec'], 
                      mew=plot_kwargs['mew'], 
                      lw=plot_kwargs['lw'], 
                      color=plot_kwargs['colors'][group],
                      ecolor=plot_kwargs['ecolor'],
                      label=labels[group],
                      zorder=plot_kwargs['zorder'])


    if plot_kwargs['show_extra_label']:
        axes.errorbar([], [], 
                      yerr=[], xerr=[], 
                      marker=plot_kwargs['marker'], 
                      ms=plot_kwargs['ms'], 
                      mec=plot_kwargs['mec'], 
                      mew=plot_kwargs['mew'], 
                      lw=plot_kwargs['lw'], 
                      color=plot_kwargs['color'],
                      label=plot_kwargs['extra_label'],
                      zorder=plot_kwargs['zorder'])
    return







def plot_metallicity_mernier17(type_='metallicity', axes=plt, bin_=None,
                               element=None, element_num=None, element_denom=None,
                               plot_kwargs=None, fig_kwargs=None):

    # Mernier+17 Observations (CHEERS Sample)
    # Radii in units of R/R500 (convert to R/R200 with R500~0.65R200 from Reiprich+13) --> FIX! (Fixed)
    # Abundances in proto-solar units of Lodders et al (2009) --> FIX! (Fixed?)
    
    if not plot_kwargs['plot']:
        if plot_kwargs['verbose']:
            print('Not plotting Mernier+17 average profiles from paper')
        return
    
    if plot_kwargs['print_labels']:
        if plot_kwargs['include_citation']:
            labels_ = {
                'groups':'#data_mernier17##Stacked Groups (%s)' % ('Mernier+17'),
                'clusters':'#data_mernier17##Stacked Clusters (%s)' % ('Mernier+17')
            }
        else:
            labels_ = {
                'groups':'#data_mernier17##Stacked Groups',
                'clusters':'#data_mernier17##Stacked Clusters',
            }
    else:
        labels_ = {
            'groups':None,
            'clusters':None
        }
        
    
    no_group = False
    no_cluster = False
    
    
    if type_.lower() == 'metallicity':
        name = element.lower()
        
        try:
            group_element_idx = obs_mernier17['median']['group_element_idx'][element]
        except:
            if plot_kwargs['verbose']:
                print('%s not available from Mernier+17 groups' % element)
            no_group = True

        try:
            cluster_element_idx = obs_mernier17['median']['cluster_element_idx'][element]
        except:
            if plot_kwargs['verbose']:
                print('%s not available from Mernier+17 clusters' % element)
            no_cluster = True
            
    elif type_.lower() == 'metallicity ratio':
        name = '%s/%s' % (element_num.lower(), element_denom.lower())
        
        try:
            group_element_num_idx = obs_mernier17['median']['group_element_idx'][element_num]
            group_element_denom_idx = obs_mernier17['median']['group_element_idx'][element_denom]
        except:
            if plot_kwargs['verbose']:
                print('%s or %s not available from Mernier+17 groups' % (element_num, element_denom))
            no_group = True

        try:
            cluster_element_num_idx = obs_mernier17['median']['cluster_element_idx'][element_num]
            cluster_element_denom_idx = obs_mernier17['median']['cluster_element_idx'][element_denom]
        except:
            if plot_kwargs['verbose']:
                print('%s or %s not available from Mernier+17 clusters' % (element_num, element_denom))
            no_cluster = True
        
    else:
        raise Exception("type_ must be 'metallicity' or 'metallicity ratio'")
    

    if plot_kwargs['plot_groups'] and not no_group:
        # Groups
        Rmin = obs_mernier17['median']['Mernier_2017_group'][:,0].copy()
        Rmax = obs_mernier17['median']['Mernier_2017_group'][:,1].copy()
        Rcentre = (Rmin+Rmax)/2.
        
        if type_.lower() == 'metallicity':
            abun_ = obs_mernier17['median']['Mernier_2017_group'][:,group_element_idx].copy()
            abun_err_ = obs_mernier17['median']['Mernier_2017_group'][:,group_element_idx+1].copy()

            if plot_kwargs['renormalize_abundances']:
                if plot_kwargs['normalizations'] is None:
                    raise Exception('Need to provide abundances to normalize by!')

                abun_ *= info_Lodders09[element]['ZX']/plot_kwargs['normalizations'][element]['ZX']
                abun_err_ *= info_Lodders09[element]['ZX']/plot_kwargs['normalizations'][element]['ZX']
         
        elif type_.lower() == 'metallicity ratio':
            abun_num_ = obs_mernier17['median']['Mernier_2017_group'][:,group_element_num_idx].copy()
            abun_err_num_ = obs_mernier17['median']['Mernier_2017_group'][:,group_element_num_idx+1].copy()

            abun_denom_ = obs_mernier17['median']['Mernier_2017_group'][:,group_element_denom_idx].copy()
            abun_err_denom_ = obs_mernier17['median']['Mernier_2017_group'][:,group_element_denom_idx+1].copy()

            if plot_kwargs['renormalize_abundances']:
                if plot_kwargs['normalizations'] is None:
                    raise Exception('Need to provide abundances to normalize by!')

                abun_num_ *= info_Lodders09[element_num]['ZX']/plot_kwargs['normalizations'][element_num]['ZX']
                abun_err_num_ *= info_Lodders09[element_num]['ZX']/plot_kwargs['normalizations'][element_num]['ZX']

                abun_denom_ *= info_Lodders09[element_denom]['ZX']/plot_kwargs['normalizations'][element_denom]['ZX']
                abun_err_denom_ *= info_Lodders09[element_denom]['ZX']/plot_kwargs['normalizations'][element_denom]['ZX']

            abun_ = abun_num_/abun_denom_
            abun_err_ = np.sqrt(abun_err_num_**2 + abun_err_denom_**2)


        if fig_kwargs['xaxis']['is_log']:
            Rcentre = np.log10(Rcentre)
            if 'r200' in fig_kwargs['xaxis']['units'].lower():
#                 Rcentre += np.log10(Rxxx_conversions['R500/R200'])
                Rcentre += np.log10(Rxxx_conversions[plot_kwargs['sim_for_R_conversion']][plot_kwargs['type_for_R_conversion']]['bin_'+str(bin_)]['R500/R200'])
        else:
            if 'r200' in fig_kwargs['xaxis']['units'].lower():
#                 Rcentre *= Rxxx_conversions['R500/R200']
                Rcentre *= Rxxx_conversions[plot_kwargs['sim_for_R_conversion']][plot_kwargs['type_for_R_conversion']]['bin_'+str(bin_)]['R500/R200']


        if fig_kwargs['yaxis']['is_log']:
            abun = np.log10(abun_)
            abun_err_lo = np.abs(abun - np.log10(abun_ - abun_err_))
            abun_err_hi = np.abs(abun - np.log10(abun_ + abun_err_))
        else:
            abun = abun_
            abun_err_lo = abun_err_
            abun_err_hi = abun_err_


        axes.fill_between(Rcentre, abun-abun_err_lo, abun+abun_err_hi,
                          color=plot_kwargs['colors']['groups'], 
                          alpha=plot_kwargs['alpha'], 
                          label='fill!'+labels_['groups'], 
                          zorder=plot_kwargs['zorder'])
        axes.plot(Rcentre, abun, 
                  color=plot_kwargs['colors']['groups'], 
                  lw=plot_kwargs['lw'], 
                  marker=plot_kwargs['marker'], 
                  ms=plot_kwargs['ms'], 
                  mec=plot_kwargs['mec'], 
                  mew=plot_kwargs['mew'], 
                  label='line!'+labels_['groups'],
                  zorder=plot_kwargs['zorder'])

    
    if plot_kwargs['plot_clusters'] and not no_cluster:
        # Clusters
        Rmin = obs_mernier17['median']['Mernier_2017_cluster'][:,0].copy()
        Rmax = obs_mernier17['median']['Mernier_2017_cluster'][:,1].copy()
        Rcentre = (Rmin+Rmax)/2.
        
        
        if type_.lower() == 'metallicity':
            abun_ = obs_mernier17['median']['Mernier_2017_cluster'][:,group_element_idx].copy()
            abun_err_ = obs_mernier17['median']['Mernier_2017_cluster'][:,group_element_idx+1].copy()

            if plot_kwargs['renormalize_abundances']:
                if plot_kwargs['normalizations'] is None:
                    raise Exception('Need to provide abundances to normalize by!')

                abun_ *= info_Lodders09[element]['ZX']/plot_kwargs['normalizations'][element]['ZX']
                abun_err_ *= info_Lodders09[element]['ZX']/plot_kwargs['normalizations'][element]['ZX']
         
        elif type_.lower() == 'metallicity ratio':
            abun_num_ = obs_mernier17['median']['Mernier_2017_cluster'][:,group_element_num_idx].copy()
            abun_err_num_ = obs_mernier17['median']['Mernier_2017_cluster'][:,group_element_num_idx+1].copy()

            abun_denom_ = obs_mernier17['median']['Mernier_2017_cluster'][:,group_element_denom_idx].copy()
            abun_err_denom_ = obs_mernier17['median']['Mernier_2017_cluster'][:,group_element_denom_idx+1].copy()

            if plot_kwargs['renormalize_abundances']:
                if plot_kwargs['normalizations'] is None:
                    raise Exception('Need to provide abundances to normalize by!')

                abun_num_ *= info_Lodders09[element_num]['ZX']/plot_kwargs['normalizations'][element_num]['ZX']
                abun_err_num_ *= info_Lodders09[element_num]['ZX']/plot_kwargs['normalizations'][element_num]['ZX']

                abun_denom_ *= info_Lodders09[element_denom]['ZX']/plot_kwargs['normalizations'][element_denom]['ZX']
                abun_err_denom_ *= info_Lodders09[element_denom]['ZX']/plot_kwargs['normalizations'][element_denom]['ZX']

            abun_ = abun_num_/abun_denom_
            abun_err_ = np.sqrt(abun_err_num_**2 + abun_err_denom_**2)


        if fig_kwargs['xaxis']['is_log']:
            Rcentre = (np.log10(Rmin)+np.log10(Rmax))/2.
            Rcentre = np.log10(Rcentre)
            if 'r200' in fig_kwargs['xaxis']['units'].lower():
#                 Rcentre += np.log10(Rxxx_conversions['R500/R200'])
                Rcentre += np.log10(Rxxx_conversions[plot_kwargs['sim_for_R_conversion']][plot_kwargs['type_for_R_conversion']]['bin_'+str(bin_)]['R500/R200'])
        else:
            if 'r200' in fig_kwargs['xaxis']['units'].lower():
#                 Rcentre *= Rxxx_conversions['R500/R200']
                Rcentre *= Rxxx_conversions[plot_kwargs['sim_for_R_conversion']][plot_kwargs['type_for_R_conversion']]['bin_'+str(bin_)]['R500/R200']


        if fig_kwargs['yaxis']['is_log']:
            abun = np.log10(abun_)
            abun_err_lo = np.abs(abun - np.log10(abun_ - abun_err_))
            abun_err_hi = np.abs(abun - np.log10(abun_ + abun_err_))
        else:
            abun = abun_
            abun_err_lo = abun_err_
            abun_err_hi = abun_err_

        
        axes.fill_between(Rcentre, abun-abun_err_lo, abun+abun_err_hi,
                          color=plot_kwargs['colors']['clusters'], 
                          alpha=plot_kwargs['alpha'], 
                          label='fill!'+labels_['clusters'], 
                          zorder=plot_kwargs['zorder'])
        axes.plot(Rcentre, abun, 
                  color=plot_kwargs['colors']['clusters'],
                  lw=plot_kwargs['lw'], 
                  marker=plot_kwargs['marker'], 
                  ms=plot_kwargs['ms'], 
                  mec=plot_kwargs['mec'], 
                  mew=plot_kwargs['mew'], 
                  label='line!'+labels_['clusters'],
                  zorder=plot_kwargs['zorder'])
        
        
    if plot_kwargs['show_extra_label']:
        axes.plot([], [], 
                  color=plot_kwargs['colors']['groups'],
                  lw=plot_kwargs['lw'], 
                  marker=plot_kwargs['marker'], 
                  ms=plot_kwargs['ms'], 
                  mec=plot_kwargs['mec'], 
                  mew=plot_kwargs['mew'], 
                  label='line!'+plot_kwargs['extra_label'],
                  zorder=plot_kwargs['zorder'])
    return







def plot_ind_metallicity_mernier17(type_='metallicity', axes=plt, bin_=None,
                                   bin_low=None, bin_high=None,
                                   element=None, element_num=None, element_denom=None, 
                                   plot_kwargs=None, fig_kwargs=None):
    
    ## Mernier+2017 observations from CHEERS sample
    ## Abundance profiles of each group
    ## Abundances in units of Lodders+2009
    
    if not plot_kwargs['plot']:
        if plot_kwargs['verbose']:
            print('Not plotting Mernier+2017 individual abundance profiles')
        return        
    
    if type_.lower() == 'metallicity':
        element_name = element
        element_for_instrument = element
    elif type_.lower() == 'metallicity ratio':
        element_name = element_num + '/' + element_denom
        element_for_instrument = element_num
    else:
        raise Exception("type_ needs to be 'metallicity' or 'metallicity ratio'")
        
    if element_for_instrument not in plot_kwargs['instruments'].keys():
        if plot_kwargs['verbose']:
            print('%s not available from Mernier+2017' % element_for_instrument)
        return        
        
        
    if plot_kwargs['print_labels']:
        if plot_kwargs['include_citation']:
            labels = {
                instrument:'#data_mernier17##Stacked Groups (Mernier+17)' + ': ' + obs_mernier17['instruments'][instrument] for instrument in plot_kwargs['instruments'][element_for_instrument]
            }
        else:
            labels = {
                instrument:'#data_mernier17##Stacked Groups: ' + obs_mernier17['instruments'][instrument] for instrument in plot_kwargs['instruments'][element_for_instrument]
            }
    else:
        labels = {
            instrument:None for instrument in plot_kwargs['instruments'][element_for_instrument]
        }
        
        
    data_bin_prop = None
    use_data_bin_prop = plot_kwargs['use_bin_prop']
    dataset = plot_kwargs['M500_dataset']
    if fig_kwargs['binning']['prop'][0] == 'M' and fig_kwargs['binning']['prop'][1] == '500':
        data_bin_prop = 'M500'
    elif bin_prop[0] == 'T':
        data_bin_prop = 'T'
        dataset = 'best'
    else:
        use_data_bin_prop = False
    
    

    ## Get exact observations to plot
    plotting_obs = {}
    for instrument in plot_kwargs['instruments'][element_for_instrument]:
        try:
            obs = obs_mernier17['individual'][type_][element_name+'_'+instrument]
        except:
            if plot_kwargs['verbose']:
                print(element_name + ' not available from Mernier+2017 with XMM-Newton/%s' % instrument)
            continue
            
        plotting_obs[instrument] = {}
            
        for group, info in obs.items():
            group_name = correct_group_name_mernier17(group)
            
            if plot_kwargs['remove_too_good_obs']:
                remove = remove_too_good_obs_mernier17(element=element_for_instrument, group=group_name)
                if remove:
                    if plot_kwargs['verbose']:
                        print('Removing %s from %s Mernier+2017 individual profiles' % (group_name, element_for_instrument))
                    continue

            if use_data_bin_prop:
                ## Remove groups that do not have bin_prop (eg. M500, kT) values
                if group_name not in obs_mernier17['group_properties'][data_bin_prop][dataset].keys():
                    continue
                ## Remove groups outside of bin_prop bin
                if (obs_mernier17['group_properties'][data_bin_prop][dataset][group_name] < pnb.array.SimArray(1e13,units='Msol') and 
                    bin_high > 10**13.5): ## need to change this to be not specifically mass, but any binning prop (so just check whether it is the first or last bin --> how to do that?)
                    continue
                elif (obs_mernier17['group_properties'][data_bin_prop][dataset][group_name] > pnb.array.SimArray(1e15,units='Msol') and 
                      bin_low < 1e14):
                    continue
                elif not include_in_bin(obs_mernier17['group_properties'][data_bin_prop][dataset][group_name], bin_low, bin_high,
                                        extra=plot_kwargs['limit_extra'], use_frac=plot_kwargs['use_limit_frac'],
                                        frac=plot_kwargs['limit_frac'], transform_func=lambda x:np.log10(x)):
                    continue
                

            rbin_centres_R500 = info[:,0].copy()
            rbin_centres_R500_err = info[:,1].copy()
            rbins_R500 = np.zeros(len(rbin_centres_R500)+1)
            for kk in range(len(rbin_centres_R500)):
                rbins_R500[kk] = rbin_centres_R500[kk] - rbin_centres_R500_err[kk]
                rbins_R500[kk+1] = rbin_centres_R500[kk] + rbin_centres_R500_err[kk]
#             Rmin = rad - rad_err
#             Rmax = rad + rad_err
#             rbins_R500 = np.unique(np.append(Rmin, Rmax))
            
            abun = info[:,2].copy()
            abun_err_lo = np.abs(info[:,3].copy())
            abun_err_hi = np.abs(info[:,4].copy())
            
            
            if plot_kwargs['renormalize_abundances']:
                if plot_kwargs['normalizations'] is None:
                    raise Exception('Need to provide abundances to normalize by!')

                if type_.lower() == 'metallicity':
                    abun *= info_Lodders09[element]['ZX']/plot_kwargs['normalizations'][element]['ZX']
                    abun_err_lo *= info_Lodders09[element]['ZX']/plot_kwargs['normalizations'][element]['ZX']
                    abun_err_hi *= info_Lodders09[element]['ZX']/plot_kwargs['normalizations'][element]['ZX']
                elif type_.lower() == 'metallicity ratio':
                    abun *= (info_Lodders09[element_num]['ZX']/plot_kwargs['normalizations'][element_num]['ZX']) / (info_Lodders09[element_denom]['ZX']/plot_kwargs['normalizations'][element_denom]['ZX'])
                    abun_err_lo *= (info_Lodders09[element_num]['ZX']/plot_kwargs['normalizations'][element_num]['ZX']) / (info_Lodders09[element_denom]['ZX']/plot_kwargs['normalizations'][element_denom]['ZX'])
                    abun_err_hi *= (info_Lodders09[element_num]['ZX']/plot_kwargs['normalizations'][element_num]['ZX']) / (info_Lodders09[element_denom]['ZX']/plot_kwargs['normalizations'][element_denom]['ZX'])

            if 'r200' in fig_kwargs['xaxis']['units'].lower():
#                 rbins_R500 *= Rxxx_conversions['R500/R200']
                rbins_R500 *= Rxxx_conversions[plot_kwargs['sim_for_R_conversion']][plot_kwargs['type_for_R_conversion']]['bin_'+str(bin_)]['R500/R200']
            elif 'r2500' in fig_kwargs['xaxis']['units'].lower():
#                 rbins_R500 *= Rxxx_conversions['R500/R2500']
                rbins_R500 *= Rxxx_conversions[plot_kwargs['sim_for_R_conversion']][plot_kwargs['type_for_R_conversion']]['bin_'+str(bin_)]['R500/R2500']
                
            
            if plot_kwargs['remove_bad_bins']:
                abun, abun_err_lo, abun_err_hi = remove_bad_bins_mernier17(element=element_for_instrument, group=group_name, 
                                                                           vals=[abun, abun_err_lo, abun_err_hi], 
                                                                           remove_or_set_to_nan='set_to_nan')

            if fig_kwargs['yaxis']['is_log']:
                # Change to derivative method?
                abun_err_lo = np.abs(np.log10(abun) - np.log10(abun - abun_err_lo))
                abun_err_hi = np.abs(np.log10(abun) - np.log10(abun + abun_err_hi))
                abun = np.log10(abun)
    #             abun_err_lo = np.log10(abun_err_lo)
    #             abun_err_hi = np.log10(abun_err_hi)


            plotting_obs[instrument][group] = {
                'xbins':rbins_R500,
                'values':abun,
                'errs_lo':abun_err_lo,
                'errs_hi':abun_err_hi,
            }
            
            
        if len(plotting_obs[instrument].keys()) == 0:
            if plot_kwargs['verbose']:
                print(element_name, 'not available from Mernier+2017 for XMM-Newton/%s' % instrument)
        
    
    if len(plotting_obs.keys()) == 0:
        if plot_kwargs['verbose']:
            print(element_name, 'not available from Mernier+2017 for chosen instruments')
        return
    
    
    
    
    ## Plot profiles
    labels ={}
    for instrument, obs in plotting_obs.items():

        color_idx = list(plot_kwargs['instruments'][element_for_instrument]).index(instrument)
        color = plot_kwargs['colors'][element_for_instrument][idx]
        
        labels[instrument] = {}
        if plot_kwargs['print_labels']:
            labels[instrument] = {
                group:'#data_mernier17##'+group for group in obs.keys()
            }
        else:
            labels[instrument] = {
                group:None for group in obs.keys()
            }
        
        
        for group, info in obs.items():
            rbins = info['xbins']
            rbin_centres = (rbins[:-1] + rbins[1:])/2.
            if fig_kwargs['xaxis']['is_log']:
                rbin_centres = np.log10(rbin_centres)
            
            values = info['values']
            errs_lo = info['errs_lo']
            errs_hi = info['errs_hi']
            
            yerr = [abun_err_lo, abun_err_hi]
            if not plot_kwargs['error_bars']:
                yerr = None
            axes.errorbar(rbin_centres, values, yerr=yerr,
                          marker=plot_kwargs['marker'], 
                          ms=plot_kwargs['ms'], 
                          mec=plot_kwargs['mec'], 
                          mew=plot_kwargs['mew'], 
                          ecolor=plot_kwargs['ecolor'], 
                          color=color,
                          mfc=plot_kwargs['mfc'], 
                          lw=plot_kwargs['lw'], 
                          alpha=plot_kwargs['alpha'], 
                          label=labels[instrument][group], 
                          zorder=plot_kwargs['zorder'])


    if plot_kwargs['show_extra_label']:
        axes.errorbar([], [], yerr=[],
                      marker=plot_kwargs['marker'], 
                      ms=plot_kwargs['ms'], 
                      mec=plot_kwargs['mec'], 
                      mew=plot_kwargs['mew'], 
                      color=plot_kwargs['color'], 
                      mfc=plot_kwargs['mfc'], 
                      lw=plot_kwargs['lw'], 
                      alpha=plot_kwargs['alpha'], 
                      label=plot_kwargs['extra_label'], 
                      zorder=plot_kwargs['zorder'])
    return






def plot_averaging_metallicity_mernier17(type_='metallicity', axes=plt, bin_=None,
                                       bin_low=None, bin_high=None, sim_rbins=None,
                                       element=None, element_num=None, element_denom=None,
                                       plot_kwargs=None, fig_kwargs=None):
    
    ## Mernier+2017 observations from CHEERS sample
    ## Abundance profiles of each group
    ## Abundances in units of Lodders+2009
    
    if not plot_kwargs['plot']:
        if plot_kwargs['verbose']:
            print('Not plotting Mernier+2017 average abundances')
        return        
    
    if type_.lower() == 'metallicity':
        if element.lower() == 'ni' and bin_high < 10**13.75:
            print('Not plotting Ni in low mass bin')
            return
        element_name = element
        element_for_instrument = element
    elif type_.lower() == 'metallicity ratio':
        if element_num.lower() == 'ni' and bin_high < 10**13.75:
            print('Not plotting Ni in low mass bin')
            return
        element_name = element_num + '/' + element_denom
        element_for_instrument = element_num
    else:
        raise Exception("type_ needs to be 'metallicity' or 'metallicity ratio'")
        
    if element_for_instrument not in plot_kwargs['instruments'].keys():
        if plot_kwargs['verbose']:
            print('%s not available from Mernier+2017' % element_for_instrument)
        return        
        
        
    if plot_kwargs['print_labels']:
        if plot_kwargs['include_citation']:
            labels = {
                instrument:'#data_mernier17##Stacked Groups (Mernier+17)' + ': ' + obs_mernier17['instruments'][instrument] for instrument in plot_kwargs['instruments'][element_for_instrument]
            }
        else:
            labels = {
                instrument:'#data_mernier17##Stacked Groups: ' + obs_mernier17['instruments'][instrument] for instrument in plot_kwargs['instruments'][element_for_instrument]
            }
    else:
        labels = {
            instrument:None for instrument in plot_kwargs['instruments'][element_for_instrument]
        }
        
        
    data_bin_prop = None
    use_data_bin_prop = plot_kwargs['use_bin_prop']
    dataset = plot_kwargs['M500_dataset']
    if fig_kwargs['binning']['prop'][0] == 'M' and fig_kwargs['binning']['prop'][1] == '500':
        data_bin_prop = 'M500'
    elif bin_prop[0] == 'T':
        data_bin_prop = 'T'
        dataset = 'best'
    else:
        use_data_bin_prop = False
        
        
    ## Get exact observations to average
    averaging_obs = {}
    for instrument in plot_kwargs['instruments'][element_for_instrument]:
        try:
            obs = obs_mernier17['individual'][type_][element_name+'_'+instrument]
        except:
            if plot_kwargs['verbose']:
                print(element_name + ' not available from Mernier+2017 with XMM-Newton/%s' % instrument)
            continue
            
        averaging_obs[instrument] = {}
            
        for group, info in obs.items():
            group_name = correct_group_name_mernier17(group)
            
            if plot_kwargs['remove_too_good_obs']:
                remove = remove_too_good_obs_mernier17(element=element_for_instrument, group=group_name)
                if remove:
                    if plot_kwargs['verbose']:
                        print('Removing %s from Mernier+2017 %s average profiles' % (group_name, element_for_instrument))
                    continue

            if use_data_bin_prop:
                ## Remove groups that do not have bin_prop (eg. M500, kT) values
                if group_name not in obs_mernier17['group_properties'][data_bin_prop][dataset].keys():
                    continue
                ## Remove groups outside of bin_prop bin
                if (obs_mernier17['group_properties'][data_bin_prop][dataset][group_name] < pnb.array.SimArray(1e13,units='Msol') and 
                    bin_high > 10**13.5): ## need to change this to be not specifically mass, but any binning prop (so just check whether it is the first or last bin --> how to do that?)
                    continue
                elif (obs_mernier17['group_properties'][data_bin_prop][dataset][group_name] > pnb.array.SimArray(1e15,units='Msol') and 
                      bin_low < 1e14):
                    continue
                elif not include_in_bin(obs_mernier17['group_properties'][data_bin_prop][dataset][group_name], bin_low, bin_high,
                                        extra=plot_kwargs['limit_extra'], use_frac=plot_kwargs['use_limit_frac'],
                                        frac=plot_kwargs['limit_frac'], transform_func=lambda x:np.log10(x)):
                    continue
                

            rbin_centres_R500 = info[:,0].copy()
            rbin_centres_R500_err = info[:,1].copy()
            rbins_R500 = np.zeros(len(rbin_centres_R500)+1)
            for kk in range(len(rbin_centres_R500)):
                rbins_R500[kk] = rbin_centres_R500[kk] - rbin_centres_R500_err[kk]
                rbins_R500[kk+1] = rbin_centres_R500[kk] + rbin_centres_R500_err[kk]
#             Rmin = np.array(rbin_centres_R500 - rbin_centres_R500_err)
#             Rmax = np.array(rbin_centres_R500 + rbin_centres_R500_err)
#             rbins_R500 = np.unique(np.append(Rmin, Rmax))
#             print()
#             print('len(rbin_centres_R500):', len(rbin_centres_R500))
#             print('rbin_centres_R500:', rbin_centres_R500)
#             print('len(rbin_centres_R500_err):', len(rbin_centres_R500_err))
#             print('rbin_centres_R500_err:', rbin_centres_R500_err)
#             print('len(Rmin):', len(Rmin))
#             print('type(Rmin):',type(Rmin))
#             print('Rmin:', Rmin)
#             print('len(Rmax):', len(Rmax))
#             print('Rmax:', Rmax)
#             print('len(rbins_R500):', len(rbins_R500))
#             print('rbins_R500:', rbins_R500)
#             print()
            
            abun = info[:,2].copy()
            abun_err_lo = np.abs(info[:,3].copy())
            abun_err_hi = np.abs(info[:,4].copy())
#             print('len(abun_err_lo):',len(abun_err_lo))
            
            
            if plot_kwargs['renormalize_abundances']:
                if plot_kwargs['normalizations'] is None:
                    raise Exception('Need to provide abundances to normalize by!')

                if type_.lower() == 'metallicity':
                    abun *= info_Lodders09[element]['ZX']/plot_kwargs['normalizations'][element]['ZX']
                    abun_err_lo *= info_Lodders09[element]['ZX']/plot_kwargs['normalizations'][element]['ZX']
                    abun_err_hi *= info_Lodders09[element]['ZX']/plot_kwargs['normalizations'][element]['ZX']
                elif type_.lower() == 'metallicity ratio':
                    abun *= (info_Lodders09[element_num]['ZX']/plot_kwargs['normalizations'][element_num]['ZX']) / (info_Lodders09[element_denom]['ZX']/plot_kwargs['normalizations'][element_denom]['ZX'])
                    abun_err_lo *= (info_Lodders09[element_num]['ZX']/plot_kwargs['normalizations'][element_num]['ZX']) / (info_Lodders09[element_denom]['ZX']/plot_kwargs['normalizations'][element_denom]['ZX'])
                    abun_err_hi *= (info_Lodders09[element_num]['ZX']/plot_kwargs['normalizations'][element_num]['ZX']) / (info_Lodders09[element_denom]['ZX']/plot_kwargs['normalizations'][element_denom]['ZX'])
                    
#             print('len(abun_err_lo):',len(abun_err_lo))

            if 'r200' in fig_kwargs['xaxis']['units'].lower():
#                 rbins_R500 *= Rxxx_conversions['R500/R200']
                rbins_R500 *= Rxxx_conversions[plot_kwargs['sim_for_R_conversion']][plot_kwargs['type_for_R_conversion']]['bin_'+str(bin_)]['R500/R200']
            elif 'r2500' in fig_kwargs['xaxis']['units'].lower():
#                 rbins_R500 *= Rxxx_conversions['R500/R2500']
                rbins_R500 *= Rxxx_conversions[plot_kwargs['sim_for_R_conversion']][plot_kwargs['type_for_R_conversion']]['bin_'+str(bin_)]['R500/R2500']
                
            
            if plot_kwargs['remove_bad_bins']:
                abun, abun_err_lo, abun_err_hi = remove_bad_bins_mernier17(element=element_for_instrument, group=group_name, 
                                                                           vals=[abun, abun_err_lo, abun_err_hi], 
                                                                           remove_or_set_to_nan='set_to_nan')
                
#             print('len(abun_err_lo):',len(abun_err_lo))
                
            if plot_kwargs['exclude_negatives']:
                abun[abun<0] = np.nan
                

            if fig_kwargs['yaxis']['is_log']:
                # Change to derivative method?
                abun_err_lo = np.abs(np.log10(abun) - np.log10(abun - abun_err_lo))
                abun_err_hi = np.abs(np.log10(abun) - np.log10(abun + abun_err_hi))
                abun = np.log10(abun)
    #             abun_err_lo = np.log10(abun_err_lo)
    #             abun_err_hi = np.log10(abun_err_hi)
    
#             print('len(abun_err_lo):',len(abun_err_lo))
    
            mean_abun_err = np.mean(np.array([abun_err_lo, abun_err_hi]), axis=0)
#             print('len(mean_abun_err):', len(mean_abun_err))
#             print('mean_abun_err:', mean_abun_err)
#             print()


            averaging_obs[instrument][group] = {
                'xbins':rbins_R500,
                'values':abun,
                'errs':mean_abun_err,
            }
            
            
        if len(averaging_obs[instrument].keys()) == 0:
            if plot_kwargs['verbose']:
                print(element_name, 'not available from Mernier+2017 for XMM-Newton/%s' % instrument)
        
    
    if len(averaging_obs.keys()) == 0:
        if plot_kwargs['verbose']:
            print(element_name, 'not available from Mernier+2017 for chosen instruments')
        return
    
    
    
    
    ## Get reference radial bins for averaging
    ref_bins = {}
    for instrument, values in averaging_obs.items():
        
        if plot_kwargs['rbins_type'] == 'provide':
            if plot_kwargs['rbins'] is None:
                raise Exception('averaging_bins must be provided if rbins_type is provide')
            averaging_bins = plot_kwargs['rbins']

        elif plot_kwargs['rbins_type'] == 'calculate':
            counter = 0

            for group, info in values.items():

                rbins = info['xbins'].copy()
                Rcentre = (rbins[:-1] + rbins[1:])/2.

                if counter == 0:
                    averaging_bins_start = rbins[0]
                    averaging_bins_end = rbins[-1]
                    data_radial_centres = Rcentre.copy()
                else:
                    averaging_bins_start = min(averaging_bins_start, rbins[0])
                    averaging_bins_end = max(averaging_bins_end, rbins[-1])
                    data_radial_centres = np.append(data_radial_centres, Rcentre)

                counter += 1

            if plot_kwargs['rbins_method'] == 'linear':
                averaging_bins = np.linspace(averaging_bins_start, averaging_bins_end, plot_kwargs['n_rbins'])
            elif plot_kwargs['rbins_method'] == 'log':
                averaging_bins = np.logspace(averaging_bins_start, averaging_bins_end, plot_kwargs['n_rbins'])
            elif plot_kwargs['rbins_method'] == 'equal':
                __, averaging_bins = pd.qcut(data_radial_centres, q=plot_kwargs['n_rbins'], 
                                             labels=None, retbins=True)

        elif plot_kwargs['rbins_type'] == 'sim':
            averaging_bins = sim_rbins
            
        elif plot_kwargs['rbins_type'] == 'dataset':
            if plot_kwargs['rbins_sample'].lower() == 'full':
                averaging_bins = np.array([0, 0.0075, 0.014, 0.02, 0.03, 0.04, 0.055, 0.065, 
                                           0.09, 0.11, 0.135, 0.16, 0.2, 0.23, 0.3, 0.55, 1.22], float)  # Full sample [R/R500]
            elif plot_kwargs['rbins_sample'].lower() == 'cluster':
                averaging_bins = np.array([0, 0.018, 0.04, 0.068, 0.1, 0.18, 0.24, 0.34, 0.5, 1.22], float) # Cluster sub-sample [R/R500]
            elif plot_kwargs['rbins_sample'].lower() == 'group':
                averaging_bins = np.array([0, 0.009, 0.024, 0.042, 0.064, 0.1, 0.15, 0.26, 0.97], float) # Group sub-sample [R/R500]
            else:
                raise Exception("averaging_bins_sample must be 'full', 'cluster', or 'groups'")
                
            if 'r200' in fig_kwargs['xaxis']['units'].lower():
#                 averaging_bins *= Rxxx_conversions['R500/R200']
                averaging_bins *= Rxxx_conversions[plot_kwargs['sim_for_R_conversion']][plot_kwargs['type_for_R_conversion']]['bin_'+str(bin_)]['R500/R200']
            elif 'r2500' in fig_kwargs['xaxis']['units'].lower():
#                 averaging_bins *= Rxxx_conversions['R500/R2500']
                averaging_bins *= Rxxx_conversions[plot_kwargs['sim_for_R_conversion']][plot_kwargs['type_for_R_conversion']]['bin_'+str(bin_)]['R500/R2500']

        else:
            raise Exception("rbins_type must be 'provide', 'calculate', or 'sim'")
            
        ref_bins[instrument] = averaging_bins
        
        
    
    #for idx in range(len(averaging_obs.keys())):
    for instrument, obs in averaging_obs.items():
        averaging_bins = ref_bins[instrument]
        label = labels[instrument]

        color_idx = list(plot_kwargs['instruments'][element_for_instrument]).index(instrument)
        color = plot_kwargs['colors'][element_for_instrument][color_idx]

        
        if plot_kwargs['avg_type'].lower() == 'mean':
            ## Average profiles according to Mernier+2017 equation 3
            result = weighted_mean_of_profiles(averaging_bins, obs)#, calc_kwargs=plot_kwargs)
            averaged_profile = result[:,0]
            std_profile = result[:,1]
            if fig_kwargs['yaxis']['is_log']:
                averaged_limits = [np.log10(10**averaged_profile - 10**std_profile), 
                                   np.log10(10**averaged_profile + 10**std_profile)]
            else:
                averaged_limits = [averaged_profile - std_profile, 
                                   averaged_profile + std_profile]
        elif plot_kwargs['avg_type'].lower() == 'median':
            result = weighted_quantiles_of_profiles(averaging_bins, obs, calc_kwargs=plot_kwargs)
            closest_idx_to_middle = np.argmin(np.abs(np.array(plot_kwargs['quantiles']) - 0.5))
            averaged_profile = result[:,closest_idx_to_middle]
            averaged_limits = [result[:,0], result[:,-1]]
#             print()
#             print('result:',result)
#             print('closest_idx_to_middle:',closest_idx_to_middle)
#             print('averaged_profile:',averaged_profile)
#             print('averaged_limits:',averaged_limits)
#             print()
        else:
            raise Exception('avg_type must be mean or median')


        ## Plot averaged profile with scatter
        averaging_bin_centres = (averaging_bins[:-1] + averaging_bins[1:])/2.
        if fig_kwargs['xaxis']['is_log']:
            averaging_bin_centres = np.log10(averaging_bin_centres)

        x_values = averaging_bin_centres
        y_values = averaged_profile
        y_errs = averaged_limits


        axes.fill_between(x_values, y_errs[0], y_errs[1],
                          color=color, 
                          alpha=plot_kwargs['alpha'], 
                          label='fill!'+label, 
                          zorder=plot_kwargs['zorder'])
        axes.plot(x_values, y_values, 
                  color=color, 
                  marker=plot_kwargs['marker'], 
                  ls=plot_kwargs['ls'],
                  lw=plot_kwargs['lw'], 
                  ms=plot_kwargs['ms'], 
                  mec=plot_kwargs['mec'], 
                  mew=plot_kwargs['mew'], 
                  label='line!'+label,
                  zorder=plot_kwargs['zorder'])


    if plot_kwargs['show_extra_label']:
        axes.plot([], [], 
              color=plot_kwargs['color'], 
              marker=plot_kwargs['marker'], 
              ls=plot_kwargs['ls'],
              lw=plot_kwargs['lw'], 
              ms=plot_kwargs['ms'], 
              mec=plot_kwargs['mec'], 
              mew=plot_kwargs['mew'], 
              label='line!'+plot_kwargs['extra_label'],
              zorder=plot_kwargs['zorder'])
    return






# def plot_combined_stacks_mernier17(type_='metallicity', axes=plt, 
#                                        bin_low=None, bin_high=None, sim_rbins=None,
#                                        element=None, element_num=None, element_denom=None,
#                                        plot_kwargs=None, fig_kwargs=None):
    
#     ## Mernier+2017 observations from CHEERS sample
#     ## Abundance profiles of each group
#     ## Abundances in units of Lodders+2009
    
#     if not plot_kwargs['plot']:
#         if plot_kwargs['verbose']:
#             print('Not plotting Mernier+2017 average abundances')
#         return        
    
#     if type_.lower() == 'metallicity':
#         element_name = element
#         element_for_instrument = element
#     elif type_.lower() == 'metallicity ratio':
#         element_name = element_num + '/' + element_denom
#         element_for_instrument = element_num
#     else:
#         raise Exception("type_ needs to be 'metallicity' or 'metallicity ratio'")
        
#     if element_for_instrument not in plot_kwargs['instruments'].keys():
#         if plot_kwargs['verbose']:
#             print('%s not available from Mernier+2017' % element_for_instrument)
#         return        
        
    
        
#     labels = {}
#     for instruments in plot_kwargs['instruments'][element_for_instrument]:
#         combined_instruments = ''
#         if plot_kwargs['include_citation']:
#             instruments_printable_name = '#data_mernier17##Combined Stacked Groups (Mernier+17): '
#         else:
#             instruments_printable_name = '#data_mernier17##Combined Stacked Groups: '
            
#         for ii in range(len(instruments)):
#             instrument = instruments[ii]

#             combined_instruments += obs_mernier17['instruments'][instrument]
#             instruments_printable_name += obs_mernier17['instruments'][instrument]
                
#             if ii == len(instruments)-2:
#                 combined_instruments += ' and '
#                 instruments_printable_name += ' and '
#             elif ii != len(instruments)-1:
#                 combined_instruments += ', '
#                 instruments_printable_name += ', '

#         if not plot_kwargs['print_labels']:
#             instruments_printable_name = None

#         labels[combined_instruments] = instruments_printable_name
        
        
#     data_bin_prop = None
#     use_data_bin_prop = plot_kwargs['use_bin_prop']
#     dataset = plot_kwargs['M500_dataset']
#     if fig_kwargs['binning']['prop'][0] == 'M' and fig_kwargs['binning']['prop'][1] == '500':
#         data_bin_prop = 'M500'
#     elif bin_prop[0] == 'T':
#         data_bin_prop = 'T'
#         dataset = 'best'
#     else:
#         use_data_bin_prop = False
        
        
#     ## Get exact observations to average
#     averaging_obs = {}
#     for instrument in plot_kwargs['instruments'][element_for_instrument]:
#         try:
#             obs = obs_mernier17['individual'][type_][element_name+'_'+instrument]
#         except:
#             if plot_kwargs['verbose']:
#                 print(element_name + ' not available from Mernier+2017 with XMM-Newton/%s' instrument)
#             continue
            
#         averaging_obs[instrument] = {}
            
#         for group, info in obs.items():
#             group_name = correct_group_name_mernier17(group)
            
#             if plot_kwargs['remove_too_good_obs']:
#                 remove = remove_too_good_obs_mernier17(element=element_for_instrument, group=group_name)
#                 if remove:
#                     if plot_kwargs['verbose']:
#                         print('Removing %s from %s Mernier+2017 average profiles' % (group_name, element_for_instrument))
#                     continue

#             if use_data_bin_prop:
#                 ## Remove groups that do not have bin_prop (eg. M500, kT) values
#                 if group_name not in obs_mernier17['group_properties'][data_bin_prop][dataset].keys():
#                     continue
#                 ## Remove groups outside of bin_prop bin
#                 if (obs_mernier17['group_properties'][data_bin_prop][dataset][group_name] < pnb.array.SimArray(1e13,units='Msol') and 
#                     bin_high > 10**13.5): ## need to change this to be not specifically mass, but any binning prop (so just check whether it is the first or last bin --> how to do that?)
#                     continue
#                 elif (obs_mernier17['group_properties'][data_bin_prop][dataset][group_name] > pnb.array.SimArray(1e15,units='Msol') and 
#                       bin_low < 1e14):
#                     continue
#                 elif not include_in_bin(obs_mernier17['group_properties'][data_bin_prop][dataset][group_name], bin_low, bin_high, 
#                                     frac=plot_kwargs['limit_frac'], transform_func=lambda x:np.log10(x)):
#                     continue
                

#             rbin_centres_R500 = obs[:,0].copy()
#             rbin_centres_R500_err = obs[:,1].copy()
#             Rmin = rad - rad_err
#             Rmax = rad + rad_err
#             rbins_R500 = np.unique(np.append(Rmin, Rmax))
            
#             abun = obs[:,2].copy()
#             abun_err_lo = np.abs(obs[:,3].copy())
#             abun_err_hi = np.abs(obs[:,4].copy())
            
            
#             if plot_kwargs['renormalize_abundances']:
#                 if plot_kwargs['normalizations'] is None:
#                     raise Exception('Need to provide abundances to normalize by!')

#                 if type_.lower() == 'metallicity':
#                     abun *= info_Lodders09[element]['ZX']/plot_kwargs['normalizations'][element]['ZX']
#                     abun_err_lo *= info_Lodders09[element]['ZX']/plot_kwargs['normalizations'][element]['ZX']
#                     abun_err_hi *= info_Lodders09[element]['ZX']/plot_kwargs['normalizations'][element]['ZX']
#                 elif type_.lower() == 'metallicity ratio':
#                     abun *= (info_Lodders09[element_num]['ZX']/plot_kwargs['normalizations'][element_num]['ZX']) / (info_Lodders09[element_denom]['ZX']/plot_kwargs['normalizations'][element_denom]['ZX'])
#                     abun_err_lo *= (info_Lodders09[element_num]['ZX']/plot_kwargs['normalizations'][element_num]['ZX']) / (info_Lodders09[element_denom]['ZX']/plot_kwargs['normalizations'][element_denom]['ZX'])
#                     abun_err_hi *= (info_Lodders09[element_num]['ZX']/plot_kwargs['normalizations'][element_num]['ZX']) / (info_Lodders09[element_denom]['ZX']/plot_kwargs['normalizations'][element_denom]['ZX'])

#             if 'r200' in fig_kwargs['xaxis']['units'].lower():
#                 rbins_R500 *= Rxxx_conversions['R500/R200']
#             elif 'r2500' in fig_kwargs['xaxis']['units'].lower():
#                 rbins_R500 *= Rxxx_conversions['R500/R2500']
                
            
#             if plot_kwargs['remove_bad_bins']:
#                 abun, abun_err_lo, abun_err_hi = remove_bad_bins_mernier17(element=element_for_instrument, group=group_name, 
#                                                                            vals=[abun, abun_err_lo, abun_err_hi], 
#                                                                            remove_or_set_to_nan='set_to_nan')

#             if fig_kwargs['yaxis']['is_log']:
#                 # Change to derivative method?
#                 abun_err_lo = np.abs(np.log10(abun) - np.log10(abun - abun_err_lo))
#                 abun_err_hi = np.abs(np.log10(abun) - np.log10(abun + abun_err_hi))
#                 abun = np.log10(abun)
#     #             abun_err_lo = np.log10(abun_err_lo)
#     #             abun_err_hi = np.log10(abun_err_hi)
    
#             mean_abun_err = np.mean(np.array([abun_err_lo, abun_err_hi]), axis=0)


#             averaging_obs[instrument][group] = {
#                 'xbins':rbins_R500,
#                 'values':abun,
#                 'errs':mean_abun_err,
#             }
            
            
#         if len(averaging_obs[instrument].keys()) == 0:
#             if plot_kwargs['verbose']:
#                 print(element_name, 'not available from Mernier+2017 for XMM-Newton/%s' % instrument)
        
    
#     if len(averaging_obs.keys()) == 0:
#         if plot_kwargs['verbose']:
#             print(element_name, 'not available from Mernier+2017 for chosen instruments')
#         return
    
    
    
    
#     ## Get reference radial bins for averaging
#     ref_bins = {}
#     for instrument, values in averaging_obs.items():
        
#         if plot_kwargs['rbins_type'] == 'provide':
#             if plot_kwargs['rbins'] is None:
#                 raise Exception('averaging_bins must be provided if rbins_type is provide')
#             averaging_bins = plot_kwargs['rbins']

#         elif plot_kwargs['rbins_type'] == 'calculate':
#             counter = 0

#             for group, info in values.items():

#                 rbins = info['xbins'].copy()
#                 Rcentre = (rbins[:-1] + rbins[1:])/2.

#                 if counter == 0:
#                     averaging_bins_start = rbins[0]
#                     averaging_bins_end = rbins[-1]
#                     data_radial_centres = Rcentre.copy()
#                 else:
#                     averaging_bins_start = min(averaging_bins_start, rbins[0])
#                     averaging_bins_end = max(averaging_bins_end, rbins[-1])
#                     data_radial_centres = np.append(data_radial_centres, Rcentre)

#                 counter += 1

#             if plot_kwargs['rbins_method'] == 'linear':
#                 averaging_bins = np.linspace(averaging_bins_start, averaging_bins_end, plot_kwargs['n_rbins'])
#             elif plot_kwargs['rbins_method'] == 'log':
#                 averaging_bins = np.logspace(averaging_bins_start, averaging_bins_end, plot_kwargs['n_rbins'])
#             elif plot_kwargs['rbins_method'] == 'equal':
#                 __, averaging_bins = pd.qcut(data_radial_centres, q=plot_kwargs['n_rbins'], 
#                                              labels=None, retbins=True)

#         elif plot_kwargs['rbins_type'] == 'sim':
#             averaging_bins = sim_rbins
            
#         elif plot_kwargs['rbins_type'] == 'dataset':
#             if plot_kwargs['rbins_sample'].lower() == 'full':
#                 averaging_bins = np.array([0, 0.0075, 0.014, 0.02, 0.03, 0.04, 0.055, 0.065, 
#                                            0.09, 0.11, 0.135, 0.16, 0.2, 0.23, 0.3, 0.55, 1.22], float)  # Full sample [R/R500]
#             elif plot_kwargs['rbins_sample'].lower() == 'cluster':
#                 averaging_bins = np.array([0, 0.018, 0.04, 0.068, 0.1, 0.18, 0.24, 0.34, 0.5, 1.22], float) # Cluster sub-sample [R/R500]
#             elif plot_kwargs['rbins_sample'].lower() == 'group':
#                 averaging_bins = np.array([0, 0.009, 0.024, 0.042, 0.064, 0.1, 0.15, 0.26, 0.97], float) # Group sub-sample [R/R500]
#             else:
#                 raise Exception("averaging_bins_sample must be 'full', 'cluster', or 'groups'")
                
#             if 'r200' in fig_kwargs['xaxis']['units'].lower():
#                 averaging_bins *= Rxxx_conversions['R500/R200']
#             elif 'r2500' in fig_kwargs['xaxis']['units'].lower():
#                 averaging_bins *= Rxxx_conversions['R500/R2500']

#         else:
#             raise Exception("rbins_type must be 'provide', 'calculate', or 'sim'")
            
#         ref_bins[instrument] = averaging_bins
        
        
    
#     #for idx in range(len(averaging_obs.keys())):
#     for instrument, obs in averaging_obs.items():
#         averaging_bins = ref_bins[instrument]
#         label = labels[instrument]

#         color_idx = list(plot_kwargs['instruments'][element_for_instrument]).index(instrument)
#         color = plot_kwargs['colors'][element_for_instrument][idx]

        
#         ## Average profiles according to Mernier+2017 equation 3
#         averaged_profile, averaged_scatter = weighted_mean_of_profiles(averaging_bins, obs, calc_kwargs=plot_kwargs)


#         ## Plot averaged profile with scatter
#         averaging_bin_centres = (averaging_bins[:-1] + averaging_bins[1:])/2.
#         if fig_kwargs['xaxis']['is_log']:
#             averaging_bin_centres = np.log10(averaging_bin_centres)

#         x_values = averaging_bin_centres
#         y_values = averaged_profile
#         y_errs = averaged_scatter


#         axes.fill_between(x_values, y_values-y_errs, y_values+y_errs,
#                           color=color, 
#                           alpha=plot_kwargs['alpha'], 
#                           label='fill!'+label, 
#                           zorder=plot_kwargs['zorder'])
#         axes.plot(x_values, y_values, 
#                   color=color, 
#                   marker=plot_kwargs['marker'], 
#                   ls=plot_kwargs['ls'],
#                   lw=plot_kwargs['lw'], 
#                   ms=plot_kwargs['ms'], 
#                   mec=plot_kwargs['mec'], 
#                   mew=plot_kwargs['mew'], 
#                   label='line!'+label,
#                   zorder=plot_kwargs['zorder'])


#     if plot_kwargs['show_extra_label']:
#         axes.plot([], [], 
#               color=plot_kwargs['color'], 
#               marker=plot_kwargs['marker'], 
#               ls=plot_kwargs['ls'],
#               lw=plot_kwargs['lw'], 
#               ms=plot_kwargs['ms'], 
#               mec=plot_kwargs['mec'], 
#               mew=plot_kwargs['mew'], 
#               label='line!'+plot_kwargs['extra_label'],
#               zorder=plot_kwargs['zorder'])
#     return





# def plot_combined_stacks_mernier17(type_='metallicity', axes=plt, bin_low=None, bin_high=None, 
#                                    element=None, element_num=None, element_denom=None, sim_rbins=None,
#                                    plot_kwargs=None, fig_kwargs=None):


#     if not plot_kwargs['plot']:
#         if plot_kwargs['verbose']:
#             print('Not plotting Mernier+17 combined stacks')
#         return
    
    
#     if type_.lower() == 'metallicity':
#         element_name = element
#         element_for_instrument = element
#     elif type_.lower() == 'metallicity ratio':
#         element_name = element_num + '/' + element_denom
#         element_for_instrument = element_num
#     else:
#         raise Exception("type_ needs to be 'metallicity' or 'metallicity ratio'")
        
#     if element_for_instrument not in plot_kwargs['instruments'].keys():
#         if plot_kwargs['verbose']:
#             print('%s not in Mernier+17' % element_for_instrument)
#         return
    
    
    
    
#     labels_ = {}
#     for instruments in plot_kwargs['instruments'][element_for_instrument]:
#         combined_instruments = ''
#         if plot_kwargs['include_citation']:
#             instruments_printable_name = '#data_mernier17##Combined Stacked Groups (Mernier+17): '
#         else:
#             instruments_printable_name = '#data_mernier17##Combined Stacked Groups: '
            
#         for ii in range(len(instruments)):
#             instrument = instruments[ii]

#             combined_instruments += obs_mernier17['instruments'][instrument]
#             instruments_printable_name += obs_mernier17['instruments'][instrument]
                
#             if ii == len(instruments)-2:
#                 combined_instruments += ' and '
#                 instruments_printable_name += ' and '
#             elif ii != len(instruments)-1:
#                 combined_instruments += ', '
#                 instruments_printable_name += ', '

#         if not plot_kwargs['print_labels']:
#             instruments_printable_name = None

#         labels_[combined_instruments] = instruments_printable_name
    
    
#     data_bin_prop = None
#     use_data_bin_prop = plot_kwargs['use_bin_prop']
#     dataset = plot_kwargs['M500_dataset']
#     if fig_kwargs['binning']['prop'][0] == 'M' and fig_kwargs['binning']['prop'][1] == '500':
#         data_bin_prop = 'M500'
#     elif bin_prop[0] == 'T':
#         data_bin_prop = 'T'
#         dataset = 'best'
#     else:
#         use_data_bin_prop = False
        
        
#     averaging_bins_dict = {}
    
#     for instruments in plot_kwargs['instruments'][element_for_instrument]:
        
#         combined_instruments = ''
#         for ii in range(len(instruments)):
#             instrument = instruments[ii]
#             combined_instruments += obs_mernier17['instruments'][instrument]
                
#             if ii == len(instruments)-2:
#                 combined_instruments += ' and '
#             elif ii != len(instruments)-1:
#                 combined_instruments += ', '
        
#         instrument = instruments[0]

#         try:
#             obs = obs_mernier17['individual'][type_][element_name+'_'+instrument]
#         except:
#             if plot_kwargs['verbose']:
#                 print(element_name + ' not available from Mernier+17')
#             continue

    
#         if plot_kwargs['rbins_type'] == 'provide':
#             if plot_kwargs['rbins'] is None:
#                 raise Exception('averaging_bins must be provided')

#             averaging_bins = plot_kwargs['rbins']

#         elif plot_kwargs['rbins_type'] == 'calculate':
            
#             counter = 0
#             for group, vals in obs.items():
#                 group_ = correct_group_name_mernier17(group)
                
#                 if plot_kwargs['remove_too_good_obs']:
#                     remove = remove_too_good_obs_mernier17(element=element_for_instrument, group=group_)
#                     if remove:
#                         if plot_kwargs['verbose']:
#                             print('Removing %s from %s individual profiles' % (group_, element_for_instrument))
#                         continue

#                 if use_data_bin_prop:
#                     ## Remove groups that do not have M500 values
#                     if group_ not in obs_mernier17['individual'][data_bin_prop][dataset].keys():
#                         continue
#                     if not include_in_bin(obs_mernier17['individual'][data_bin_prop][dataset][group_], bin_low, bin_high, 
#                                       frac=plot_kwargs['limit_frac'], transform_func=lambda x:np.log10(x)):
#                         continue


#                 rad = vals[:,0].copy()
#                 rad_err = vals[:,1].copy()


#                 Rmin = rad - rad_err
#                 Rmax = rad + rad_err
#                 Rcentre = (Rmin+Rmax)/2.
                
#                 if plot_kwargs['remove_bad_bins']:
#                     Rcentre, Rmin, Rmax = remove_bad_bins_mernier17(element=element_for_instrument, group=group_, 
#                                                                     vals=[Rcentre, Rmin, Rmax])

#                 if counter == 0:
#                     averaging_bins_start = Rmin[0]
#                     averaging_bins_end = Rmax[-1]
#                     data_radial_centres = Rcentre.copy()
#                 else:
#                     averaging_bins_start = min(averaging_bins_start, Rmin[0])
#                     averaging_bins_end = max(averaging_bins_end, Rmax[-1])
#                     data_radial_centres = np.append(data_radial_centres, Rcentre)

#                 counter += 1


#             if plot_kwargs['rbins_method'] == 'linear':
#                 averaging_bins = np.linspace(averaging_bins_start, averaging_bins_end, plot_kwargs['n_rbins'])
#             elif plot_kwargs['rbins_method'] == 'log':
#                 averaging_bins = np.logspace(averaging_bins_start, averaging_bins_end, plot_kwargs['n_rbins'])
#             elif plot_kwargs['rbins_method'] == 'equal':
#                 __, averaging_bins = pd.qcut(data_radial_centres, q=plot_kwargs['n_rbins'], 
#                                              labels=None, retbins=True)


#         elif plot_kwargs['rbins_type'] == 'sim':
#             averaging_bins = sim_rbins

#         elif plot_kwargs['rbins_type'] == 'dataset':
#             if plot_kwargs['rbins_sample'].lower() == 'full':
#                 averaging_bins = np.array([0, 0.0075, 0.014, 0.02, 0.03, 0.04, 0.055, 0.065, 
#                                            0.09, 0.11, 0.135, 0.16, 0.2, 0.23, 0.3, 0.55, 1.22], float)  # Full sample [R/R500]
#             elif plot_kwargs['rbins_sample'].lower() == 'cluster':
#                 averaging_bins = np.array([0, 0.018, 0.04, 0.068, 0.1, 0.18, 0.24, 0.34, 0.5, 1.22], float) # Cluster sub-sample [R/R500]
#             elif plot_kwargs['rbins_sample'].lower() == 'group':
#                 averaging_bins = np.array([0, 0.009, 0.024, 0.042, 0.064, 0.1, 0.15, 0.26, 0.97], float) # Group sub-sample [R/R500]
#             else:
#                 raise Exception("averaging_bins_sample must be 'full', 'cluster', or 'groups'")

#             if fig_kwargs['xaxis']['units'].lower() == 'r200':
#                 averaging_bins *= Rxxx_conversions['R500/R200']
#             if fig_kwargs['xaxis']['units'].lower() == 'r2500':
#                 averaging_bins *= Rxxx_conversions['R500/R2500']


#         else:
#             raise Exception("rbins_type must be 'provide', 'calculate', 'sim', or 'dataset'")
            
            
#         averaging_bins_dict[combined_instruments] = averaging_bins


    
#     ## Average profiles according to Mernier+2017 equation 3

#     for idx in range(len(plot_kwargs['instruments'][element_for_instrument])):
#         instruments = plot_kwargs['instruments'][element_for_instrument][idx]
#         color = plot_kwargs['colors'][element_for_instrument][idx]
        
#         combined_instruments = ''
#         for ii in range(len(instruments)):
#             instrument = instruments[ii]
#             combined_instruments += obs_mernier17['instruments'][instrument]
            
#             if ii == len(instruments)-2:
#                 combined_instruments += ' and '
#             elif ii != len(instruments)-1:
#                 combined_instruments += ', '
        
#         numerator = 0
#         denominator = 0
        
#         for instrument_idx in range(len(instruments)):
#             instrument = instruments[instrument_idx]

#             try:
#                 obs = obs_mernier17['individual'][type_][element_name+'_'+instrument]
#             except:
#                 if plot_kwargs['verbose']:
#                     print(element_name + ' not available from Mernier+17')
#                 continue



#             averaged_profile, averaged_scatter = calc_stack_mernier17(averaging_bins_dict, instrument, obs, type_=type_, 
#                                                                     element=element, element_num=element_num, element_denom=element_denom,
#                                                                       element_for_instrument=element_for_instrument,
#                                                                       instrument_for_rbins=combined_instruments,
#                                                                       use_data_bin_prop=use_data_bin_prop, data_bin_prop=data_bin_prop, 
#                                                                       dataset=dataset, bin_low=bin_low, bin_high=bin_high, 
#                                                                       plot_kwargs=plot_kwargs, fig_kwargs=fig_kwargs)
        
#             numerator += averaged_profile/averaged_scatter**2
#             denominator += averaged_scatter**(-2)
            
#             if instrument_idx == 0:
#                 upper_bounds = averaged_profile + averaged_scatter
#                 lower_bounds = averaged_profile - averaged_scatter
#             else:
#                 both_upper_bounds = np.array([upper_bounds, averaged_profile + averaged_scatter])
#                 both_lower_bounds = np.array([lower_bounds, averaged_profile - averaged_scatter])
                
#                 upper_bounds = np.max(both_upper_bounds, axis=0)
#                 lower_bounds = np.min(both_lower_bounds, axis=0)
                
                
#         combined_avg_profile = numerator/denominator


#         ## Plot averaged profile with scatter
#         averaging_bin_centres = (averaging_bins_dict[combined_instruments][:-1] + averaging_bins_dict[combined_instruments][1:])/2.

#         if fig_kwargs['xaxis']['is_log']:
#             averaging_bin_centres = np.log10(averaging_bin_centres)


#         axes.fill_between(averaging_bin_centres, lower_bounds, upper_bounds,
#                           color=color, 
#                           alpha=plot_kwargs['alpha'], 
#                           label='fill!'+labels_[combined_instruments], 
#                           zorder=plot_kwargs['zorder'])
#         axes.plot(averaging_bin_centres, combined_avg_profile, 
#                   color=color, 
#                   marker=plot_kwargs['marker'], 
#                   lw=plot_kwargs['lw'], 
#                   ms=plot_kwargs['ms'], 
#                   mec=plot_kwargs['mec'], 
#                   mew=plot_kwargs['mew'], 
#                   label='line!'+labels_[combined_instruments],
#                   zorder=plot_kwargs['zorder'])


#     if plot_kwargs['show_extra_label']:
#         axes.plot([], [], 
#                   color=plot_kwargs['color'], 
#                   marker=plot_kwargs['marker'], 
#                   lw=plot_kwargs['lw'], 
#                   ms=plot_kwargs['ms'], 
#                   mec=plot_kwargs['mec'], 
#                   mew=plot_kwargs['mew'], 
#                   label='line!'+plot_kwargs['extra_label'],
#                   zorder=plot_kwargs['zorder'])
#     return





    
    


def plot_avg_metallicity_mernier18b(type_='metallicity', axes=plt, bin_=None,
                                    element=None, element_num=None, element_denom=None,
                                    plot_kwargs=None, fig_kwargs=None):
    
    ## Mernier+2018b CHEERS sample
    ## Lodders+09 solar abundancess
    ## Core abundances within 0.05R500 and 0.2R500 (only for hotter groups for which it could be measured)
    ## Same as Mernier+2016a, but with updated spectral code
    ## Significantly affects Cr/Fe and Ni/Fe metallicities
    ## Like Mernier+2017, uses XMM-Newton/EPIC MOS 0.5/0.6-10 keV for global fits
    
    if not plot_kwargs['plot']:
        if plot_kwargs['verbose']:
            print('Not plotting Mernier+2018b average core abundances')
        return
    
    if type_.lower() == 'metallicity':
        element_name = element
    elif type_.lower() == 'metallicity ratio':
        element_name = element_num + '/' + element_denom
    else:
        raise Exception("type_ needs to be 'metallicity' or 'metallicity ratio'")
        
    if plot_kwargs['print_labels']:
        if plot_kwargs['include_citation']:
            labels = {
                'avg':'#data_mernier18b##Stacked Groups (Mernier+18b)'
            }
        else:
            labels = {
                'avg':'#data_mernier18b##Stacked Groups'
            }
    else:
        labels = {
            'avg':None
        }
        
        
    try:
        obs = obs_mernier18b['average']['metallicity'][element_name]
    except:
        if plot_kwargs['verbose']:
            print(element_name + ' not available from Mernier+2018b')
        return
    
    rbins_R500 = obs['rbins_R500'].copy()
    abun = obs['value'].copy()
    abun_err_lo = np.abs(obs['err_lo'].copy())
    abun_err_hi = np.abs(obs['err_hi'].copy())
    
    
    if 'r200' in fig_kwargs['xaxis']['units'].lower():
#         rbins_R500 *= Rxxx_conversions['R500/R200']
        rbins_R500 *= Rxxx_conversions[plot_kwargs['sim_for_R_conversion']][plot_kwargs['type_for_R_conversion']]['bin_'+str(bin_)]['R500/R200']
    elif 'r2500' in fig_kwargs['xaxis']['units'].lower():
#         rbins_R500 *= Rxxx_conversions['R500/R2500']
        rbins_R500 *= Rxxx_conversions[plot_kwargs['sim_for_R_conversion']][plot_kwargs['type_for_R_conversion']]['bin_'+str(bin_)]['R500/R2500']
        
    if fig_kwargs['xaxis']['is_log']:
        rbins_R500 = np.log10(rbins_R500)
    
    
    if plot_kwargs['renormalize_abundances']:
        if plot_kwargs['normalizations'] is None:
            raise Exception('Need to provide abundances to normalize by!')

        if type_.lower() == 'metallicity':
            abun *= info_Lodders09[element]['ZX']/plot_kwargs['normalizations'][element]['ZX']
            abun_err_lo *= info_Lodders09[element]['ZX']/plot_kwargs['normalizations'][element]['ZX']
            abun_err_hi *= info_Lodders09[element]['ZX']/plot_kwargs['normalizations'][element]['ZX']
        elif type_.lower() == 'metallicity ratio':
            abun *= (info_Lodders09[element_num]['ZX']/plot_kwargs['normalizations'][element_num]['ZX']) / (info_Lodders09[element_denom]['ZX']/plot_kwargs['normalizations'][element_denom]['ZX'])
            abun_err_lo *= (info_Lodders09[element_num]['ZX']/plot_kwargs['normalizations'][element_num]['ZX']) / (info_Lodders09[element_denom]['ZX']/plot_kwargs['normalizations'][element_denom]['ZX'])
            abun_err_hi *= (info_Lodders09[element_num]['ZX']/plot_kwargs['normalizations'][element_num]['ZX']) / (info_Lodders09[element_denom]['ZX']/plot_kwargs['normalizations'][element_denom]['ZX'])

    if fig_kwargs['yaxis']['is_log']:
        # Change to derivative method?
        abun_err_lo = np.abs(np.log10(abun) - np.log10(abun - abun_err_lo))
        abun_err_hi = np.abs(np.log10(abun) - np.log10(abun + abun_err_hi))
        abun = np.log10(abun)
#         abun_err_lo = np.log10(abun_err_lo)
#         abun_err_hi = np.log10(abun_err_hi)
        

    
    ## Plot average profile with scatter
    x_values = np.array([rbins_R500[0]] + list(np.repeat(rbins_R500[1:-1], repeats=2)) + [rbins_R500[-1]])
    y_values = np.array([abun[0]] + list(np.repeat(abun[1:-1], repeats=2)) + [abun[-1]])
    y_errs_lo = np.array([abun_err_lo[0]] + list(np.repeat(abun_err_lo[1:-1], repeats=2)) + [abun_err_lo[-1]])
    y_errs_hi = np.array([abun_err_hi[0]] + list(np.repeat(abun_err_hi[1:-1], repeats=2)) + [abun_err_hi[-1]])


    axes.fill_between(x_values, y_values-y_errs_lo, y_values+y_errs_hi,
                      color=plot_kwargs['color'], 
                      alpha=plot_kwargs['alpha'], 
                      label='fill!'+labels['avg'],
                      zorder=plot_kwargs['zorder'])
    axes.plot(x_values, y_values, 
              color=plot_kwargs['color'], 
              marker=plot_kwargs['marker'], 
              ls=plot_kwargs['ls'],
              lw=plot_kwargs['lw'], 
              ms=plot_kwargs['ms'], 
              mec=plot_kwargs['mec'], 
              mew=plot_kwargs['mew'], 
              label='line!'+labels['avg'],
              zorder=plot_kwargs['zorder'])


    if plot_kwargs['show_extra_label']:
        axes.plot([], [], 
              color=plot_kwargs['color'], 
              marker=plot_kwargs['marker'], 
              ls=plot_kwargs['ls'],
              lw=plot_kwargs['lw'], 
              ms=plot_kwargs['ms'], 
              mec=plot_kwargs['mec'], 
              mew=plot_kwargs['mew'], 
              label='line!'+plot_kwargs['extra_label'],
              zorder=plot_kwargs['zorder'])
    return





def plot_avg_metallicity_mao19(type_='metallicity', axes=plt, bin_=None,
                               bin_low=None, bin_high=None, sim_rbins=None,
                               element=None, element_num=None, element_denom=None,
                               plot_kwargs=None, fig_kwargs=None):
    
    ## Mao+2019 observations from CHEERS sample
    ## Just core metallicity of each group
    ## Have size of core for each group in units of R/R500 from paper, so can just treat as a profile with one bin
    ## Abundances in units of Lodders+2009
    
    if not plot_kwargs['plot']:
        if plot_kwargs['verbose']:
            print('Not plotting Mao+2019 average core abundances')
        return
    
    if type_.lower() == 'metallicity':
        element_name = element
    elif type_.lower() == 'metallicity ratio':
        element_name = element_num + '/' + element_denom
    else:
        raise Exception("type_ needs to be 'metallicity' or 'metallicity ratio'")
        
    if plot_kwargs['print_labels']:
        if plot_kwargs['include_citation']:
            labels = {
                'avg':'#data_mao19##Stacked Groups (Mao+19)'
            }
        else:
            labels = {
                'avg':'#data_mao19##Stacked Groups'
            }
    else:
        labels = {
            'avg':None
        }
        
        
    data_bin_prop = None
    use_data_bin_prop = plot_kwargs['use_bin_prop']
    if fig_kwargs['binning']['prop'][0] == 'M' and fig_kwargs['binning']['prop'][1] == '500':
        data_bin_prop = 'M500'
    elif bin_prop[0] == 'T':
        data_bin_prop = 'kT'
    else:
        use_data_bin_prop = False
        
        
    ## Get exact observations to average
    averaging_obs = {}
    for group, info in obs_mao19['individual'].items():
        try:
            obs = info['metallicity'][plot_kwargs['region']][element_name]
        except:
            continue

        if use_data_bin_prop:
            ## Remove groups that do not have bin_prop (eg. M500, kT) values
            if data_bin_prop not in info.keys():
                continue
            if not include_in_bin(info[data_bin_prop], bin_low, bin_high, 
                                  extra=plot_kwargs['limit_extra'], use_frac=plot_kwargs['use_limit_frac'],
                                  frac=plot_kwargs['limit_frac'], transform_func=lambda x:np.log10(x)):
                continue

                
        rbins_R500 = obs['rbins_R500'].copy()
        abun = obs['value'].copy()
        abun_err_lo = np.abs(obs['err_lo'].copy())
        abun_err_hi = np.abs(obs['err_hi'].copy())

        if plot_kwargs['renormalize_abundances']:
            if plot_kwargs['normalizations'] is None:
                raise Exception('Need to provide abundances to normalize by!')

            if type_.lower() == 'metallicity':
                abun *= info_Lodders09[element]['ZX']/plot_kwargs['normalizations'][element]['ZX']
                abun_err_lo *= info_Lodders09[element]['ZX']/plot_kwargs['normalizations'][element]['ZX']
                abun_err_hi *= info_Lodders09[element]['ZX']/plot_kwargs['normalizations'][element]['ZX']
            elif type_.lower() == 'metallicity ratio':
                abun *= (info_Lodders09[element_num]['ZX']/plot_kwargs['normalizations'][element_num]['ZX']) / (info_Lodders09[element_denom]['ZX']/plot_kwargs['normalizations'][element_denom]['ZX'])
                abun_err_lo *= (info_Lodders09[element_num]['ZX']/plot_kwargs['normalizations'][element_num]['ZX']) / (info_Lodders09[element_denom]['ZX']/plot_kwargs['normalizations'][element_denom]['ZX'])
                abun_err_hi *= (info_Lodders09[element_num]['ZX']/plot_kwargs['normalizations'][element_num]['ZX']) / (info_Lodders09[element_denom]['ZX']/plot_kwargs['normalizations'][element_denom]['ZX'])

        if 'r200' in fig_kwargs['xaxis']['units'].lower():
#             rbins_R500 *= Rxxx_conversions['R500/R200']
            rbins_R500 *= Rxxx_conversions[plot_kwargs['sim_for_R_conversion']][plot_kwargs['type_for_R_conversion']]['bin_'+str(bin_)]['R500/R200']
        elif 'r2500' in fig_kwargs['xaxis']['units'].lower():
#             rbins_R500 *= Rxxx_conversions['R500/R2500']
            rbins_R500 *= Rxxx_conversions[plot_kwargs['sim_for_R_conversion']][plot_kwargs['type_for_R_conversion']]['bin_'+str(bin_)]['R500/R2500']
            
            
        if plot_kwargs['exclude_negatives']:
            abun[abun<0] = np.nan


        if fig_kwargs['yaxis']['is_log']:
            # Change to derivative method?
            abun_err_lo = np.abs(np.log10(abun) - np.log10(abun - abun_err_lo))
            abun_err_hi = np.abs(np.log10(abun) - np.log10(abun + abun_err_hi))
            abun = np.log10(abun)
#             abun_err_lo = np.log10(abun_err_lo)
#             abun_err_hi = np.log10(abun_err_hi)

        mean_abun_err = np.mean(np.array([abun_err_lo, abun_err_hi]), axis=0)

        
        averaging_obs[group] = {
            'xbins':rbins_R500,
            'values':abun,
            'errs':mean_abun_err,
#             'values_err_lo':abun_err_lo,
#             'values_err_hi':abun_err_hi,
        }
        
    
    if len(averaging_obs.keys()) == 0:
        if plot_kwargs['verbose']:
            print(element_name, 'not available from Mao+2019')
        return
        
        
    
    ## Get reference radial bins for averaging
    if plot_kwargs['rbins_type'] == 'provide':
        if plot_kwargs['rbins'] is None:
            raise Exception('averaging_bins must be provided if rbins_type is provide')
        averaging_bins = plot_kwargs['rbins']
        
    elif plot_kwargs['rbins_type'] == 'calculate':
        counter = 0
    
        for group, info in averaging_obs.items():
            
            rbins = info['xbins'].copy()
            Rcentre = (rbins[:-1] + rbins[1:])/2.

            if counter == 0:
                averaging_bins_start = rbins[0]
                averaging_bins_end = rbins[-1]
                data_radial_centres = Rcentre.copy()
            else:
                averaging_bins_start = min(averaging_bins_start, rbins[0])
                averaging_bins_end = max(averaging_bins_end, rbins[-1])
                data_radial_centres = np.append(data_radial_centres, Rcentre)

            counter += 1
            
        if plot_kwargs['rbins_method'] == 'linear':
            averaging_bins = np.linspace(averaging_bins_start, averaging_bins_end, plot_kwargs['n_rbins'])
        elif plot_kwargs['rbins_method'] == 'log':
            averaging_bins = np.logspace(averaging_bins_start, averaging_bins_end, plot_kwargs['n_rbins'])
        elif plot_kwargs['rbins_method'] == 'equal':
            __, averaging_bins = pd.qcut(data_radial_centres, q=plot_kwargs['n_rbins'], 
                                         labels=None, retbins=True)

    elif plot_kwargs['rbins_type'] == 'sim':
        averaging_bins = sim_rbins

    else:
        raise Exception("rbins_type must be 'provide', 'calculate', or 'sim'")
        

    ## Average profiles
    if plot_kwargs['avg_type'].lower() == 'mean':
        ## Average profiles according to Mernier+2017 equation 3
        result = weighted_mean_of_profiles(averaging_bins, averaging_obs)
        averaged_profile = result[:,0]
        std_profile = result[:,1]
        if fig_kwargs['yaxis']['is_log']:
            averaged_limits = [np.log10(10**averaged_profile - 10**std_profile), 
                               np.log10(10**averaged_profile + 10**std_profile)]
        else:
            averaged_limits = [averaged_profile - std_profile, 
                               averaged_profile + std_profile]
    elif plot_kwargs['avg_type'].lower() == 'median':
        result = weighted_quantiles_of_profiles(averaging_bins, averaging_obs, calc_kwargs=plot_kwargs)
        closest_idx_to_middle = np.argmin(np.abs(np.array(plot_kwargs['quantiles']) - 0.5))
        averaged_profile = result[:,closest_idx_to_middle]
        averaged_limits = [result[:,0], result[:,-1]]
    else:
        raise Exception('avg_type must be mean or median')

    
#     if plot_kwargs['avg_type'].lower() == 'mean':
#         ## Average profiles according to Mernier+2017 equation 3
#         result = weighted_mean_of_profiles(averaging_bins, averaging_obs)#, calc_kwargs=plot_kwargs)
#         averaged_profile = result[0]
#         averaged_scatter = [result[1], result[1]]
#     elif plot_kwargs['avg_type'].lower() == 'median':
#         result = weighted_quantiles_of_profiles(averaging_bins, averaging_obs)#, calc_kwargs=plot_kwargs)
#         closest_idx_to_median = np.argmin(np.abs(plot_kwargs['quantiles'] - 0.5))
#         averaged_profile = result[:,closest_idx_to_median]
#         averaged_scatter = [result[0], result[-1]]
#     else:
#         raise Exception('avg_type must be mean or median')


    ## Plot averaged profile with scatter
    if fig_kwargs['xaxis']['is_log']:
        averaging_bins = np.log10(averaging_bins)

    x_values = np.array([averaging_bins[0]] + list(np.repeat(averaging_bins[1:-1], repeats=2)) + [averaging_bins[-1]])
    y_values = np.array([averaged_profile[0]] + list(np.repeat(averaged_profile[1:-1], repeats=2)) + [averaged_profile[-1]])
    y_errs = [
        np.array([averaged_limits[0][0]] + list(np.repeat(averaged_limits[0][1:-1], repeats=2)) + [averaged_limits[0][-1]]),
        np.array([averaged_limits[1][0]] + list(np.repeat(averaged_limits[1][1:-1], repeats=2)) + [averaged_limits[1][-1]]),
             ]


    axes.fill_between(x_values, y_errs[0], y_errs[1],
                      color=plot_kwargs['color'], 
                      alpha=plot_kwargs['alpha'], 
                      label='fill!'+labels['avg'], 
                      zorder=plot_kwargs['zorder'])
    axes.plot(x_values, y_values, 
              color=plot_kwargs['color'], 
              marker=plot_kwargs['marker'], 
              ls=plot_kwargs['ls'],
              lw=plot_kwargs['lw'], 
              ms=plot_kwargs['ms'], 
              mec=plot_kwargs['mec'], 
              mew=plot_kwargs['mew'], 
              label='line!'+labels['avg'],
              zorder=plot_kwargs['zorder'])


    if plot_kwargs['show_extra_label']:
        axes.plot([], [], 
              color=plot_kwargs['color'], 
              marker=plot_kwargs['marker'], 
              ls=plot_kwargs['ls'],
              lw=plot_kwargs['lw'], 
              ms=plot_kwargs['ms'], 
              mec=plot_kwargs['mec'], 
              mew=plot_kwargs['mew'], 
              label='line!'+plot_kwargs['extra_label'],
              zorder=plot_kwargs['zorder'])
    return









def plot_avg_metallicity_ghizzardi21(type_='metallicity', axes=plt, 
                                     bin_=None, bin_low=None, bin_high=None,
                                     element=None, element_num=None, element_denom=None,
                                     plot_kwargs=None, fig_kwargs=None):
    
    ## Ghizzardi+2021 X-COP sample
    ## Anders & Grevesse (1989) solar abundances
    ## Mean abundance profiles from spectral fitting for total sample, and CC and NCC subsamples
    ## X-COP = XMM-Newton Cluster Outskirts Project
    ## EPIC MOS and pn instruments
    ## 0.5-12 keV for spectral fitting
    
    if not plot_kwargs['plot']:
        if plot_kwargs['verbose']:
            print('Not plotting Ghizzardi+2021 average abundance profiles')
        return
    
    if type_.lower() == 'metallicity':
        element_name = element
    elif type_.lower() == 'metallicity ratio':
        element_name = element_num + '/' + element_denom
    else:
        raise Exception("type_ needs to be 'metallicity' or 'metallicity ratio'")
        
    if bin_low >= 1e14 and bin_low <= 4e14 and bin_high >= 8e14 and bin_high <= 1e15:
        pass
    else:
        if plot_kwargs['verbose']:
            print('Ghizzardi+2021 sample outside mass range')
        return
        
    
    if plot_kwargs['print_labels']:
        if plot_kwargs['include_citation']:
            labels = {
                sample:'#data_ghizzardi21##Stacked Groups (%s)' % obs_ghizzardi21['average'][sample]['label'] for sample in obs_ghizzardi21['average'].keys()
            }
        else:
            labels = {
                sample:'#data_ghizzardi21b##Stacked Groups' for sample in obs_ghizzardi21['average'].keys()
            }
    else:
        labels = {
            sample:None for sample in obs_ghizzardi21['average'].keys()
        }
        
        
        
    for sample, color, ls in zip(plot_kwargs['samples'], plot_kwargs['colors'], plot_kwargs['ls']):

        try:
            obs = obs_ghizzardi21['average'][sample]['metallicity'][element_name]
        except:
            if plot_kwargs['verbose']:
                print(element_name + ' not available from Ghizzardi+2021 ' + sample)
            return

        rbins_R500 = obs['rbins_R500'].copy()
        abun = obs['value'].copy()
        abun_err_lo = np.abs(obs['err_lo'].copy())
        abun_err_hi = np.abs(obs['err_hi'].copy())


        if 'r200' in fig_kwargs['xaxis']['units'].lower():
    #         rbins_R500 *= Rxxx_conversions['R500/R200']
            rbins_R500 *= Rxxx_conversions[plot_kwargs['sim_for_R_conversion']][plot_kwargs['type_for_R_conversion']]['bin_'+str(bin_)]['R500/R200']
        elif 'r2500' in fig_kwargs['xaxis']['units'].lower():
    #         rbins_R500 *= Rxxx_conversions['R500/R2500']
            rbins_R500 *= Rxxx_conversions[plot_kwargs['sim_for_R_conversion']][plot_kwargs['type_for_R_conversion']]['bin_'+str(bin_)]['R500/R2500']

        if fig_kwargs['xaxis']['is_log']:
            rbins_R500 = np.log10(rbins_R500)


        if plot_kwargs['renormalize_abundances']:
            if plot_kwargs['normalizations'] is None:
                raise Exception('Need to provide abundances to normalize by!')

            if type_.lower() == 'metallicity':
                abun *= info_AG89[element]['ZX']/plot_kwargs['normalizations'][element]['ZX']
                abun_err_lo *= info_AG89[element]['ZX']/plot_kwargs['normalizations'][element]['ZX']
                abun_err_hi *= info_AG89[element]['ZX']/plot_kwargs['normalizations'][element]['ZX']
            elif type_.lower() == 'metallicity ratio':
                abun *= (info_AG89[element_num]['ZX']/plot_kwargs['normalizations'][element_num]['ZX']) / (info_AG89[element_denom]['ZX']/plot_kwargs['normalizations'][element_denom]['ZX'])
                abun_err_lo *= (info_AG89[element_num]['ZX']/plot_kwargs['normalizations'][element_num]['ZX']) / (info_AG89[element_denom]['ZX']/plot_kwargs['normalizations'][element_denom]['ZX'])
                abun_err_hi *= (info_AG89[element_num]['ZX']/plot_kwargs['normalizations'][element_num]['ZX']) / (info_AG89[element_denom]['ZX']/plot_kwargs['normalizations'][element_denom]['ZX'])

        if fig_kwargs['yaxis']['is_log']:
            # Change to derivative method?
            abun_err_lo = np.abs(np.log10(abun) - np.log10(abun - abun_err_lo))
            abun_err_hi = np.abs(np.log10(abun) - np.log10(abun + abun_err_hi))
            abun = np.log10(abun)
    #         abun_err_lo = np.log10(abun_err_lo)
    #         abun_err_hi = np.log10(abun_err_hi)



        ## Plot average profile with scatter        
        x_values = (rbins_R500[:-1] + rbins_R500[1:])/2.
        y_values = abun
        y_errs_lo = abun_err_lo
        y_errs_hi = abun_err_hi


        axes.fill_between(x_values, y_values-y_errs_lo, y_values+y_errs_hi,
                          color=color, 
                          alpha=plot_kwargs['alpha'], 
                          label='fill!'+labels[sample],
                          zorder=plot_kwargs['zorder'])
        axes.plot(x_values, y_values, 
                  color=color, 
                  marker=plot_kwargs['marker'], 
                  ls=ls,
                  lw=plot_kwargs['lw'], 
                  ms=plot_kwargs['ms'], 
                  mec=plot_kwargs['mec'], 
                  mew=plot_kwargs['mew'], 
                  label='line!'+labels[sample],
                  zorder=plot_kwargs['zorder'])


        if plot_kwargs['show_extra_label']:
            axes.plot([], [], 
                  color=color, 
                  marker=plot_kwargs['marker'], 
                  ls=ls,
                  lw=plot_kwargs['lw'], 
                  ms=plot_kwargs['ms'], 
                  mec=plot_kwargs['mec'], 
                  mew=plot_kwargs['mew'], 
                  label='line!'+plot_kwargs['extra_label'],
                  zorder=plot_kwargs['zorder'])
            
    return








def plot_ind_metallicity_sarkar22(type_='metallicity', axes=plt, bin_prop=None, bin_=None,
                                  bin_low=None, bin_high=None,
                              element=None, element_num=None, element_denom=None, 
                              plot_kwargs=None, fig_kwargs=None):

    # Sarkar+22 observations
    # Normalized to solar abundances of Asplund+09
    
    if not plot_kwargs['plot']:
        if plot_kwargs['verbose']:
            print('Not plotting Sarkar+22')
        return
    
    
    if type_.lower() == 'metallicity':
        name = element
    elif type_.lower() == 'metallicity ratio':
        name = '%s/%s' % (element_num, element_denom)
    else:
        raise Exception("type_ must be 'metallicity' or 'metallicity ratio'")
    
    try:
        test = obs_sarkar22['profiles'][type_.lower()][name].keys()
    except:
        if plot_kwargs['verbose']:
            print("%s not available from Sarkar+22" % name)
        return
    
    
    data_bin_prop = None
    use_data_bin_prop = plot_kwargs['use_bin_prop']
    if fig_kwargs['binning']['prop'][0] == 'M' and fig_kwargs['binning']['prop'][1] == '500':
        data_bin_prop = 'M500'
    elif bin_prop[0] == 'T':
        data_bin_prop = 'T'
    else:
        use_data_bin_prop = False
    
    
    for gal, obs in obs_sarkar22['profiles'][type_.lower()][name].items():
        
        if use_data_bin_prop:
            if not include_in_bin(obs_sarkar22[data_bin_prop][plot_kwargs['M500_dataset']][gal], bin_low, bin_high,
                                  extra=plot_kwargs['limit_extra'], use_frac=plot_kwargs['use_limit_frac'],
                                  frac=plot_kwargs['limit_frac'], transform_func=lambda x:np.log10(x)):
                if plot_kwargs['verbose']:
                    print(gal, 'from Sarkar+22 outside current %s bin' % (data_bin_prop))
                continue
        
        if plot_kwargs['print_labels']:
            if plot_kwargs['include_citation']:
                label_ = '#data_sarkar22##%s (%s)' % (gal, 'Sarkar+22')
            else:
                label_ = '#data_sarkar22##%s' % gal
        else:
            label_ = None
            
        # Radii in units of R/R200
        rad = obs[:,0].copy()
        rad_err_lo = obs[:,1].copy()
        rad_err_hi = obs[:,2].copy()

        Rmin = rad - rad_err_lo
        Rmax = rad + rad_err_hi
        Rcentre = (Rmin+Rmax)/2.

        abun_ = obs[:,3].copy()
        abun_err_lo_ = obs[:,4].copy()
        abun_err_hi_ = obs[:,5].copy()
        
        
        if plot_kwargs['renormalize_abundances']:
            if plot_kwargs['normalizations'] is None:
                raise Exception('Need to provide abundances to normalize by!')
                
            if type_.lower() == 'metallicity':
                abun_ *= info_Asplund09[element]['ZX']/plot_kwargs['normalizations'][element]['ZX']
                abun_err_lo_ *= info_Asplund09[element]['ZX']/plot_kwargs['normalizations'][element]['ZX']
                abun_err_hi_ *= info_Asplund09[element]['ZX']/plot_kwargs['normalizations'][element]['ZX']
                
            elif type_.lower() == 'metallicity ratio':
                abun_ *= (info_Asplund09[element_num]['ZX']/plot_kwargs['normalizations'][element_num]['ZX'])/(info_Asplund09[element_denom]['ZX']/plot_kwargs['normalizations'][element_denom]['ZX'])
                abun_err_lo_ *= (info_Asplund09[element_num]['ZX']/plot_kwargs['normalizations'][element_num]['ZX'])/(info_Asplund09[element_denom]['ZX']/plot_kwargs['normalizations'][element_denom]['ZX'])
                abun_err_hi_ *= (info_Asplund09[element_num]['ZX']/plot_kwargs['normalizations'][element_num]['ZX'])/(info_Asplund09[element_denom]['ZX']/plot_kwargs['normalizations'][element_denom]['ZX'])


        if fig_kwargs['xaxis']['is_log']:
            Rmin = np.log10(Rmin)
            Rmax = np.log10(Rmax)
            Rcentre = np.log10(Rcentre)
            if 'r500' in fig_kwargs['xaxis']['units'].lower():
#                 Rmin += np.log10(Rxxx_conversions['R200/R500'])
                Rmin += np.log10(Rxxx_conversions[plot_kwargs['sim_for_R_conversion']][plot_kwargs['type_for_R_conversion']]['bin_'+str(bin_)]['R200/R500'])
#                 Rmax += np.log10(Rxxx_conversions['R200/R500'])
                Rmax += np.log10(Rxxx_conversions[plot_kwargs['sim_for_R_conversion']][plot_kwargs['type_for_R_conversion']]['bin_'+str(bin_)]['R200/R500'])
#                 Rcentre += np.log10(Rxxx_conversions['R200/R500'])
                Rcentre += np.log10(Rxxx_conversions[plot_kwargs['sim_for_R_conversion']][plot_kwargs['type_for_R_conversion']]['bin_'+str(bin_)]['R200/R500'])
        else:
            if 'r500' in fig_kwargs['xaxis']['units'].lower():
#                 Rmin *= Rxxx_conversions['R200/R500']
                Rmin *= Rxxx_conversions[plot_kwargs['sim_for_R_conversion']][plot_kwargs['type_for_R_conversion']]['bin_'+str(bin_)]['R200/R500']
#                 Rmax *= Rxxx_conversions['R200/R500']
                Rmax *= Rxxx_conversions[plot_kwargs['sim_for_R_conversion']][plot_kwargs['type_for_R_conversion']]['bin_'+str(bin_)]['R200/R500']
#                 Rcentre *= Rxxx_conversions['R200/R500']
                Rcentre *= Rxxx_conversions[plot_kwargs['sim_for_R_conversion']][plot_kwargs['type_for_R_conversion']]['bin_'+str(bin_)]['R200/R500']

        if fig_kwargs['yaxis']['is_log']:
            # Change to derivative method?
            abun = np.log10(abun_)
            abun_err_lo = np.abs(abun - np.log10(abun_ - abun_err_lo_))
            abun_err_hi = np.abs(abun - np.log10(abun_ + abun_err_hi_))
        else:
            abun = abun_
            abun_err_lo = abun_err_lo_
            abun_err_hi = abun_err_hi_

        
        xerr = None
        yerr = None
        if plot_kwargs['plot_xerror']:
            xerr = [np.abs(Rcentre - Rmin), np.abs(Rcentre - Rmax)]
        if plot_kwargs['plot_yerror']:
            yerr = [abun_err_lo, abun_err_hi]
        axes.errorbar(Rcentre, abun, yerr=yerr, xerr=xerr, label=label_,
                      marker=plot_kwargs['marker'], 
                      ms=plot_kwargs['ms'], 
                      mec=plot_kwargs['mec'], 
                      mew=plot_kwargs['mew'], 
                      lw=plot_kwargs['lw'], 
                      color=plot_kwargs['colors'][gal],
                      ecolor=plot_kwargs['ecolor'], 
                      zorder=plot_kwargs['zorder'])
        
        
    if plot_kwargs['show_extra_label']:
        axes.errorbar([], [], yerr=None, xerr=None, 
                      label=plot_kwargs['extra_label'],
                      marker=plot_kwargs['marker'], 
                      ms=plot_kwargs['ms'], 
                      mec=plot_kwargs['mec'], 
                      mew=plot_kwargs['mew'], 
                      lw=plot_kwargs['lw'], 
                      color=plot_kwargs['color'],
                      zorder=plot_kwargs['zorder'])
    return





def plot_avg_metallicity_fukushima23(type_='metallicity', axes=plt, bin_=None,
                                       bin_low=None, bin_high=None, sim_rbins=None,
                                       element=None, element_num=None, element_denom=None,
                                       plot_kwargs=None, fig_kwargs=None):
    
    ## Fukushima+2023 observations from CHEERS sample
    ## Just core metallicity of each group
    ## Already calculated size of core for each group in units of R/R500, so can just treat as a profile with one bin
    ## Abundances in units of Lodders+2009
    ## region can be either 'core' or 'profile'
    
    if not plot_kwargs['plot']:
        if plot_kwargs['verbose']:
            print('Not plotting Fukushima+2023 average', plot_kwargs['region'], 'abundances')
        return
    
    if type_.lower() == 'metallicity':
        element_name = element
    elif type_.lower() == 'metallicity ratio':
        element_name = element_num + '/' + element_denom
    else:
        raise Exception("type_ needs to be 'metallicity' or 'metallicity ratio'")
        
    if plot_kwargs['print_labels']:
        if plot_kwargs['include_citation']:
            labels = {
                'avg':'#data_fukushima23##Stacked Groups (Fukushima+23)'
            }
        else:
            labels = {
                'avg':'#data_fukushima23##Stacked Groups'
            }
    else:
        labels = {
            'avg':None
        }
        
        
    data_bin_prop = None
    use_data_bin_prop = plot_kwargs['use_bin_prop']
    if fig_kwargs['binning']['prop'][0] == 'M' and fig_kwargs['binning']['prop'][1] == '500':
        data_bin_prop = 'M500'
    elif bin_prop[0] == 'T':
        data_bin_prop = 'kT'
    else:
        use_data_bin_prop = False
        
        
    ## Get exact observations to average
    averaging_obs = {}
    for group, info in obs_fukushima23['individual'].items():
        try:
            obs = info['metallicity'][plot_kwargs['region']][element_name]
        except:
            continue

        if use_data_bin_prop:
            ## Remove groups that do not have bin_prop (eg. M500, kT) values
            if data_bin_prop not in info.keys():
                continue
            ## Remove groups outside of bin_prop bin
            
            if not include_in_bin(info[data_bin_prop], bin_low, bin_high, 
                                  extra=plot_kwargs['limit_extra'], use_frac=plot_kwargs['use_limit_frac'],
                                  frac=plot_kwargs['limit_frac'], transform_func=lambda x:np.log10(x)):
                continue

                
        rbins_R500 = obs['rbins_R500'].copy()
        abun = obs['value'].copy()
        abun_err_lo = np.abs(obs['err_lo'].copy())
        abun_err_hi = np.abs(obs['err_hi'].copy())

        if plot_kwargs['renormalize_abundances']:
            if plot_kwargs['normalizations'] is None:
                raise Exception('Need to provide abundances to normalize by!')

            if type_.lower() == 'metallicity':
                abun *= info_Lodders09[element]['ZX']/plot_kwargs['normalizations'][element]['ZX']
                abun_err_lo *= info_Lodders09[element]['ZX']/plot_kwargs['normalizations'][element]['ZX']
                abun_err_hi *= info_Lodders09[element]['ZX']/plot_kwargs['normalizations'][element]['ZX']
            elif type_.lower() == 'metallicity ratio':
                abun *= (info_Lodders09[element_num]['ZX']/plot_kwargs['normalizations'][element_num]['ZX']) / (info_Lodders09[element_denom]['ZX']/plot_kwargs['normalizations'][element_denom]['ZX'])
                abun_err_lo *= (info_Lodders09[element_num]['ZX']/plot_kwargs['normalizations'][element_num]['ZX']) / (info_Lodders09[element_denom]['ZX']/plot_kwargs['normalizations'][element_denom]['ZX'])
                abun_err_hi *= (info_Lodders09[element_num]['ZX']/plot_kwargs['normalizations'][element_num]['ZX']) / (info_Lodders09[element_denom]['ZX']/plot_kwargs['normalizations'][element_denom]['ZX'])

        if 'r200' in fig_kwargs['xaxis']['units'].lower():
#             rbins_R500 *= Rxxx_conversions['R500/R200']
            rbins_R500 *= Rxxx_conversions[plot_kwargs['sim_for_R_conversion']][plot_kwargs['type_for_R_conversion']]['bin_'+str(bin_)]['R500/R200']
        elif 'r2500' in fig_kwargs['xaxis']['units'].lower():
#             rbins_R500 *= Rxxx_conversions['R500/R2500']
            rbins_R500 *= Rxxx_conversions[plot_kwargs['sim_for_R_conversion']][plot_kwargs['type_for_R_conversion']]['bin_'+str(bin_)]['R500/R2500']
            
            
        if plot_kwargs['exclude_negatives']:
            abun[abun<0] = np.nan


        if fig_kwargs['yaxis']['is_log']:
            # Change to derivative method?
            abun_err_lo = np.abs(np.log10(abun) - np.log10(abun - abun_err_lo))
            abun_err_hi = np.abs(np.log10(abun) - np.log10(abun + abun_err_hi))
            abun = np.log10(abun)
#             abun_err_lo = np.log10(abun_err_lo)
#             abun_err_hi = np.log10(abun_err_hi)

        mean_abun_err = np.mean(np.array([abun_err_lo, abun_err_hi]), axis=0)

        
        averaging_obs[group] = {
            'xbins':rbins_R500,
            'values':abun,
            'errs':mean_abun_err,
#             'values_err_lo':abun_err_lo,
#             'values_err_hi':abun_err_hi,
        }
        
        
    if len(averaging_obs.keys()) == 0:
        if plot_kwargs['verbose']:
            print(element_name, 'not available from Fukushima+2023')
        return
        
        
    
    ## Get reference radial bins for averaging
    if plot_kwargs['rbins_type'] == 'provide':
        if plot_kwargs['rbins'] is None:
            raise Exception('averaging_bins must be provided if rbins_type is provide')
        averaging_bins = plot_kwargs['rbins']
        
    elif plot_kwargs['rbins_type'] == 'calculate':
        counter = 0
    
        for group, info in averaging_obs.items():
            
            rbins = info['xbins'].copy()
            Rcentre = (rbins[:-1] + rbins[1:])/2.

            if counter == 0:
                averaging_bins_start = rbins[0]
                averaging_bins_end = rbins[-1]
                data_radial_centres = Rcentre.copy()
            else:
                averaging_bins_start = min(averaging_bins_start, rbins[0])
                averaging_bins_end = max(averaging_bins_end, rbins[-1])
                data_radial_centres = np.append(data_radial_centres, Rcentre)

            counter += 1
            
        if plot_kwargs['rbins_method'] == 'linear':
            averaging_bins = np.linspace(averaging_bins_start, averaging_bins_end, plot_kwargs['n_rbins'])
        elif plot_kwargs['rbins_method'] == 'log':
            averaging_bins = np.logspace(averaging_bins_start, averaging_bins_end, plot_kwargs['n_rbins'])
        elif plot_kwargs['rbins_method'] == 'equal':
            __, averaging_bins = pd.qcut(data_radial_centres, q=plot_kwargs['n_rbins'], 
                                         labels=None, retbins=True)

    elif plot_kwargs['rbins_type'] == 'sim':
        averaging_bins = sim_rbins

    else:
        raise Exception("rbins_type must be 'provide', 'calculate', or 'sim'")
        
        
    ## Average profiles
    if plot_kwargs['avg_type'].lower() == 'mean':
        ## Average profiles according to Mernier+2017 equation 3
        result = weighted_mean_of_profiles(averaging_bins, averaging_obs)
        averaged_profile = result[:,0]
        std_profile = result[:,1]
        if fig_kwargs['yaxis']['is_log']:
            averaged_limits = [np.log10(10**averaged_profile - 10**std_profile), 
                               np.log10(10**averaged_profile + 10**std_profile)]
        else:
            averaged_limits = [averaged_profile - std_profile, 
                               averaged_profile + std_profile]
    elif plot_kwargs['avg_type'].lower() == 'median':
        result = weighted_quantiles_of_profiles(averaging_bins, averaging_obs, calc_kwargs=plot_kwargs)
        closest_idx_to_middle = np.argmin(np.abs(np.array(plot_kwargs['quantiles']) - 0.5))
        averaged_profile = result[:,closest_idx_to_middle]
        averaged_limits = [result[:,0], result[:,-1]]
    else:
        raise Exception('avg_type must be mean or median')
    
#     if plot_kwargs['avg_type'].lower() == 'mean':
#         ## Average profiles according to Mernier+2017 equation 3
#         result = weighted_mean_of_profiles(averaging_bins, averaging_obs)#, calc_kwargs=plot_kwargs)
#         averaged_profile = result[0]
#         averaged_scatter = [result[1], result[1]]
#     elif plot_kwargs['avg_type'].lower() == 'median':
#         result = weighted_quantiles_of_profiles(averaging_bins, averaging_obs)#, calc_kwargs=plot_kwargs)
#         closest_idx_to_median = np.argmin(np.abs(plot_kwargs['quantiles'] - 0.5))
#         averaged_profile = result[:,closest_idx_to_median]
#         averaged_scatter = [result[0], result[-1]]
#     else:
#         raise Exception('avg_type must be mean or median')


    ## Plot averaged profile with scatter
    if plot_kwargs['region'].lower() == 'core':
        if fig_kwargs['xaxis']['is_log']:
            averaging_bins = np.log10(averaging_bins)
            
        x_values = np.array([averaging_bins[0]] + list(np.repeat(averaging_bins[1:-1], repeats=2)) + [averaging_bins[-1]])
        y_values = np.array([averaged_profile[0]] + list(np.repeat(averaged_profile[1:-1], repeats=2)) + [averaged_profile[-1]])
#         y_errs = np.array([averaged_scatter[0]] + list(np.repeat(averaged_scatter[1:-1], repeats=2)) + [averaged_scatter[-1]])
        y_errs = [
            np.array([averaged_limits[0][0]] + list(np.repeat(averaged_limits[0][1:-1], repeats=2)) + [averaged_limits[0][-1]]),
            np.array([averaged_limits[1][0]] + list(np.repeat(averaged_limits[1][1:-1], repeats=2)) + [averaged_limits[1][-1]]),
                 ]

    elif plot_kwargs['region'].lower() == 'profile':
        averaging_bin_centres = (averaging_bins[:-1] + averaging_bins[1:])/2.
        if fig_kwargs['xaxis']['is_log']:
            averaging_bin_centres = np.log10(averaging_bin_centres)
            
        x_values = averaging_bin_centres
        y_values = averaged_profile
        y_errs = averaged_scatter


    axes.fill_between(x_values, y_errs[0], y_errs[1],
                      color=plot_kwargs['color'], 
                      alpha=plot_kwargs['alpha'], 
                      label='fill!'+labels['avg'], 
                      zorder=plot_kwargs['zorder'])
    axes.plot(x_values, y_values, 
              color=plot_kwargs['color'], 
              marker=plot_kwargs['marker'], 
              ls=plot_kwargs['ls'],
              lw=plot_kwargs['lw'], 
              ms=plot_kwargs['ms'], 
              mec=plot_kwargs['mec'], 
              mew=plot_kwargs['mew'], 
              label='line!'+labels['avg'],
              zorder=plot_kwargs['zorder'])


    if plot_kwargs['show_extra_label']:
        axes.plot([], [], 
              color=plot_kwargs['color'], 
              marker=plot_kwargs['marker'], 
              ls=plot_kwargs['ls'],
              lw=plot_kwargs['lw'], 
              ms=plot_kwargs['ms'], 
              mec=plot_kwargs['mec'], 
              mew=plot_kwargs['mew'], 
              label='line!'+plot_kwargs['extra_label'],
              zorder=plot_kwargs['zorder'])
    return















    
    



def plot_metallicity_Gastaldello21(element, axes=plt, logx=False, logy=False, xunits='R500', #zorder=100,
#                                    data_ms=10, data_lw=1, data_colors=['black']*4, 
                                   obs_plot_kwargs=None, 
                                   renormalize_abundances=False, normalizations=None, 
                                   print_labels=False, extra_label=None, verbose=False):

    # Collection of Fe abundances from various studies, all from Gastaldello+21
    # Radii in units of R/R500
    # Abundances in proto-solar units of Asplund et al (2009)
    
    if element.lower() != 'fe':
        if verbose:
            print("Gastaldello+21 only has abundance profiles for Fe")
        return
    
#     plot_type = {
#             'lovisari19':'13 groups (Lovisari+ 2019)',
#             'mernier17':'21 groups (CHEERS - Mernier+ 2017)',
#             'sasaki14':'4 groups (Sasaki+ 2014)',
#             'sun12':'27 groups (Sun 2012)',
#             'su15':'RXJ1159+5531 (Su+ 2015)',
#             'thoelken16':'UGC 03957 (Thoelken+ 2016)',
#         }

    plot_colors_ = {
            'lovisari19':'red',
            'mernier17':'green',
            'sasaki14':'blue',
            'sun12':'orange',
            'su15':'purple',
            'thoelken16':'black',
        }
    
    if obs_plot_kwargs['Gastaldello+21']['print_labels']:
        labels_ = {
            'lovisari19':'#data_gastaldello21##13 groups (Lovisari+ 2019)',
            'mernier17':'#data_gastaldello21##21 groups (CHEERS - Mernier+ 2017)',
            'sasaki14':'#data_gastaldello21##4 groups (Sasaki+ 2014)',
            'sun12':'#data_gastaldello21##27 groups (Sun 2012)',
            'su15':'#data_gastaldello21##RXJ1159+5531 (Su+ 2015)',
            'thoelken16':'#data_gastaldello21##UGC 03957 (Thoelken+ 2016)',
        }
    else:
        labels_ = {
            'lovisari19':None,
            'mernier17':None,
            'sasaki14':None,
            'sun12':None,
            'su15':None,
            'thoelken16':None,
        }
        
        
    for author_, values_ in obs_gastaldello21.items():
        if author_ in ['lovisari19', 'mernier17']:
            x_ = values_[:,0]
            y_ = values_[:,1]
            y_lo_ = values_[:,2]
            y_hi_ = values_[:,3]
            y_err_lo_ = np.abs(y_-y_lo_)
            y_err_hi_ = np.abs(y_-y_hi_)
        elif author_ in ['sasaki14', 'su15', 'sun12', 'thoelken16']:
            x_ = values_[:,0]
            x_err_ = values_[:,1]
            y_ = values_[:,2]
            y_err_hi_ = np.abs(values_[:,3])
            y_err_lo_ = np.abs(values_[:,4])
            
        if renormalize_abundances:
            if normalizations is None:
                raise Exception('Need to provide abundances to normalize by!')

            y_ *= info_Asplund09[element]['ZX']/normalizations[element]['ZX']
            y_err_lo_ *= info_Asplund09[element]['ZX']/normalizations[element]['ZX']
            y_err_hi_ *= info_Asplund09[element]['ZX']/normalizations[element]['ZX']
            
        if logx:
            x = np.log10(x_)
            if 'r200' in xunits.lower():
#                 x += np.log10(Rxxx_conversions['R500/R200'])
                x += np.log10(Rxxx_conversions[plot_kwargs['sim_for_R_conversion']][plot_kwargs['type_for_R_conversion']]['bin_'+str(bin_)]['R500/R200'])
        else:
            x = x_
            if 'r200' in xunits.lower():
#                 x *= Rxxx_conversions['R500/R200']
                x *= Rxxx_conversions[plot_kwargs['sim_for_R_conversion']][plot_kwargs['type_for_R_conversion']]['bin_'+str(bin_)]['R500/R200']


        if logy:
            y = np.log10(y_)
            y_err_lo = np.abs(y - np.log10(y_ - y_err_lo))
            y_err_hi = np.abs(y - np.log10(y_ + y_err_hi))
        else:
            y = y_
            y_err_lo = y_err_lo_
            y_err_hi = y_err_hi_
            
            
        axes.fill_between(x, y-y_err_lo, y+y_err_hi,
                         color=plot_colors_[author_], alpha=0.25, label=labels_[author_], zorder=zorder)
#         axes.plot(x, y, color=plot_colors_[author_], marker='s', ms=10, mec='black', mew=2)
        
        
    axes.plot([], [], label=obs_plot_kwargs['Gastaldello+21']['extra_label'])

    

def plot_sims_oppenheimer21(prop, axes=plt, logx=False, logy=False, xunits='R500', yunits=None,
                            zorder=100, multiplier=None, print_labels=False, extra_label=None, verbose=False):
    
    ## Mass-weighted simulation profiles from Oppenheimer+21
    
    ## x values: log10(R/R500)
    ## For entropy: log10(K/K500)
    ## For pressure: log10(P/P500)
    ## For density: log(ne[cm^-3])
    ## For temperature: log(T[keV])
    
    prop = prop.lower()
    
    try:
        test_ = obs_oppenheimer21[prop]
    except:
        if verbose:
            print("Oppenheimer+21 does not have", prop)
        return
     
    plot_colors_ = {
        'SIMBA':'red',
        'EAGLE':'indigo',
        'TNG100':'cyan',
    }
    
    plot_units_ = {
        'entropy':'',
        'pressure':'',
        'density':'cm**-3',
        'temperature':'keV',
    }
    
    if yunits is None:
        yunits = plot_units_[prop]
        
    if verbose:
        print('yunits:', yunits)
        
    if multiplier is None:
        multiplier = pnb.array.SimArray(1, units='')
    
    for sim_, vals_ in obs_oppenheimer21[prop].items():
#         print(sim_)
        log_R_R500 = copy.deepcopy(vals_[0][:,0])
#         print(log_R_R500)

        log_median = copy.deepcopy(vals_[0][:,1])
        median = pnb.array.SimArray(10**log_median, units=plot_units_[prop])
        median *= multiplier
        log_median = np.log10(median.in_units(yunits))
        
        log_percentile_16 = copy.deepcopy(vals_[0][:,2])
        percentile_16 = pnb.array.SimArray(10**log_percentile_16, units=plot_units_[prop])
        percentile_16 *= multiplier
        log_percentile_16 = np.log10(percentile_16.in_units(yunits))
        
        log_percentile_84 = copy.deepcopy(vals_[0][:,3])
        percentile_84 = pnb.array.SimArray(10**log_percentile_84, units=plot_units_[prop])
        percentile_84 *= multiplier
        log_percentile_84 = np.log10(percentile_84.in_units(yunits))
        
        nhalos = copy.deepcopy(vals_[0][:,4])
        
#         log_median += np.log10(multiplier)
#         percentile_16 += np.log10(multiplier)
#         percentile_84 += np.log10(multiplier)
        
        if 'r200' in xunits.lower():
#             log_R_R500 += np.log10(Rxxx_conversions['R500/R200'])
            log_R_R500 += np.log10(Rxxx_conversions[plot_kwargs['sim_for_R_conversion']][plot_kwargs['type_for_R_conversion']]['bin_'+str(bin_)]['R500/R200'])
        elif 'r500' in xunits.lower():
            log_R_R500 += 0
        else:
            if verbose:
                print('Invalid x units')
            return
        
        if not logx:
            x_ = 10**log_R_R500
        else:
            x_ = log_R_R500
        
        if not logy:
            y_ = 10**log_median
            y_lo_ = 10**log_percentile_16
            y_hi_ = 10**log_percentile_84
        else:
            y_ = log_median
            y_lo_ = log_percentile_16
            y_hi_ = log_percentile_84
            
        
        if print_labels:
            label_ = sim_
        else:
            label_ = None
            
        
        axes.plot(x_, y_, color=plot_colors_[sim_], label='line!'+label_, zorder=zorder)
        axes.fill_between(x_, y_lo_, y_hi_, color=plot_colors_[sim_], alpha=0.1, label='fill!'+label_, zorder=zorder)
    
    axes.plot([], [], label=extra_label)



def plot_obs_osullivan17(prop, axes=plt, logx=False, logy=False, xunits='R500', yunits=None,
                         data_ms=10, data_lw=1, data_colors=['black']*4, obs_plot_kwargs=None, 
                         zorder=100, multiplier=None, print_labels=False, extra_label=None, verbose=False):
    
    ## Mass-weighted simulation profiles from Oppenheimer+21
    
    ## x values: log10(R/R500)
    ## For entropy: log10(K/K500)
    ## For pressure: log10(P/P500)
    ## For density: log(ne[cm^-3])
    ## For temperature: log(T[keV])
    
    prop = prop.lower()
    
    try:
        test_ = obs_oppenheimer21[prop]
    except:
        if verbose:
            print("Oppenheimer+21 does not have", prop)
        return
     
    plot_colors_ = {
        'SIMBA':'red',
        'EAGLE':'indigo',
        'TNG100':'cyan',
    }
    
    plot_units_ = {
        'entropy':'',
        'temperature':'keV',
    }
    
    if yunits is None:
        yunits = plot_units_[prop]
        
    if verbose:
        print('yunits:', yunits)
        
    if multiplier is None:
        multiplier = pnb.array.SimArray(1, units='')
    
    for sim_, vals_ in obs_osullivan17[prop].items():
#         print(sim_)
        log_R_R500 = copy.deepcopy(vals_[0][:,0])
#         print(log_R_R500)

        log_median = copy.deepcopy(vals_[0][:,1])
        median = pnb.array.SimArray(10**log_median, units=plot_units_[prop])
        median *= multiplier
        log_median = np.log10(median.in_units(yunits))
        
        log_percentile_16 = copy.deepcopy(vals_[0][:,2])
        percentile_16 = pnb.array.SimArray(10**log_percentile_16, units=plot_units_[prop])
        percentile_16 *= multiplier
        log_percentile_16 = np.log10(percentile_16.in_units(yunits))
        
        log_percentile_84 = copy.deepcopy(vals_[0][:,3])
        percentile_84 = pnb.array.SimArray(10**log_percentile_84, units=plot_units_[prop])
        percentile_84 *= multiplier
        log_percentile_84 = np.log10(percentile_84.in_units(yunits))
        
        nhalos = copy.deepcopy(vals_[0][:,4])
        
#         log_median += np.log10(multiplier)
#         percentile_16 += np.log10(multiplier)
#         percentile_84 += np.log10(multiplier)
        
        if 'r200' in xunits.lower():
#             log_R_R500 += np.log10(Rxxx_conversions['R500/R200'])
            log_R_R500 += np.log10(Rxxx_conversions[plot_kwargs['sim_for_R_conversion']][plot_kwargs['type_for_R_conversion']]['bin_'+str(bin_)]['R500/R200'])
        elif 'r500' in xunits.lower():
            log_R_R500 += 0
        else:
            if verbose:
                print('Invalid x units')
            return
        
        if not logx:
            x_ = 10**log_R_R500
        else:
            x_ = log_R_R500
        
        if not logy:
            y_ = 10**log_median
            y_lo_ = 10**log_percentile_16
            y_hi_ = 10**log_percentile_84
        else:
            y_ = log_median
            y_lo_ = log_percentile_16
            y_hi_ = log_percentile_84
            
        
        if print_labels:
            label_ = sim_
        else:
            label_ = None
            
        
        axes.plot(x_, y_, color=plot_colors_[sim_], label=label_, zorder=zorder)
        axes.fill_between(x_, y_lo_, y_hi_, color=plot_colors_[sim_], alpha=0.1, zorder=zorder)
    
    axes.plot([], [], label=extra_label)

    
    

def plot_obs_xgap(prop, axes=plt, logx=False, logy=False, xunits='R500', yunits=None,
                  data_ms=10, data_lw=1, data_colors=['black']*4, obs_plot_kwargs=None, 
                  zorder=100, multiplier=None, color='black', print_labels=False, extra_label=None, verbose=False):
    
    ## Observational profiles from X-GAP
    
    ## x values: R/R500
    ## For entropy: K [] or K/K500
    ## For pressure: P [] or P/P500
    ## For density: ne[cm^-3]) or ne/ne500
    ## For temperature: T[keV] or T/T500
    
    prop = prop.lower()
    
    try:
        test_ = obs_oppenheimer21[prop]
    except:
        if verbose:
            print("Oppenheimer+21 does not have", prop)
        return
     
    plot_colors_ = {
        'SIMBA':'red',
        'EAGLE':'indigo',
        'TNG100':'cyan',
    }
    
    plot_units_ = {
        'entropy':'keV cm**-2',
        'temperature':'keV',
    }
    
    if yunits is None:
        yunits = plot_units_[prop]
    
    if verbose:
        print('yunits:', yunits)
        
    if multiplier is None:
        multiplier = pnb.array.SimArray(1, units='')
    
    for sim_, vals_ in obs_osullivan17[prop].items():
#         print(sim_)
        log_R_R500 = copy.deepcopy(vals_[0][:,0])
#         print(log_R_R500)

        log_median = copy.deepcopy(vals_[0][:,1])
        median = pnb.array.SimArray(10**log_median, units=plot_units_[prop])
        median *= multiplier
        log_median = np.log10(median.in_units(yunits))
        
        log_percentile_16 = copy.deepcopy(vals_[0][:,2])
        percentile_16 = pnb.array.SimArray(10**log_percentile_16, units=plot_units_[prop])
        percentile_16 *= multiplier
        log_percentile_16 = np.log10(percentile_16.in_units(yunits))
        
        log_percentile_84 = copy.deepcopy(vals_[0][:,3])
        percentile_84 = pnb.array.SimArray(10**log_percentile_84, units=plot_units_[prop])
        percentile_84 *= multiplier
        log_percentile_84 = np.log10(percentile_84.in_units(yunits))
        
        nhalos = copy.deepcopy(vals_[0][:,4])
        
#         log_median += np.log10(multiplier)
#         percentile_16 += np.log10(multiplier)
#         percentile_84 += np.log10(multiplier)
        
        if 'r200' in xunits.lower():
#             log_R_R500 += np.log10(Rxxx_conversions['R500/R200'])
            log_R_R500 += np.log10(Rxxx_conversions[plot_kwargs['sim_for_R_conversion']][plot_kwargs['type_for_R_conversion']]['bin_'+str(bin_)]['R500/R200'])
        elif 'r500' in xunits.lower():
            log_R_R500 += 0
        else:
            if verbose:
                print('Invalid x units')
            return
        
        if not logx:
            x_ = 10**log_R_R500
        else:
            x_ = log_R_R500
        
        if not logy:
            y_ = 10**log_median
            y_lo_ = 10**log_percentile_16
            y_hi_ = 10**log_percentile_84
        else:
            y_ = log_median
            y_lo_ = log_percentile_16
            y_hi_ = log_percentile_84
            
        
        if print_labels:
            label_ = sim_
        else:
            label_ = None
            
        
        axes.plot(x_, y_, color=plot_colors_[sim_], label=label_)
        axes.fill_between(x_, y_lo_, y_hi_, color=plot_colors_[sim_], alpha=0.1)
    
    axes.plot([], [], label=extra_label)
    
    
    
    








## Full observation plotting function

def plot_observations(prop, info, bin_, type_='regular', axes=plt, 
                      element=None, element_num=None, element_denom=None, 
                      bin_low=None, bin_high=None, sm=None,
                      obs_plot_kwargs=None, plot_kwargs=None, sim_plot_kwargs=None):
    
    sim_rbins=info['XAXIS_BINS']
    
    bin_prop = plot_kwargs['binning']['prop']
    log_bin_low = np.log10(bin_low)
    log_bin_high = np.log10(bin_high)
    
    
    if type_.lower() == 'regular':
        
        if prop.startswith('K') and prop.endswith(('K200', 'K200ideal', 'S200', 'S200ideal')) and plot_kwargs['xaxis']['units'].lower()=='r500':
            if plot_kwargs['xaxis']['is_log']:
                x = 10**info['XAXIS']
            else:
                x = info['XAXIS']

            # Plot baseline approximately self-similar entropy profile seen in non-radiative simulations of groups/clusters
            # K(r)/K200 = 1.32 (r/r200)^1.1  (voit et al 2005) --> should it be r^-1.1???
            # K(r)/K500 = 1.32 (r/r200)^1.1 (K200/K500)
            # K(r)/K500 = 1.32 (r/r500)^1.1
            K_K200_baseline = 0.9 * 1.32 * x**(1.1)
            #K_K500_baseline = K_K200_baseline

            if plot_kwargs['yaxis']['is_log']:
                y = np.log10(K_K200_baseline)
            else:
                y = K_K200_baseline

            axes.plot(info['XAXIS'], y, ls='--', color='black', lw=3, 
                      label=r'$K \propto r^{1.1}$ (Lewis+2000, Babul+2002, Voit+2005)', 
                      zorder=obs_plot_kwargs['Oppenheimer+21']['entropy_theory_zorder'])


            # Plot low mass (and low radius) group entropy profile from observations in Panagoulia et al, 2014
            # Using approximate match to fig. 6 low mass/low radius (purple lines) in Oppenheimer et al, 2021
            # Obtaining normalization such that at log(R/R500)=-2, log(K/K500)~-0.4
            K_K200_lowmass = 0.8 * 15. * x**(0.7) * 1./50.

            if plot_kwargs['yaxis']['is_log']:
                y = np.log10(K_K200_lowmass)
            else:
                y = K_K200_lowmass

            axes.plot(info['XAXIS'], y, ls='-.', color='grey', lw=3, 
                      label=r'$K \propto r^{0.7}$ (Panagoulia+2014)', 
                      zorder=obs_plot_kwargs['Oppenheimer+21']['entropy_theory_zorder'])
    
    

        if prop.startswith('K') and prop.endswith(('K500', 'K500ideal', 'S500', 'S500ideal')) and plot_kwargs['xaxis']['units'].lower()=='r500':
            if plot_kwargs['xaxis']['is_log']:
                x = 10**info['XAXIS']
            else:
                x = info['XAXIS']

    #         # Plot baseline approximately self-similar entropy profile seen in non-radiative simulations of groups/clusters
    #         # K(r)/K200 = 1.32 (r/r200)^1.1  (voit et al 2005, Fig. 1) --> should it be r^-1.1???
    #         # K(r)/K500 = 1.32 (r/r200)^1.1 (K200/K500)
    #         # K(r)/K500 = 1.32 (r/r500)^1.1 (r500/r200)^1.1 (K200/K500)
    #         # K200/K500 = (T200/T500) (ne200/ne500)^-2/3 = (M200/M500) (r500/r200) (200/500)^-2/3
    #         # M200/M500 = (200/500) (r200/r500)^3
    #         # K/K500 = 1.32 (r/r500)^1.1 (r200/r500)^(4-1.1) (200/500)^(1/3)
    #         K_K500_baseline = 1.32 * x**(1.1) * R200/R500**(4.-1.1) * (200./500.)**(1./3.)
            K_K500_baseline = 0.9 * 1.32/1.09 * (x)**1.1  # Oppenheimer+21 plotting script
    #         #K_K500_baseline = K_K200_baseline

            if plot_kwargs['yaxis']['is_log']:
                y = np.log10(K_K500_baseline)
            else:
                y = K_K500_baseline

            axes.plot(info['XAXIS'], y, ls='--', color='black', lw=3, 
                      label=r'$K \propto r^{1.1}$ (Lewis+2000, Babul+2002, Voit+2005)', 
                      zorder=obs_plot_kwargs['Oppenheimer+21']['entropy_theory_zorder'])


    #         # Plot low mass (and low radius) group entropy profile from observations in Panagoulia et al, 2014 (using 0.7 power law)
    #         # Using approximate match to fig. 6 low mass/low radius (purple lines) in Oppenheimer et al, 2021
    #         # Obtaining normalization such that at log(R/R500)=-2, log(K/K500)~-0.4
#             K_K500_lowmass = 0.8 * 15. * x**(0.7) * Rxxx_conversions['R200/R500']**(4.-0.7) * (200./500.)**(1./3.) * 1./50.
            K_K500_lowmass = 0.8 * 15. * x**(0.7) * Rxxx_conversions['simba-c']['value_ratios']['bin_'+str(bin_)]['R200/R500']**(4.-0.7) * (200./500.)**(1./3.) * 1./50.

            if plot_kwargs['yaxis']['is_log']:
                y = np.log10(K_K500_lowmass)
            else:
                y = K_K500_lowmass

            axes.plot(info['XAXIS'], y, ls='-.', color='grey', lw=3, 
                      label=r'$K \propto r^{0.7}$ (Panagoulia+2014)', 
                      zorder=obs_plot_kwargs['Oppenheimer+21']['entropy_theory_zorder'])




            if not obs_plot_kwargs['Oppenheimer+21']['plot_just_entropy_theory']:
                opp21_plot('entropy_norm', log_bin_low, log_bin_high, sm=sm, cm=plot_kwargs['cbar']['cm'], axes_=axes, 
                           plot_individual_observations=obs_plot_kwargs['Oppenheimer+21']['plot_individual_observations'], 
                           plot_only_Lovisari_Sun_groups=obs_plot_kwargs['Oppenheimer+21']['plot_only_Lovisari_Sun_groups'], 
                           plot_x_log=plot_kwargs['xaxis']['is_log'], plot_y_log=plot_kwargs['yaxis']['is_log'], 
                           use_cbar=plot_kwargs['cbar']['plot'], 
                           log_cbar=plot_kwargs['cbar']['log_cbar'],
                           obs_plot_kwargs=obs_plot_kwargs)



        if prop.startswith(('ne')) and not prop.endswith(('200', '500')):

            if plot_kwargs['xaxis']['units'].lower()=='r500':
                opp21_plot('density', log_bin_low, log_bin_high, sm=sm, cm=plot_kwargs['cbar']['cm'], axes_=axes, 
                           plot_individual_observations=obs_plot_kwargs['Oppenheimer+21']['plot_individual_observations'], 
                           plot_only_Lovisari_Sun_groups=obs_plot_kwargs['Oppenheimer+21']['plot_only_Lovisari_Sun_groups'], 
                           plot_x_log=plot_kwargs['xaxis']['is_log'], plot_y_log=plot_kwargs['yaxis']['is_log'], 
                           use_cbar=plot_kwargs['cbar']['plot'], log_cbar=plot_kwargs['cbar']['log_cbar'],
                           obs_plot_kwargs=obs_plot_kwargs)


        if prop.startswith('T') and not prop.endswith(('200', '500')):

            if plot_kwargs['xaxis']['units'].lower()=='r500':
                opp21_plot('temperature', log_bin_low, log_bin_high, sm=sm, cm=plot_kwargs['cbar']['cm'], axes_=axes, 
                           plot_individual_observations=obs_plot_kwargs['Oppenheimer+21']['plot_individual_observations'], 
                           plot_only_Lovisari_Sun_groups=obs_plot_kwargs['Oppenheimer+21']['plot_only_Lovisari_Sun_groups'], 
                           plot_x_log=plot_kwargs['xaxis']['is_log'], plot_y_log=plot_kwargs['yaxis']['is_log'], 
                           use_cbar=plot_kwargs['cbar']['plot'], log_cbar=plot_kwargs['cbar']['log_cbar'],
                           obs_plot_kwargs=obs_plot_kwargs)

        if prop.startswith('kT') and not prop.endswith(('200', '500')):

            if plot_kwargs['xaxis']['units'].lower()=='r500':
                opp21_plot('ktemperature', log_bin_low, log_bin_high, sm=sm, cm=plot_kwargs['cbar']['cm'], axes_=axes, 
                           plot_individual_observations=obs_plot_kwargs['Oppenheimer+21']['plot_individual_observations'], 
                           plot_only_Lovisari_Sun_groups=obs_plot_kwargs['Oppenheimer+21']['plot_only_Lovisari_Sun_groups'], 
                           plot_x_log=plot_kwargs['xaxis']['is_log'], plot_y_log=plot_kwargs['yaxis']['is_log'], 
                           use_cbar=plot_kwargs['cbar']['plot'], log_cbar=plot_kwargs['cbar']['log_cbar'],
                           obs_plot_kwargs=obs_plot_kwargs)

        if prop.startswith(('T', 'kT')) and prop.endswith(('500', '500ideal')):

            if plot_kwargs['xaxis']['units'].lower()=='r500':
                opp21_plot('temperature_norm', log_bin_low, log_bin_high, sm=sm, cm=plot_kwargs['cbar']['cm'], axes_=axes, 
                           plot_individual_observations=obs_plot_kwargs['Oppenheimer+21']['plot_individual_observations'], 
                           plot_only_Lovisari_Sun_groups=obs_plot_kwargs['Oppenheimer+21']['plot_only_Lovisari_Sun_groups'], 
                           plot_x_log=plot_kwargs['xaxis']['is_log'], plot_y_log=plot_kwargs['yaxis']['is_log'], 
                           use_cbar=plot_kwargs['cbar']['plot'], log_cbar=plot_kwargs['cbar']['log_cbar'],
                           obs_plot_kwargs=obs_plot_kwargs)

        if prop.startswith('P') and prop.endswith(('500', '500ideal')):

            if plot_kwargs['xaxis']['units'].lower()=='r500':
                opp21_plot('pressure_norm', log_bin_low, log_bin_high, sm=sm, cm=plot_kwargs['cbar']['cm'], axes_=axes, 
                           plot_individual_observations=obs_plot_kwargs['Oppenheimer+21']['plot_individual_observations'], 
                           plot_only_Lovisari_Sun_groups=obs_plot_kwargs['Oppenheimer+21']['plot_only_Lovisari_Sun_groups'], 
                           plot_x_log=plot_kwargs['xaxis']['is_log'], plot_y_log=plot_kwargs['yaxis']['is_log'], 
                           use_cbar=plot_kwargs['cbar']['plot'], log_cbar=plot_kwargs['cbar']['log_cbar'],
                           obs_plot_kwargs=obs_plot_kwargs)
            
            
            
    elif type_.lower() == 'metallicity' or type_.lower() == 'metallicity ratio':
        
        ## Plot individual Werner+2006 profiles
        plot_ind_metallicity_werner06(type_=type_, axes=axes, bin_=bin_,
                                      bin_low=bin_low, bin_high=bin_high,
                                      element=element, element_num=element_num, element_denom=element_denom,
                                      plot_kwargs=obs_plot_kwargs['Werner+06']['ind'], fig_kwargs=plot_kwargs)
        
        ## Plot individual Grange+2011 profiles
        plot_ind_metallicity_grange11(type_=type_, axes=axes, bin_=bin_,
                                      bin_low=bin_low, bin_high=bin_high,
                                      element=element, element_num=element_num, element_denom=element_denom,
                                      plot_kwargs=obs_plot_kwargs['Grange+11']['ind'], fig_kwargs=plot_kwargs)
        
        ## Plot Mernier+17 average profiles (from paper)
        plot_metallicity_mernier17(type_=type_, axes=axes, bin_=bin_,
                                  element=element, element_num=element_num, element_denom=element_denom,
                                  plot_kwargs=obs_plot_kwargs['Mernier+17']['average'], fig_kwargs=plot_kwargs)
        
        ## Plot Mernier+17 individual profiles
        plot_ind_metallicity_mernier17(type_=type_, axes=axes, bin_=bin_, bin_low=bin_low, bin_high=bin_high, 
                                       element=element, element_num=element_num, element_denom=element_denom, 
                                       plot_kwargs=obs_plot_kwargs['Mernier+17']['ind'], fig_kwargs=plot_kwargs)
        
        ## Calcuate and plot Mernier+17 average profile from individual profiles
        plot_averaging_metallicity_mernier17(type_=type_, axes=axes, bin_=bin_, bin_low=bin_low, bin_high=bin_high, 
                                             element=element, element_num=element_num, element_denom=element_denom, sim_rbins=sim_rbins,
                                             plot_kwargs=obs_plot_kwargs['Mernier+17']['averaging'], fig_kwargs=plot_kwargs)
            
        ## Calcuate and plot Mernier+17 combined average profile from individual profiles
#         plot_combined_stacks_mernier17(type_=type_, axes=axes, bin_low=bin_low, bin_high=bin_high, 
#                                        element=element, element_num=element_num, element_denom=element_denom, sim_rbins=sim_rbins,
#                                        plot_kwargs=obs_plot_kwargs['Mernier+17']['combining_stacks'], fig_kwargs=plot_kwargs)
        
        ## Plot Mernier+2018b average profile directly from paper
        plot_avg_metallicity_mernier18b(type_=type_, axes=axes, bin_=bin_,
                                        element=element, element_num=element_num, element_denom=element_denom,
                                        plot_kwargs=obs_plot_kwargs['Mernier+18b']['average'], fig_kwargs=plot_kwargs)
        
        ## Calcuate and plot Mao+2019 average profile from individual profiles
        plot_avg_metallicity_mao19(type_=type_, axes=axes, bin_=bin_,
                                   bin_low=bin_low, bin_high=bin_high, sim_rbins=sim_rbins,
                                   element=element, element_num=element_num, element_denom=element_denom,
                                   plot_kwargs=obs_plot_kwargs['Mao+19']['averaging'], fig_kwargs=plot_kwargs)
        
        ## Plot Ghizzardi+2021 average profile directly from paper
        plot_avg_metallicity_ghizzardi21(type_=type_, axes=axes, bin_=bin_, bin_low=bin_low, bin_high=bin_high,
                                         element=element, element_num=element_num, element_denom=element_denom,
                                         plot_kwargs=obs_plot_kwargs['Ghizzardi+21']['average'], fig_kwargs=plot_kwargs)
        
        ## Plot Sarkar+22 profiles
        plot_ind_metallicity_sarkar22(type_=type_, axes=axes, bin_=bin_, bin_low=bin_low, bin_high=bin_high, 
                                  element=element, element_num=element_num, element_denom=element_denom,
                                  plot_kwargs=obs_plot_kwargs['Sarkar+22'], fig_kwargs=plot_kwargs)

        ## Calcuate and plot Fukushima+2023 average profile from individual profiles
        plot_avg_metallicity_fukushima23(type_=type_, axes=axes,
                                         bin_low=bin_low, bin_high=bin_high, sim_rbins=sim_rbins,
                                         element=element, element_num=element_num, element_denom=element_denom, 
                                         plot_kwargs=obs_plot_kwargs['Fukushima+23']['averaging'], fig_kwargs=plot_kwargs)
            
            
            
## Plotting function for simulations from other studiess
def plot_simulations(prop, info, bin, element=None, element_num=None, element_denom=None, normalize_abundances=True,
                      axes=plt, logx=False, logy=False, xunits='R500', yunits=None, multiplier=None,
                      print_labels=False, extra_label=None, verbose=False):
    
    if prop in ['Kmass_K500']:
        if info_['profiles']['bin_edges'][bin]>=10**13.5 and info_['profiles']['bin_edges'][bin+1]<=10**14:
            plot_sims_oppenheimer21('entropy', axes=axes, logx=logx, logy=logy, 
                                    xunits=xunits, yunits=yunits, print_labels=print_labels, extra_label=extra_label)

    if prop in ['Pmass_P500']:
        if info_['profiles']['bin_edges'][bin]>=10**13.5 and info_['profiles']['bin_edges'][bin+1]<=10**14:
            plot_sims_oppenheimer21('pressure', axes=axes, logx=logx, logy=logy, 
                                    xunits=xunits, yunits=yunits, print_labels=print_labels, extra_label=extra_label)

    if prop in ['Tmass']:
        if info_['profiles']['bin_edges'][bin]>=10**13.5 and info_['profiles']['bin_edges'][bin+1]<=10**14:
            plot_sims_oppenheimer21('temperature', axes=axes, logx=logx, logy=logy, multiplier=kB**(-1),
                                    xunits=xunits, yunits=yunits, print_labels=print_labels, extra_label=extra_label)

    if prop in ['ne']:
        if info_['profiles']['bin_edges'][bin]>=10**13.5 and info_['profiles']['bin_edges'][bin+1]<=10**14:
            plot_sims_oppenheimer21('density', axes=axes, logx=logx, logy=logy,
                                    xunits=xunits, yunits=yunits, print_labels=print_labels, extra_label=extra_label)