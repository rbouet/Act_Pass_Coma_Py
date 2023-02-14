#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:47:15 2019

MAJ
    13/04/21    add Elan imporatation

@author: romain.bouet
"""


import mne
from mne import io
import sys
sys.path.append('/Users/romain/Datas/Python')

from Utils import Import_Micromed
from Utils import Import_Elan
from Utils import Preproc_RB
from ASR import tools, asr, raw_asrcalibration

import scipy
import numpy as np
import neo

import matplotlib.pyplot as plt
from mne.viz import plot_evoked_topo

from mne.stats import f_threshold_mway_rm, f_mway_rm, fdr_correction



###############################################################################
# Functions Definition
###############################################################################

def Load_Preproc(filename):
    # This function load and preprocess datas
    
    # Import
    file_ext = filename.split(sep = '.')[-1]
    if file_ext == 'eeg':
        raw = Import_Elan.Import_Elan_neo2mne(filename)
        del filename
        print('\nLoad Elan file\n')
 
    elif file_ext == 'TRC':
        raw = Import_Micromed.Import_Micromed_neo2mne(filename)
        del filename
        print('\nLoad Micromed file\n')
 
    else:
        print('Incorrect file extention')
    del file_ext
    
    
    
    ## PREPROCESSING  _________________________________________________________    
    # layout
    # montage = mne.channels.read_montage('standard_1020')
    montage = mne.channels.make_standard_montage('standard_1020')
    # raw.set_montage(montage = montage)
    del montage
    
    
    
    # mne.pick_channels(raw.info['ch_names'], include = montage.ch_names)
        
           
    # reference
    # raw, _ = mne.set_eeg_reference(raw, ['M1', 'M2'], projection=False)
        
    return raw





def Correction_ICA(raw, picks):    
    # this function correct blink artefacts by ICA
    # there are few channels so we kept EOG+ channel to improve a specific component
    
    raw_copy = raw.copy()
    raw_out = raw.copy()
     
    ica = mne.preprocessing.ICA(method='fastica').fit(raw_copy.filter(l_freq = 0.1, h_freq = 20, picks = picks), picks = picks)
    #ica.plot_sources(inst = raw_copy)
    
    eog_epochs = mne.preprocessing.create_eog_epochs(raw, ch_name = 'EOG+')  # get single EOG trials
    eog_inds, scores = ica.find_bads_eog(eog_epochs,l_freq = 0.1, h_freq = 20)  # find via correlation
    remove_comp = np.where(np.abs(scores) == np.max(np.abs(scores)))[0][0]
    
    if np.abs(scores[remove_comp]) > 0.4:
        ica.exclude.extend([remove_comp])
        raw_corr_ica = ica.apply(raw_out)
    else:
        raw_corr_ica = raw.copy()
       
    

    del raw_copy, raw_out, 
    
    return raw_corr_ica, ica, np.abs(scores[remove_comp])





def Correction_Regression(raw, picks):        
    # this function correct blink artefacts by EOG regression 
    
    raw_corr_reg = raw.copy()
    raw_corr_eog = raw.copy().filter(l_freq = 0.1, h_freq = 45)
    raw_corr_reg._data[picks,] = Preproc_RB.EOG_regression_correction(raw_corr_eog.get_data()[raw.info['ch_names'].index('EOG+'),], 
                                                                          raw.get_data()[picks,])

    return raw_corr_reg



def ASR_Calib(raw_ref, ChanName4VEOG):
    
    """
    ASR calibration 
    On continu signal
    
    EX : ASR_Calib(raw.crop(tmin = tmin, tmax = tmax),
                            'EOG+')
    
    """
    
    # Calibration    
    rawCalibAsr = raw_ref.copy()
    del raw_ref
    # ChanName4VEOG = ['EOG+']
    cutoff = 5
    Yule_Walker_filtering = True
    state = raw_asrcalibration.raw_asrcalibration(rawCalibAsr, ChanName4VEOG, cutoff, Yule_Walker_filtering)

    return state


def ASR_Correction(raw_epoch, state):
    
    """
    ASR Correction   
    On evoked signal
    Need "state" from ASR_Calib()

    EX: ASR_Correction(mne.Epochs(raw_corr.filter(0, 30), events, **cfg, reject = def_th_reject), 
                       state)   
    
    """
    
    Data4detect = raw_epoch.copy().get_data()
    Data2Correct = raw_epoch.copy().get_data()
    
    DataClean = np.zeros((Data2Correct.shape))
    
    for i_epoch in range(Data4detect.shape[0]):
        
        EpochYR = Data4detect[i_epoch,:,:]
        Epoch2Corr = Data2Correct[i_epoch,:,:]
        
        DataClean[i_epoch,:,:] = asr.asr_process_on_epoch(Epoch2Corr,
                                                          EpochYR,
                                                          state)

    # Build new MNE epoched object
    epochs_clean = mne.EpochsArray(DataClean,
                                   info = raw_epoch.info,
                                   events = raw_epoch.events,
                                   event_id = raw_epoch.event_id,
                                   tmin = raw_epoch.times[0])
    

    return epochs_clean





def Clean_events_after(events):
    # this function remove standart after deviant
    
    events_copy = events.copy()
    del events
    
    for lab_std in [12,22,32]:        
        events_copy[np.where(events_copy == lab_std)[0]+1]=6693
    
    return events_copy
    




def Clean_events_befor(events):
    # this function keep only standart befor deviant
    
    events_copy = events.copy()
    del events
    
    for lab_std in [12,22,32]:        
        id_befor = np.where(events_copy == lab_std)[0]-1
        events_copy[np.where(events_copy == lab_std-1)[0]]  = 6693
        events_copy[id_befor] = lab_std - 1
               
    
    return events_copy





def Disp_stat_clust_one(condition1, condition2, title):
    # This function comput end display the clustering statistirc 
    # only for one channel
    
    threshold = 6.0
    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test([condition1, condition2], n_permutations = 1000,
                                                                               threshold = threshold, tail = 0, n_jobs = 1)
    
    
    # Display
    times = epoch_reject.times
    plt.figure(figsize=(20, 20))
    plt.subplot(211)
    plt.plot(times, condition1.mean(axis=0), 'g--', label='Standart')
    plt.plot(times, condition2.mean(axis=0), 'r--', label='Deviant)')
    plt.plot(times, condition2.mean(axis=0) - condition1.mean(axis=0), 'k-', label='ERF Contrast (Deviant - Standard)')
    plt.title(title)
    plt.legend(loc='upper left')

    plt.subplot(212)
    plt.plot(times, T_obs, 'g')
    plt.xlabel("time (ms)")
    plt.ylabel("f-values")
    plt.show()
    for i_c, c in enumerate(clusters):
        c = c[0]
        if cluster_p_values[i_c] <= 0.05:
            h = plt.axvspan(times[c[0]], times[c[-1] - 1],
                            color='r', alpha=0.3)
            plt.legend((h, ), ('cluster p-value < 0.05', ))
        else:
            plt.axvspan(times[c[0]], times[c[-1] - 1], color=(0.3, 0.3, 0.3),
                        alpha=0.3)
            
    



def Disp_stat_clust_all(epochs,
                        cond1_name,cond2_name,
                        title):
    # This function comput end display the clustering statistic 
    # for several channels
    
    
    picks = mne.pick_types(epochs.info, meg  =False, 
                           eeg  = True, 
                           eog  = True,
                           stim = False, exclude='bads')
    
    condition1 = epochs[cond1_name].get_data()
    condition2 = epochs[cond2_name].get_data()
    
    Ymax  =  np.abs(np.append(condition1.mean(axis=0), condition2.mean(axis=0))).max() + 1e-6
    
    
    f = plt.figure(figsize=(20, 20))
    plt.suptitle(title)
    
    for place, xi_chan in enumerate(picks):
        
        ###### STATISTIC
        #T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test([condition1[:,xi_chan,:], condition2[:,xi_chan,:]], n_permutations = 1000,
        #                                                                       threshold = 3.868241491981564, tail = 1, n_jobs = 3)
        
        n_conditions = 2
        n_replications = (condition1.shape[0] + condition1.shape[0])  // n_conditions
        factor_levels = [2]      #[2, 2]  # number of levels in each factor
        effects = 'A'  # this is the default signature for computing all effects
        # Other possible options are 'A' or 'B' for the corresponding main effects
        # or 'A:B' for the interaction effect only
        
        pthresh = 0.05  # set threshold rather high to save some time
        f_thresh = f_threshold_mway_rm(n_replications,
                                       factor_levels,
                                       effects,
                                       pthresh)
        del n_conditions, n_replications, factor_levels, effects, pthresh
        
        tail = 1  # f-test, so tail > 0
        T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test([condition1[:,xi_chan,:], condition2[:,xi_chan,:]],
                                                                                   threshold = f_thresh, 
                                                                                   tail = tail, 
                                                                                   n_jobs = 3, n_permutations = 1000)
        del f_thresh, tail
        
        
        
        
        ###### DISPLAY
        times = epochs.times
        plt.subplot(np.int16(np.fix(np.sqrt(picks.size))),
                    np.int16(np.ceil(np.sqrt(picks.size))),
                    place+1)
        
        times[0].shape
        condition1.shape
  
        plt.plot([0, 0], [-Ymax, Ymax], c = [0.8, 0.8, 0.8], lw = 1, ls = '-')
        plt.plot([0.1, 0.1], [-Ymax, Ymax], c = [0.8, 0.8, 0.8], lw = 1, ls = ':')
        plt.plot([-0.1, 1], [0,0], c = [0.8, 0.8, 0.8], lw = 1, ls = '-')
      
        h_cond1 = plt.plot(times, condition1[:,xi_chan,:].mean(axis=0), 'g--', label = cond1_name + '  n=' + str(condition1.shape[0]))
        plt.fill_between(times, 
                         condition1[:,xi_chan,:].mean(axis=0)-(condition1[:,xi_chan,:].std(axis=0)/np.sqrt(condition1.shape[0])),
                         condition1[:,xi_chan,:].mean(axis=0)+(condition1[:,xi_chan,:].std(axis=0)/np.sqrt(condition1.shape[0])),
                         color = 'g', alpha = 0.2)
        h_cond2 = plt.plot(times, condition2[:,xi_chan,:].mean(axis=0), 'r--', label = cond2_name + '  n=' + str(condition2.shape[0]))
        plt.fill_between(times, 
                         condition2[:,xi_chan,:].mean(axis=0)-(condition2[:,xi_chan,:].std(axis=0)/np.sqrt(condition2.shape[0])),
                         condition2[:,xi_chan,:].mean(axis=0)+(condition2[:,xi_chan,:].std(axis=0)/np.sqrt(condition2.shape[0])),
                         color = 'r', alpha = 0.2)
        h_diff = plt.plot(times, condition2[:,xi_chan,:].mean(axis=0) - condition1[:,xi_chan,:].mean(axis=0), 'k-', label = 'Contrast   ' + cond2_name + ' - ' + cond1_name)
        plt.title(epochs.info['ch_names'][xi_chan])
        
        
        plt.xlim([-0.1, 1])
        plt.ylim([-Ymax, Ymax])
        # inverse y axis
        plt.gca().set_ylim(plt.gca().get_ylim()[::-1])

        for i_c, c in enumerate(clusters):
            c = c[0]

            if cluster_p_values[i_c] <= 0.05:
                h_clust = plt.axvspan(times[c[0]], times[c[-1] - 1],
                                      color='r', alpha=0.3, label = 'cluster p-value < 0.05')
            else:
                plt.axvspan(times[c[0]], times[c[-1] - 1], color=(0.3, 0.3, 0.3),
                            alpha=0.3)
            
        if xi_chan == 0:  #condition1.shape[1]-1:
           plt.legend() 
           #plt.legend(loc='upper center', bbox_to_anchor=(1, 0.8))
            


    return f
    
    
    
def Eval_rejection(raw_corr_ica, events, cfg, def_ampl = 10e-5, def_pct = 10):

    """
    Threshold search provide < 10 % of signal rejection
    
    """
    raw_rej = raw_corr_ica.copy() 
    
    nb_std_pass = mne.Epochs(raw_rej.filter(0.1, 30), events, **cfg)['Std/Pass'].get_data().shape[0]
    nb_dev_pass = mne.Epochs(raw_rej.filter(0.1, 30), events, **cfg)['Dev/Pass'].get_data().shape[0]
    nb_std_dist = mne.Epochs(raw_rej.filter(0.1, 30), events, **cfg)['Std/Dist'].get_data().shape[0]
    nb_dev_dist = mne.Epochs(raw_rej.filter(0.1, 30), events, **cfg)['Dev/Dist'].get_data().shape[0]
    nb_std_act  = mne.Epochs(raw_rej.filter(0.1, 30), events, **cfg)['Std/Act'].get_data().shape[0]
    nb_dev_act  = mne.Epochs(raw_rej.filter(0.1, 30), events, **cfg)['Dev/Act'].get_data().shape[0]
    
    max_rej = 101
    # 10 % signal rejected max
    while max_rej > def_pct:
        rej_std_pass = np.round(1-mne.Epochs(raw_rej.filter(0.1, 30), events, **cfg, reject = def_ampl)['Std/Pass'].get_data().shape[0]/nb_std_pass,2)*100
        rej_dev_pass = np.round(1-mne.Epochs(raw_rej.filter(0.1, 30), events, **cfg, reject = def_ampl)['Dev/Pass'].get_data().shape[0]/nb_dev_pass,2)*100
        rej_std_dist = np.round(1-mne.Epochs(raw_rej.filter(0.1, 30), events, **cfg, reject = def_ampl)['Std/Dist'].get_data().shape[0]/nb_std_dist,2)*100
        rej_dev_dist = np.round(1-mne.Epochs(raw_rej.filter(0.1, 30), events, **cfg, reject = def_ampl)['Dev/Dist'].get_data().shape[0]/nb_dev_dist,2)*100
        rej_std_act  = np.round(1-mne.Epochs(raw_rej.filter(0.1, 30), events, **cfg, reject = def_ampl)['Std/Act'].get_data().shape[0]/nb_std_act,2)*100
        rej_dev_act  = np.round(1-mne.Epochs(raw_rej.filter(0.1, 30), events, **cfg, reject = def_ampl)['Dev/Act'].get_data().shape[0]/nb_dev_act,2)*100

        max_rej = np.max([rej_std_pass, rej_std_dist, rej_std_act, rej_dev_pass, rej_dev_dist, rej_dev_act])
        def_ampl['eeg'] = def_ampl['eeg']+1e-5
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects_std  = ax.bar(np.arange(3)-0.2, [rej_std_pass, rej_std_dist, rej_std_act], width = 0.4, color='green', label = 'Std')
    rects_dev = ax.bar(np.arange(3)+0.2, [rej_dev_pass, rej_dev_dist, rej_dev_act],width = 0.4, color='red', label = 'Dev')
    plt.setp(ax.set_xticklabels(['', '', 'Passive', '', 'Distractive', '',  'Active']), rotation=35, fontsize=10)
    ax.set_ylabel('% rejected')
    ax.set_title('th_ampl = ' + str(np.round(def_ampl['eeg'], 6)) + 'V    th_pct = ' + str(def_pct) + '%')
    plt.legend() 
#    plt.ylim([0, 30])
    
    del raw_rej, ax, rects_std, rects_dev
    return [fig, def_ampl]


    
if __name__ == '__main__':
    Load_Preproc()
    Correction_ICA()
    Correction_Regression()
    ASR_Calib()
    ASR_Correction()
    Clean_events_after()
    Clean_events_befor()
    Disp_stat_clust_one()
    Disp_stat_clust_all()
    Eval_rejection()