#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 10:27:46 2019

@author: romain.bouet
"""



import os
import mne
from mne import io
import sys
sys.path.append('/Users/romain.bouet/Datas/Python/')

from Utils import Import_Micromed
from Utils import Preproc_RB
from Utils import Coma_Analysis_Def
from ASR import tools, asr, raw_asrcalibration

import scipy
import numpy as np
import neo

import matplotlib.pyplot as plt
from mne.viz import plot_evoked_topo



#filename = '/Volumes/crnldata/dycog/dom/ActPass/P65/EEG_1881.TRC'
#filename = '/Volumes/crnldata/dycog/Epilepto/Coma/Act_Pass/datas_raw/JJ/EEG_2314_ActPass.TRC'

def Lance_All_Analyse(filename, out_path, correct_methodo):

    """
    
    correct_methodo: 'Regress', 'ICA' or 'ASR'
    
    """
    
    
    
    os.mkdir(out_path + filename.split(os.sep)[-2] + "_" + correct_methodo)
    prefix_out_filename =  filename.split(os.sep)[-2] + "_" + correct_methodo + os.sep + filename.split(os.sep)[-2] + '_' + os.path.splitext(filename.split(os.sep)[-1])[0] 
    
    raw = Coma_Analysis_Def.Load_Preproc(filename)

    # raw.plot(n_channels = 10)
    
    # bad channels
#    raw.info['bads']+=['EMG1+', 'EMG2+','EMG3+','EMG4+',
#                       'PNG1+','PNG2+','PNG3+','PNG4+',
#                       'ECG1+',
#                       'MKR+', 'STI 014',
#                       'O2', 'M1', 'M2']
    
#    raw.info['bads']+=['MKR+', 'STI 014',
#                        'M1', 'M2']
    
    
    ch_name_raw = set(raw.info['ch_names'])
    bads_test = set(['EMG1+', 'EMG2+','EMG3+','EMG4+',
                     'PNG1+','PNG2+','PNG3+','PNG4+',
                     'ECG1+',
                     'MKR+', 'STI 014',
                     'O2', 'M1', 'M2'])
    bads_channels = ch_name_raw.intersection(bads_test)
    raw.info['bads']+= bads_channels
    del(bads_test, ch_name_raw)
    
    picks = mne.pick_types(raw.info, meg  =False, 
                           eeg  = True, 
                           eog  = True,
                           stim = False, exclude='bads')
        
    
    
    
    
    
    
    #______________________________________________________________________________
    # Events
    events =  mne.find_events(raw, stim_channel = 'STI 014')
    
    events[:,2] =  Coma_Analysis_Def.Clean_events_after(events[:,2])
    #events[:,2] =  Coma_Analysis_Def.Clean_events_befor(events[:,2])
    
    event_id = {'Std/Pass': 11, 'Dev/Pass': 12,
                'Std/Dist': 21,  'Dev/Dist': 22,
                'Std/Act': 31, 'Dev/Act': 32,}
    
    color = {11: 'green', 12: 'yellow', 21: 'red', 22: 'c', 31: 'black', 32: 'blue'}
    
    mne.viz.plot_events(events, raw.info['sfreq'], raw.first_samp, color=color,
                        event_id=event_id).savefig(out_path + prefix_out_filename + '_Event.pdf')
    
   
    
    
    
    #______________________________________________________________________________
    # Correction


    if correct_methodo == "ICA":
        
        raw_corr, ica, score_ica = Coma_Analysis_Def.Correction_ICA(raw, picks)
        
        if not ica:
            ica.plot_sources(inst = raw, exclude = 6, start = 0, stop = 120).savefig(out_path +prefix_out_filename + '_ICA.pdf')
        else:
            ica.plot_sources(inst = raw, start = 0, stop = 120).savefig(out_path + prefix_out_filename + '_ICA_comp.pdf')
            ica.plot_components(inst = raw)
        del(ica)
        
    elif correct_methodo == "Regress":
        
        raw_corr = Coma_Analysis_Def.Correction_Regression(raw, picks)
    
    elif correct_methodo == "ASR":

        raw_corr = raw.copy().filter(l_freq = 2, h_freq = 30, picks = picks)
        state_2_30 = Coma_Analysis_Def.ASR_Calib(raw_corr.copy().drop_channels(raw_corr.info['bads']),
                                                 'EOG+')
        del raw_corr
        raw_corr = raw.copy().filter(l_freq = 0, h_freq = 30, picks = picks)
        state_0_30 = Coma_Analysis_Def.ASR_Calib(raw_corr.copy().drop_channels(raw_corr.info['bads']),
                                                 'EOG+')
  
    elif correct_methodo == 'None':
    
        raw_corr = raw.copy()


    #______________________________________________________________________________
    # Epoching properties
    cfg = dict(baseline = (-0.1, 0),
               tmin     = -0.2,
               tmax     = 1 ,
               event_id = event_id,
               proj     = True,
               picks    = picks,
               preload  = False,
               detrend = 1)



    #______________________________________________________________________________
    # artefact threshold search + Epoching 
    if correct_methodo != 'ASR':
        
        def_th_reject = dict(eeg = 10e-5)
        fig_rej, def_th_reject = Coma_Analysis_Def.Eval_rejection(raw_corr, events, cfg, def_th_reject)
        fig_rej.savefig(out_path + prefix_out_filename + '_PCT_Rejection.pdf')
        del fig_rej
        
        epoch_corr = mne.Epochs(raw_corr.copy().filter(0, 30), events, **cfg, reject = def_th_reject)



    # Epoching + ASR correction
    if correct_methodo == 'ASR':
        
        def_th_reject = dict(eeg = 1)
        epoch_to_clean = mne.Epochs(raw_corr.copy().filter(0, 30), events, **cfg, reject = def_th_reject)
        
        epoch_corr = Coma_Analysis_Def.ASR_Correction(epoch_to_clean,
                                                      state_0_30)

        del epoch_to_clean
        
       
        
     
    
    #______________________________________________________________________________
    # Stat
    
    Coma_Analysis_Def.Disp_stat_clust_all(epoch_corr,
                                          'Std/Dist',
                                          'Dev/Dist',
                                          'cond: Distractive    cor: ' + correct_methodo).savefig(out_path + prefix_out_filename + '_Distractive_STD_DEV.pdf')
    
    Coma_Analysis_Def.Disp_stat_clust_all(epoch_corr,
                                          'Std/Act',
                                          'Dev/Act',
                                          'cond: Active    cor: ' + correct_methodo).savefig(out_path + prefix_out_filename + '_Active_STD_DEV.pdf')
    
    Coma_Analysis_Def.Disp_stat_clust_all(epoch_corr,
                                          'Dev/Dist',
                                          'Dev/Act',
                                          'cond: Active/Dist    cor: ' + correct_methodo).savefig(out_path + prefix_out_filename + '_ActDis_DEV.pdf')
    
    Coma_Analysis_Def.Disp_stat_clust_all(epoch_corr,
                                          'Std/Dist',
                                          'Std/Act',
                                          'cond: Active/Dist    cor: ' + correct_methodo).savefig(out_path + prefix_out_filename + '_ActDis_STD.pdf')
    
    



##______________________________________________________________________________
## ERP standards

    event_std = events.copy()
    
    event_std[np.where(event_std[:,2] == 11),2] = 1111
    event_std[np.where(event_std[:,2] == 21),2] = 1111
    event_std[np.where(event_std[:,2] == 31),2] = 1111
    event_std[np.where(event_std[:,2] == 12),2] = 2222
    event_std[np.where(event_std[:,2] == 22),2] = 2222
    event_std[np.where(event_std[:,2] == 32),2] = 2222
    
    
    picks = mne.pick_types(raw.info, meg  =False, 
                           eeg  = True, 
                           eog  = True,
                           stim = False, exclude='bads')
        
    
    cfg = dict(baseline = (-0.1, 0),
               tmin     = -0.2,
               tmax     = 0.7 ,
               event_id = {'Std': 1111, 'Dev': 2222},
               proj     = True,
               picks    = picks,
               preload  = False,
               detrend = 1)
        
    
    if correct_methodo != 'ASR':    
        
        mne.Epochs(raw_corr.copy().filter(2, 30), event_std, **cfg, reject = def_th_reject).average().plot(titles = 'All Stim').savefig(out_path + prefix_out_filename + '_All_Stim.pdf')
        plt.gca().set_ylim(plt.gca().get_ylim()[::-1])
    
    elif correct_methodo == 'ASR':
    
        #def_th_reject = dict(eeg = 1)
        Coma_Analysis_Def.ASR_Correction(mne.Epochs(raw_corr.copy().filter(2, 30), event_std, **cfg, reject = def_th_reject),
                                         state_2_30).average().plot(titles = 'All Stim')





##______________________________________________________________________________
## MMN    std VS dev    front average  -  mastoid average

    raw_tmp = raw_corr.copy().filter(2, 30)
    
    raw_tmp.info['bads'] = bads_channels
        
    # Fronto-Mastoide difference and other channels usefull
    array_raw = raw_tmp.get_data()
    quel_front = [raw.info['ch_names'].index('F3'),
                  raw.info['ch_names'].index('F4'),
                  raw.info['ch_names'].index('Fz'),
                  raw.info['ch_names'].index('Cz')]
    quel_masto = [raw.info['ch_names'].index('M1'),
                  raw.info['ch_names'].index('M2')]
    
    front = array_raw[np.array(quel_front),:].mean(0).reshape(1, array_raw.shape[1])
    masto = array_raw[np.array(quel_masto),:].mean(0).reshape(1, array_raw.shape[1])
    PZ = array_raw[np.array([raw.info['ch_names'].index('Pz')]),:]
    sti014 = np.zeros((1,array_raw.shape[1]))
    eog = array_raw[np.array([raw.info['ch_names'].index('EOG+')]),:]
    new_datas = np.append(front, masto, axis = 0)
    new_datas = np.append(new_datas, PZ, axis = 0)
    new_datas = np.append(new_datas, front-masto, axis = 0)
    new_datas = np.append(new_datas, eog, axis = 0)
    new_datas = np.append(new_datas, sti014, axis = 0)
    
    del array_raw, front, masto, sti014, eog
    
    chan = list()
    chan.append('4Frontal')
    chan.append('2Mastoid')
    chan.append('PZ')
    chan.append('Fronto-Masto')
    chan.append('EOG+')
    chan.append('STI 014')
    
    ch_types = list()
    ch_types.append('eeg')
    ch_types.append('eeg')
    ch_types.append('eeg')
    ch_types.append('eeg')
    ch_types.append('eog')
    ch_types.append('stim')
    
    sFreq = raw_tmp.info['sfreq']
    
    new_raw_info = mne.create_info(chan, sFreq, ch_types)    
    mean_raw_corr_ica = mne.io.RawArray(new_datas, new_raw_info, verbose =1)
    mean_raw_corr_ica.add_events(events, 'STI 014')
    mean_raw_corr_ica.info['bads'] += ['STI 014']

    del raw_tmp, new_datas, chan, ch_types, sFreq, new_raw_info
    
    # Epoching
    cfg = dict(baseline = (-0.1, 0),
               tmin     = -0.2,
               tmax     = 1 ,
               event_id = event_id,
               proj     = True,
               picks    = mne.pick_types(mean_raw_corr_ica.info, 
                                         meg  =False, 
                                         eeg  = True, 
                                         eog  = True,
                                         stim = False, exclude='bads'),
               preload  = False,
               detrend = 1)
    
    
    if correct_methodo != 'ASR':    
        
        epoch_mean_ica = mne.Epochs(mean_raw_corr_ica, events, **cfg, reject = def_th_reject)
    
    elif correct_methodo == 'ASR':
    
        
        state_2_30 = Coma_Analysis_Def.ASR_Calib(mean_raw_corr_ica.copy().drop_channels(mean_raw_corr_ica.info['bads']),
                                                 'EOG+')
        
        epoch_mean_ica = Coma_Analysis_Def.ASR_Correction(mne.Epochs(mean_raw_corr_ica, events, **cfg, reject = def_th_reject),
                                                          state_2_30)

        
        
        
    
    Coma_Analysis_Def.Disp_stat_clust_all(epoch_mean_ica,
                                          'Std/Pass',
                                          'Dev/Pass',
                                          'cond: Passive    cor: ' + correct_methodo).savefig(out_path + prefix_out_filename + '_Passive_STD_DEV_4F2M.pdf')
    
    Coma_Analysis_Def.Disp_stat_clust_all(epoch_mean_ica,
                                          'Std/Dist',
                                          'Dev/Dist',
                                          'cond: Distractive    cor: ' + correct_methodo).savefig(out_path + prefix_out_filename + '_Distractive_STD_DEV_4F2M.pdf')

    Coma_Analysis_Def.Disp_stat_clust_all(epoch_mean_ica,
                                          'Std/Act',
                                          'Dev/Act',
                                          'cond: Active    cor: ' + correct_methodo).savefig(out_path + prefix_out_filename + '_Active_STD_DEV_4F2M.pdf')
    
    
    del cfg, epoch_mean_ica
    
    
    
    
    cfg = dict(baseline = (-0.1, 0),
               tmin     = -0.2,
               tmax     = 1 ,
               event_id = {'Std': 1111,
                           'Dev': 2222},
               proj     = True,
               picks    = mne.pick_types(mean_raw_corr_ica.info, meg  =False, 
                           eeg  = True, 
                           eog  = True,
                           stim = False, exclude='bads'),
               preload  = False,
               detrend = 1)
    
    
    if correct_methodo != 'ASR':    
        
        epoch_mean_ica = mne.Epochs(mean_raw_corr_ica, event_std, **cfg, reject = def_th_reject)
    
    elif correct_methodo == 'ASR':

        #def_th_reject = dict(eeg = 1)
        epoch_mean_ica = Coma_Analysis_Def.ASR_Correction(mne.Epochs(mean_raw_corr_ica, event_std, **cfg, reject = def_th_reject),
                                                          state_2_30)
 
       
    

    
    Coma_Analysis_Def.Disp_stat_clust_all(epoch_mean_ica,
                                          'Std',
                                          'Dev',
                                          'cond: ALL    cor: ' + correct_methodo).savefig(out_path + prefix_out_filename + '_ALL_STD_DEV_4F2M.pdf')
    
    
    del cfg, epoch_mean_ica


    plt.close('all')





    
if __name__ == '__main__':
    Lance_All_Analyse()

   