#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:47:15 2019

@author: romain.bouet


MAJ:
    11:01:22     rb
    update according to neo.0.10
    channel names extraction
"""

import mne
import neo
import numpy as np

def Import_Micromed_neo2mne(filename):

    raw_micromed = neo.MicromedIO(filename = filename)
    
    # Read with neo
    seg = raw_micromed.read_segment()
    
    ### Get informations and data to convert it to MNE needs
    
    sFreq = seg.analogsignals[0].sampling_rate 
    
    # Datas
    convert_fac = 1. * seg.analogsignals[0].units
    convert_fac.units = 'V' ## Convertion to Volts
    data = np.asarray(seg.analogsignals[0]).T
    data = data * convert_fac.magnitude  
    data = np.concatenate((data, np.zeros((1,np.shape(data)[1]))), axis=0) #adding trigger chan
        
    
    # Triggers
    events_time = seg.events[0].times * sFreq
    nb_event = len(events_time) 
    events =  np.zeros([nb_event,3])
    events[:,0] = events_time
    events[:,2] = seg.events[0].labels.astype(int)
    
    
    # Channels names and types    
    # chan = seg.analogsignals[0].name.split(sep = ',')
    # chan[0]  = chan[0].split(sep = '(')[1]
    # chan[-1] = chan[-1].split(sep = ')')[0]
    
    # ch_types = list()
    # for xi, elem in enumerate(chan):
    #     if (elem == 'EOG+'):
    #         ch_types.append('eog')
    #     else:
    #         ch_types.append('eeg')
             
     
    chan = raw_micromed.header['signal_channels']
    ch_types = list()
    chan     = list()
    for id_chan in range(0, raw_micromed.header['signal_channels'].shape[0]):
        if (raw_micromed.header['signal_channels'][id_chan][0] == 'EOG+'):
            ch_types.append('eog')
            chan.append(raw_micromed.header['signal_channels'][id_chan][0])
        else:
            ch_types.append('eeg')
            chan.append(raw_micromed.header['signal_channels'][id_chan][0])
    
        
    
    # Add stim channel
    chan.append('STI 014')
    ch_types.append('stim')
    
    
    seg.annotate(material = "micromed")
    
    
     ###MNE raw data format creation with data, infos and events
    raw_info = mne.create_info(chan, sFreq, ch_types)    
    raw = mne.io.RawArray(data, raw_info, verbose =1)
    raw.add_events(events, 'STI 014')
    
    return raw
    
if __name__ == '__main__':
    Import_Micromed_neo2mne()