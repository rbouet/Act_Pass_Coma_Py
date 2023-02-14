#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 17:15:04 2021

@author: romain.bouet
"""


import mne
import neo
import numpy as np

def Import_Elan_neo2mne(filename):

    raw_elan = neo.elanio.ElanIO(filename = filename)

        
    # Read with neo
    seg = raw_elan.read_segment()
    
    ### Get informations and data to convert it to MNE needs
    
    sFreq = seg.analogsignals[0].sampling_rate.rescale('Hz').magnitude

    # Datas
#    convert_fac = 1. * seg.analogsignals[0].units
#    convert_fac.units = 'uV' ## Convertion to Volts
#    data = np.asarray(seg.analogsignals[0]).T
    data = seg.analogsignals[0].rescale('V').magnitude.T
#    data = data * convert_fac.magnitude  
    data = np.concatenate((data, np.zeros((1,np.shape(data)[1]))), axis=0) #adding trigger chan
        
    
    # Triggers
    events_time = seg.events[0].times * sFreq
    nb_event = len(events_time) 
    events =  np.zeros([nb_event,3])
    events[:,0] = events_time.astype(int)
    events[:,2] = seg.events[0].labels.astype(int)
    
  
    
    # Channels names and types
    chan = seg.analogsignals[0].name.split(sep = ',')
    chan[0]  = chan[0].split(sep = '(')[1]
    chan[-1] = chan[-1].split(sep = ')')[0]
    
    ch_types = list()
    chan_short = list()
    for xi, elem in enumerate(chan):
        chan_short.append(elem.split(sep = '.')[0])
        if (elem == 'EOG+'):
            ch_types.append('eog')
        else:
            ch_types.append('eeg')
    del chan
            
    
    # Add stim channel
    chan_short.append('STI 014')
    ch_types.append('stim')
    
    
    seg.annotate(material = "elan")
    
    
     ###MNE raw data format creation with data, infos and events
    raw_info = mne.create_info(chan_short, sFreq, ch_types)    
    raw = mne.io.RawArray(data, raw_info, verbose =1)
    raw.add_events(events, 'STI 014')
    
    return raw
    
if __name__ == '__main__':
    Import_Elan_neo2mne()