#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:47:15 2019

@author: romain.bouet
"""

import mne
import neo
import numpy as np

def EOG_regression_correction(eog_vect, eeg_vect):
    
    # EEG correction by EOG regression
    #
    # only one EOG vector
    #
    # according to    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8369420
    
    beta = (eog_vect*eeg_vect).sum(axis = 1)/(eog_vect*eog_vect).sum()
    eeg_corrected = eeg_vect - (eog_vect.repeat(eeg_vect.shape[0]).reshape(eog_vect.shape[0],eeg_vect.shape[0])*beta).T

    
    return eeg_corrected
    

if __name__ == '__main__':
    EOG_regression_correction()