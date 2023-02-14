#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 16:02:31 2020

@author: romain.bouet
"""

import sys
sys.path.append('/Users/romain/Datas/Python')

from Utils import ActPass_analysis_General_V04




filename = '/Users/romain/Study/Coma/Act_Pass/datas_raw/JJ/EEG_2314_ActPass.TRC'
filename = '/Users/romain/Study/Coma/Act_Pass/datas_raw/P14/EEG_202.TRC'
filename = '/Users/romain/Study/Coma/Act_Pass/datas_raw/P16/EEG_245.TRC'
filename = '/Users/romain/Study/Coma/Act_Pass/datas_raw/P17/EEG_256.TRC'

out_path = '/Users/romain/Study/Coma/Act_Pass/Figures/'

ActPass_analysis_General_V04.Lance_All_Analyse(filename, out_path, 'None', 10e-5, 100)   
ActPass_analysis_General_V04.Lance_All_Analyse(filename, out_path, 'Regress')   
ActPass_analysis_General_V04.Lance_All_Analyse(filename, out_path, 'Regress', 10e-5, 100)   
# ActPass_analysis_General_V04.Lance_All_Analyse(filename, out_path, 'ASR')   
ActPass_analysis_General_V04.Lance_All_Analyse(filename, out_path, 'ICA', 10e-5, 100)   




raw = Import_Micromed.Import_Micromed_neo2mne(filename)
raw.plot(lowpass=30, n_channels = 5)


