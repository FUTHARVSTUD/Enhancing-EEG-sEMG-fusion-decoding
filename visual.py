### Import #####################
import os,mne,numpy as np, pandas as pd
from mne.preprocessing import ICA
import bad_eeg_chs from global_defn.py
#############################################

################## Customization ##################
# data_dir = 'MultiEEGEMG_stroke/'
data_dir = 'MultiEEGEMG_stroke'
subj_idx = '50'
contraction_type = 'iVC'
session_idx = 's06'
sfreq_emg=1000
sfreq_eeg=500
reject_criteria_eeg = dict(eeg=6e-4)       # 600 μV, do not exclude epochs containing ocular artifact
flat_criteria_eeg = dict(eeg=1e-6)           # 1 μV


################ fName
emg_fName = os.path.join(data_dir,'subj'+subj_idx,'EMG','subj'+subj_idx+'_'+contraction_type+'_'+session_idx+'.txt')
eeg_fName = os.path.join(data_dir,'subj'+subj_idx,'EEG','subj'+subj_idx+'_'+contraction_type+'_'+session_idx+'.set')
ica_dir = os.path.join(data_dir,'subj'+subj_idx,'ica')
epochs_beforeICA_dir = os.path.join(data_dir,'subj'+subj_idx,'epochs_beforeICA')
epochs_preped_dir = os.path.join(data_dir,'subj'+subj_idx,'epochs_preped')
epochs_hybrid_dir = os.path.join(data_dir,'subj'+subj_idx,'epochs_hybrid')
if not os.path.exists(ica_dir):
    os.makedirs(ica_dir)
if not os.path.exists(epochs_beforeICA_dir):
    os.makedirs(epochs_beforeICA_dir)
if not os.path.exists(epochs_preped_dir):
    os.makedirs(epochs_preped_dir)
if not os.path.exists(epochs_hybrid_dir):
    os.makedirs(epochs_hybrid_dir)
    
ica_push_fName = os.path.join(data_dir,
                              'subj'+subj_idx,'ica','subj'+subj_idx+'_'+contraction_type+'_'+session_idx+'_push_ica.fif')
ica_pull_fName = os.path.join(data_dir,
                              'subj'+subj_idx,'ica','subj'+subj_idx+'_'+contraction_type+'_'+session_idx+'_pull_ica.fif')

epochs_beforeICA_push_fName = os.path.join(data_dir,'subj'+subj_idx,'epochs_beforeICA',
                                      'subj'+subj_idx+'_'+contraction_type+'_'+session_idx+'_beforeICA_push_epo.fif')
epochs_beforeICA_pull_fName = os.path.join(data_dir,'subj'+subj_idx,'epochs_beforeICA',
                                      'subj'+subj_idx+'_'+contraction_type+'_'+session_idx+'_beforeICA_pull_epo.fif')

epochs_preped_push_fName = os.path.join(data_dir,'subj'+subj_idx,'epochs_preped',
                                      'subj'+subj_idx+'_'+contraction_type+'_'+session_idx+'_preped_push_epo.fif')
epochs_preped_pull_fName = os.path.join(data_dir,'subj'+subj_idx,'epochs_preped',
                                      'subj'+subj_idx+'_'+contraction_type+'_'+session_idx+'_preped_pull_epo.fif')

epochs_hybrid_push_fName = os.path.join(data_dir,'subj'+subj_idx,'epochs_hybrid',
                                      'subj'+subj_idx+'_'+contraction_type+'_'+session_idx+'_hybrid_push_epo.fif')
epochs_hybrid_pull_fName = os.path.join(data_dir,'subj'+subj_idx,'epochs_hybrid',
                                      'subj'+subj_idx+'_'+contraction_type+'_'+session_idx+'_hybrid_pull_epo.fif')

alignmentInfo_fName = os.path.join(data_dir,'subj'+subj_idx,'subj'+subj_idx+'_alignmentInfo.txt')
alignmentInfo = pd.read_csv(alignmentInfo_fName, skiprows=0, sep = ',',engine = 'python')
############################################################

####### eeg reading ###########
raw_eeg = mne.io.read_raw_eeglab(eeg_fName,preload=True)
raw_eeg.set_montage('standard_1020')
if subj_idx not in bad_eeg_chs.keys():
    print('please add '+subj_idx+'in bad chs list')
raw_eeg.info['bads']=bad_eeg_chs[subj_idx][contraction_type][session_idx]
raw_eeg.crop(tmin = alignmentInfo.loc[(alignmentInfo['sessionIdx']==session_idx) & 
                                      (alignmentInfo['contraction_type']==contraction_type),
                                      'EEG'].values[0]/raw_eeg.info['sfreq'])
raw_eeg.filter(l_freq=1,h_freq=45)
########################################
raw_eeg.plot(scalings=dict(eeg=1e-4),duration = 10, n_channels = 18)
