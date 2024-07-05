import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, stft
from sklearn.preprocessing import StandardScaler
import pywt

# Load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Extract trials and labels
def extract_trials(data, num_trials, trial_duration_samples, total_samples_per_trial):
    trials = []
    labels = []

    start_index = data[(data['Trigger'] == 1)].index[0]  # First trigger point
    for i in range(num_trials):
        trial_start = start_index + i * total_samples_per_trial
        trial_end = trial_start + trial_duration_samples
        trial_data = data.iloc[trial_start:trial_end]
        trials.append(trial_data.iloc[:, 4:].values)  # Exclude first 4 columns (Time, Trigger, Label, Planning)
        labels.append(trial_data['Label'].values[0])  # Label is consistent within a trial

    return np.array(trials), np.array(labels)

# Apply bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_data = sosfilt(sos, data, axis=0)
    return filtered_data

# Define transformations
def compute_stft(data, fs=250, nperseg=128, noverlap=64):
    f, t, Zxx = stft(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return np.abs(Zxx)

def compute_cwt(data, wavelet='morl', scales=np.arange(1, 31)):
    coeffs, freqs = pywt.cwt(data, scales, wavelet)
    return np.abs(coeffs)

def apply_transforms(trials, transform_funcs, target_shape):
    transformed_trials = []
    for trial in trials:
        trial_transformed = []
        for func in transform_funcs:
            transformed_data = [func(channel) for channel in trial.T]
            transformed_data = [np.pad(td, pad_width=((0, max(0, target_shape[0] - td.shape[0])), 
                                                      (0, max(0, target_shape[1] - td.shape[1]))), 
                                      mode='constant') for td in transformed_data]
            trial_transformed.append(np.stack(transformed_data, axis=0))
        transformed_trials.append(np.concatenate(trial_transformed, axis=0))
    return np.array(transformed_trials)

# Normalize data
def normalize_data(trials_transformed):
    scaler = StandardScaler()
    trials_reshaped = trials_transformed.reshape(-1, trials_transformed.shape[-1])
    trials_normalized = scaler.fit_transform(trials_reshaped).reshape(trials_transformed.shape)
    return trials_normalized
