# import numpy as np
# from scipy import signal
# from scipy.signal import stft, cwt, morlet2
# import mne
# import os
# from scipy.special import factorial
# import pywt
# from ssqueezepy import ssq_cwt

# # def cgau8(t, s):
# #     print("Inside cgau8:")
# #     print(type(t), t.shape)
# #     print(type(s), s.shape)
# #     prefactor = (-1)**4 * (s ** 8) / np.sqrt(np.math.factorial(8))
# #     exponent = - (t**2) / (2 * s**2)
# #     derivative = np.polynomial.hermite.hermval(8, t / s)

# #     return prefactor * derivative * np.exp(exponent)

# def stockwell_transform(x, fs, nperseg):
#     freqs = np.fft.fftfreq(nperseg, d=1/fs)
#     scales = np.arange(1, int(len(x)/2))
 
#     stft_x = stft(x, fs=fs, window='hann', nperseg=nperseg)[-1]

#     Sx = np.zeros((nperseg, len(scales)), dtype=complex)

#     for i, scale in enumerate(scales):
#         if (i + 1) * nperseg > stft_x.shape[1]:
#             stft_x_segment = stft_x[:, i * nperseg :]
#         else:
#             stft_x_segment = stft_x[:, i * nperseg : (i + 1) * nperseg]
#             window = np.exp(-(scale * freqs / fs)**2 / 2)
#             window = np.tile(window, (stft_x.shape[0], 1))
#             Sx[:, i] = np.mean(np.fft.ifft(stft_x_segment * window), axis=0)

#     return Sx

# def load_trial_data(subj_id, trial_id, data_dir="dataset"):

#     eeg_file_path = os.path.join(data_dir, 'EEG', f"{subj_id}_{trial_id}.set")
#     semg_file_path = os.path.join(data_dir, 'sEMG', f"{subj_id}_{trial_id}.txt")

#     eeg_data = mne.io.read_raw_eeglab(eeg_file_path).get_data()
#     eeg_data = eeg_data.astype(float)  
#     semg_data = np.loadtxt(semg_file_path, skiprows=4)[:, :16] 

#     return eeg_data, semg_data

# def preprocess_and_extract_features(eeg_data, semg_data):
#     all_features = []
#     widths = np.arange(1, 128)

#     for channel_idx in range(eeg_data.shape[0]):
#         eeg_channel_data = eeg_data[channel_idx, :]
#         morlet = pywt.ContinuousWavelet('morl')
#         print(morlet.center_frequency)

#         eeg_stft_out = stft(eeg_channel_data, fs=500, window='hann', nperseg=128, noverlap=128//2)
#         eeg_stft = eeg_stft_out[0]
#         stft_shape = eeg_stft.shape
#         #eeg_cwt_result = pywt.cwt(eeg_channel_data, scales=widths, wavelet=morlet, method='conv')
#         Tx, Wx, freqs, scales = ssq_cwt(eeg_channel_data, fs=500, wavelet=('morlet', {'mu': 5}), scales='log', nv=32, squeezing='sum')  
#         eeg_cwt = Tx[0]
#         print("eeg_cwt shape:", eeg_cwt.shape)
#         eeg_stockwell = stockwell_transform(eeg_channel_data, fs=500, nperseg=128)

#         eeg_stft_filtered = filter_spectrogram(eeg_stft, freq_range=(8, 30), stft_shape=stft_shape, fs=500)
#         eeg_cwt_filtered = filter_spectrogram(eeg_cwt, freq_range=(8, 30), stft_shape=stft_shape, wavelet=morlet, fs=500)
#         eeg_stockwell_filtered = filter_spectrogram(eeg_stockwell, freq_range=(8, 30), stft_shape=stft_shape)

#         channel_features = np.concatenate([eeg_stft_filtered, eeg_cwt_filtered, eeg_stockwell_filtered], axis=0)
#         all_features.append(channel_features)

#     for channel_idx in range(semg_data.shape[0]):
#         semg_stft = stft(semg_data, fs=1000, window='hann', nperseg=128, noverlap=128//2) 
#         semg_cwt = cwt(semg_data, wavelet=morlet, scales=125) 
#         semg_stockwell = stockwell_transform(semg_data, fs=1000, nperseg=128)

#         semg_stft_filtered = filter_spectrogram(semg_stft, freq_range=(10, 200))
#         semg_cwt_filtered = filter_spectrogram(semg_cwt, freq_range=(10, 200))
#         semg_stockwell_filtered = filter_spectrogram(semg_stockwell, freq_range=(10, 200))

#         channel_features = np.concatenate([semg_stft_filtered, semg_cwt_filtered, semg_stockwell_filtered], axis=0)
#         all_features.append(channel_features)

#     # fused_spectrogram = np.concatenate([eeg_stft_filtered[2], semg_stft_filtered[2], eeg_cwt_filtered[2],  semg_cwt_filtered[2], eeg_stockwell_filtered[2], semg_stockwell_filtered[2]], axis=0) 
#     fused_spectrogram = np.concatenate(all_features, axis=0)

#     return fused_spectrogram

# def filter_spectrogram(spectrogram, freq_range, stft_shape, wavelet=None, fs=None):
#     freq_bins = calculate_freq_bins(spectrogram.shape, fs=fs, wavelet=wavelet)
#     mask = (freq_bins >= freq_range[0]) & (freq_bins <= freq_range[1])
#     filtered_spectrogram = spectrogram[mask]
#     return filtered_spectrogram

# def calculate_cwt_frequencies(scales, wavelet, fs):
#     if wavelet.center_frequency is None:
#         lowest_freq = 1.0 / scales[-1]
#     else:
#         lowest_freq = wavelet.center_frequency / fs
#     lowest_freq = wavelet.center_frequency / fs 
#     scale_factor = wavelet.center_frequency / lowest_freq  
#     return lowest_freq * scale_factor / scales

# def calculate_freq_bins(spectrogram_shape, fs=None, wavelet=None):
#     if wavelet is not None:
#       return calculate_cwt_frequencies(spectrogram_shape[0], wavelet, fs)
#     else:  
#         if fs is None:
#             raise ValueError("fs must be provided for STFT frequency calculation")
#         n_bins = spectrogram_shape[0] 
#         return np.fft.fftfreq(n_bins, d=1/fs)
#       #np.fft.fftfreq(stft_shape[0], )


import numpy as np
from scipy import signal
from scipy.signal import stft
import mne
import os
from scipy.special import factorial
import pywt
from ssqueezepy import ssq_cwt

# def stockwell_transform(x, fs, nperseg):
#     freqs = np.fft.fftfreq(nperseg, d=1/fs)
#     scales = np.arange(1, int(len(x)/2))
#     stft_x = stft(x, fs=fs, window='hann', nperseg=nperseg)[-1]
#     Sx = np.zeros((nperseg, len(scales)), dtype=complex)
#     for i, scale in enumerate(scales):
#         window = np.exp(-(scale * freqs / fs)**2 / 2)
#         window = np.tile(window, (stft_x.shape[0], 1))
#         Sx[:, i] = np.mean(np.fft.ifft(stft_x * window, axis=1), axis=0)
#     return Sx

def stockwell_transform(x, fs, nperseg, window_type='hann'):
    freqs = np.fft.fftfreq(nperseg, d=1/fs)
    scales = np.arange(1, int(len(x)/2))
    stft_x = stft(x, fs=fs, window=window_type, nperseg=nperseg)[-1]
    Sx = np.zeros((nperseg, len(scales)), dtype=complex)
    stft_x_truncated = stft_x[:, :128]
    for i, scale in enumerate(scales):
        window = np.exp(-(scale * freqs / fs)**2 / 2)
        window = np.tile(window, (stft_x.shape[0], 1))
        Sx[:, i] = np.mean(np.fft.ifft(stft_x_truncated * window, axis=1), axis=0)
    return Sx

def load_trial_data(subj_id, trial_id, data_dir="dataset"):
    eeg_file_path = os.path.join(data_dir, 'EEG', f"{subj_id}_{trial_id}.set")
    semg_file_path = os.path.join(data_dir, 'sEMG', f"{subj_id}_{trial_id}.txt")
    eeg_data = mne.io.read_raw_eeglab(eeg_file_path).get_data()
    eeg_data = eeg_data.astype(float)
    semg_data = np.loadtxt(semg_file_path, skiprows=4)[:, :16]
    return eeg_data, semg_data

def preprocess_and_extract_features(eeg_data, semg_data):
    all_features = []
    widths = np.arange(1, 128)
    fs_eeg = 500
    fs_semg = 1000
    for channel_idx in range(eeg_data.shape[0]):
        eeg_channel_data = eeg_data[channel_idx, :]
        
        # STFT
        _, _, eeg_stft = stft(eeg_channel_data, fs=fs_eeg, window='hann', nperseg=128, noverlap=128//2)
        stft_shape = eeg_stft.shape
        
        # CWT with ssq_cwt
        _, _, freqs, scales = ssq_cwt(eeg_channel_data, fs=fs_eeg, wavelet='morlet', scales='log', nv=32, squeezing='sum')
        eeg_cwt = np.abs(ssq_cwt(eeg_channel_data, fs=fs_eeg, wavelet='morlet', scales=scales, nv=32, squeezing='sum')[0])
        
        # Stockwell Transform
        eeg_stockwell = np.abs(stockwell_transform(eeg_channel_data, fs=fs_eeg, nperseg=128))

        # Filter Spectrograms (Adjust freq_range as needed)
        eeg_stft_filtered = filter_spectrogram(eeg_stft, freq_range=(8, 30), stft_shape=stft_shape, fs=fs_eeg)
        eeg_cwt_filtered = filter_spectrogram(eeg_cwt, freq_range=(8, 30), stft_shape=stft_shape, wavelet=pywt.ContinuousWavelet('morl'), fs=fs_eeg)
        eeg_stockwell_filtered = filter_spectrogram(eeg_stockwell, freq_range=(8, 30), stft_shape=stft_shape, fs=fs_eeg)

        channel_features = np.concatenate([eeg_stft_filtered, eeg_cwt_filtered, eeg_stockwell_filtered], axis=0)
        all_features.append(channel_features)

    for channel_idx in range(semg_data.shape[1]):  # Iterate over sEMG channels
        semg_channel_data = semg_data[:, channel_idx]

        # STFT
        _, _, semg_stft = stft(semg_channel_data, fs=fs_semg, window='hann', nperseg=128, noverlap=128//2)
        stft_shape = semg_stft.shape

        # CWT with PyWavelets
        coef, freqs = pywt.cwt(semg_channel_data, scales=widths, wavelet='morl')
        semg_cwt = np.abs(coef) 
        
        # Stockwell Transform
        semg_stockwell = np.abs(stockwell_transform(semg_channel_data, fs=fs_semg, nperseg=128))

        # Filter Spectrograms (Adjust freq_range as needed)
        semg_stft_filtered = filter_spectrogram(semg_stft, freq_range=(10, 200), stft_shape=stft_shape, fs=fs_semg)
        semg_cwt_filtered = filter_spectrogram(semg_cwt, freq_range=(10, 200), stft_shape=stft_shape, wavelet=pywt.ContinuousWavelet('morl'), fs=fs_semg)
        semg_stockwell_filtered = filter_spectrogram(semg_stockwell, freq_range=(10, 200), stft_shape=stft_shape, fs=fs_semg)

        channel_features = np.concatenate([semg_stft_filtered, semg_cwt_filtered, semg_stockwell_filtered], axis=0)
        all_features.append(channel_features)

    fused_spectrogram = np.concatenate(all_features, axis=0)
    return fused_spectrogram
# def preprocess_and_extract_features(eeg_data, semg_data):
#     all_features = []
#     widths = np.arange(1, 128)
#     for channel_idx in range(eeg_data.shape[0]):
#         eeg_channel_data = eeg_data[channel_idx, :]
        
#         STFT
#         eeg_stft_out = stft(eeg_channel_data, fs=500, window='hann', nperseg=128, noverlap=128//2)
#         eeg_stft = eeg_stft_out[2]  # Take the spectrogram
#         stft_shape = eeg_stft.shape
        
#         CWT
#         morlet = pywt.ContinuousWavelet('morl')
#         Tx, Wx, freqs, scales = ssq_cwt(eeg_channel_data, fs=500, wavelet=morlet, scales='log', nv=32, squeezing='sum')
#         Tx, Wx, freqs, scales = ssq_cwt(eeg_channel_data, fs=500, wavelet='morlet', scales='log', nv=32, squeezing='sum')
#         eeg_cwt = np.abs(Tx[0])  # Take the magnitude
        
#         Stockwell Transform
#         eeg_stockwell = np.abs(stockwell_transform(eeg_channel_data, fs=500, nperseg=128))  # Take the magnitude

#         Filter Spectrograms (Adjust freq_range as needed)
#         eeg_stft_filtered = filter_spectrogram(eeg_stft, freq_range=(8, 30), stft_shape=stft_shape, fs=500)
#         eeg_cwt_filtered = filter_spectrogram(eeg_cwt, freq_range=(8, 30), stft_shape=stft_shape, wavelet=morlet, fs=500)
#         eeg_stockwell_filtered = filter_spectrogram(eeg_stockwell, freq_range=(8, 30), stft_shape=stft_shape, fs=500)

#         channel_features = np.concatenate([eeg_stft_filtered, eeg_cwt_filtered, eeg_stockwell_filtered], axis=0)
#         all_features.append(channel_features)

#     for channel_idx in range(semg_data.shape[1]):  # Iterate over sEMG channels
#         semg_channel_data = semg_data[:, channel_idx]  # Extract sEMG channel data

#         STFT
#         semg_stft_out = stft(semg_channel_data, fs=1000, window='hann', nperseg=128, noverlap=128//2)
#         semg_stft = semg_stft_out[2] # Take the spectrogram
#         stft_shape = semg_stft.shape

#         CWT
#         semg_cwt_result = pywt.cwt(semg_channel_data, scales=widths, wavelet=morlet, method='conv')
#         semg_cwt = np.abs(semg_cwt_result[0]) # Take the magnitude

#         Stockwell Transform
#         semg_stockwell = np.abs(stockwell_transform(semg_channel_data, fs=1000, nperseg=128)) # Take the magnitude

#         Filter Spectrograms (Adjust freq_range as needed)
#         semg_stft_filtered = filter_spectrogram(semg_stft, freq_range=(10, 200), stft_shape=stft_shape, fs=1000)
#         semg_cwt_filtered = filter_spectrogram(semg_cwt, freq_range=(10, 200), stft_shape=stft_shape, wavelet=morlet, fs=1000)
#         semg_stockwell_filtered = filter_spectrogram(semg_stockwell, freq_range=(10, 200), stft_shape=stft_shape, fs=1000)

#         channel_features = np.concatenate([semg_stft_filtered, semg_cwt_filtered, semg_stockwell_filtered], axis=0)
#         all_features.append(channel_features)

#     fused_spectrogram = np.concatenate(all_features, axis=0)
#     return fused_spectrogram


def filter_spectrogram(spectrogram, freq_range, stft_shape, wavelet=None, fs=None):
    freq_bins = calculate_freq_bins(spectrogram.shape, fs=fs, wavelet=wavelet)
    mask = (freq_bins >= freq_range[0]) & (freq_bins <= freq_range[1])
    filtered_spectrogram = spectrogram[mask]
    return filtered_spectrogram

def calculate_cwt_frequencies(scales, wavelet, fs):
    print("wavelet center frequency:", wavelet.center_frequency)
    lowest_freq = wavelet.center_frequency / fs
    scale_factor = wavelet.center_frequency / lowest_freq
    return lowest_freq * scale_factor / scales

# def calculate_freq_bins(spectrogram_shape, fs=None, wavelet=None):
#     if wavelet is not None:
#         return calculate_cwt_frequencies(spectrogram_shape[0], wavelet, fs)
#     else:
#         if fs is None:
#             raise ValueError("fs must be provided for STFT frequency calculation")
#         n_bins = spectrogram_shape[0]
#         return np.fft.fftfreq(n_bins, d=1/fs)
def calculate_freq_bins(spectrogram_shape, fs=None, wavelet=None):
    if wavelet is not None:
        # Ensure we're working with a ContinuousWavelet object
        if not isinstance(wavelet, pywt.ContinuousWavelet):
            raise ValueError("wavelet must be a ContinuousWavelet object for CWT frequency calculation")
        return calculate_cwt_frequencies(spectrogram_shape[0], wavelet, fs)
    else:
        if fs is None:
            raise ValueError("fs must be provided for STFT frequency calculation")
        n_bins = spectrogram_shape[0]
        return np.fft.fftfreq(n_bins, d=1/fs)
