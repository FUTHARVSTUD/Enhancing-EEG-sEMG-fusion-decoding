import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from scipy.signal import butter, sosfilt, stft, cwt, ricker
import pywt
from scipy.signal import hilbert
from sklearn.decomposition import PCA

# Define the notch filter
def butter_notch(center_freq, fs, quality_factor=30.0):
    nyquist = 0.5 * fs
    low = center_freq / nyquist - center_freq / (2 * quality_factor * nyquist)
    high = center_freq / nyquist + center_freq / (2 * quality_factor * nyquist)
    sos = butter(2, [low, high], btype='bandstop', output='sos')
    return sos

def apply_notch_filter(data, center_freq, fs, quality_factor=30.0):
    sos = butter_notch(center_freq, fs, quality_factor)
    filtered_data = sosfilt(sos, data, axis=0)
    return filtered_data

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], btype='bandpass', output='sos')
    return sos

def apply_bandpass_filter(data, lowcut, highcut, fs):
    sos = butter_bandpass(lowcut, highcut, fs)
    filtered_data = sosfilt(sos, data, axis=0)
    return filtered_data

def compute_stft(data):
    f, t, Zxx = stft(data, fs=250, nperseg=128, noverlap=64)
    return np.abs(Zxx)

def compute_cwt(data):
    scales = np.arange(1, 31)
    coeffs, freqs = pywt.cwt(data, scales, wavelet='morl')
    return np.abs(coeffs)

def compute_hht(data):
    emd = EMD()
    imfs = emd(data)
    
    hht_result = []
    for imf in imfs:
        analytic_signal = hilbert(imf)
        amplitude_envelope = np.abs(analytic_signal)
        hht_result.append(amplitude_envelope)
    
    return np.array(hht_result)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class PSABlock(nn.Module):
    def __init__(self, in_channels):
        super(PSABlock, self).__init__()
        self.group_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=4)
        self.conv_list = nn.ModuleList([
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1),
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=5, padding=2),
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=7, padding=3),
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=9, padding=4)
        ])
        self.se_blocks = nn.ModuleList([SEBlock(in_channels // 4) for _ in range(4)])

    def forward(self, x):
        x = self.group_conv(x)
        x_split = torch.split(x, x.size(1) // 4, dim=1)
        outputs = []
        for x_i, conv_i, se_i in zip(x_split, self.conv_list, self.se_blocks):
            att = se_i(conv_i(x_i))
            outputs.append(x_i * att)
        return torch.cat(outputs, dim=1)

class TemporalConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TemporalConvUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x

class TimeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TimeConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 10), padding=(0, 5))
        self.unit1 = TemporalConvUnit(out_channels, out_channels)
        self.unit2 = TemporalConvUnit(out_channels, out_channels)
        self.se = SEBlock(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.unit1(x)
        x = self.unit2(x)
        x = self.se(x)
        return x

class FreqConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FreqConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(10, 1), stride=(2, 1), padding=(5, 0))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.se = SEBlock(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.se(x)
        return x

class TFConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TFConvUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x

class TFConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TFConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(5, 5), padding=(2, 2))
        self.unit1 = TFConvUnit(out_channels, out_channels)
        self.unit2 = TFConvUnit(out_channels, out_channels)
        self.psa = PSABlock(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.unit1(x)
        x = self.unit2(x)
        x = self.psa(x)
        return x

class SimpleAMPCNet(nn.Module):
    def __init__(self, num_classes, input_channels, target_shape):
        super(SimpleAMPCNet, self).__init__()

        # Time convolutional block
        self.time_conv = nn.Sequential(
            nn.Conv2d(input_channels, 4, kernel_size=(1, 10), padding=(0, 5)),  # Reduced channels
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.MaxPool2d((1, 2))
        )

        # Frequency convolutional block
        self.freq_conv = nn.Sequential(
            nn.Conv2d(input_channels, 4, kernel_size=(10, 1), stride=(2, 1), padding=(5, 0)),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )

        # Time-Frequency convolutional block
        self.tf_conv = nn.Sequential(
            nn.Conv2d(input_channels, 4, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        # HHT convolutional block (optional based on HHT features)
        self.hht_conv = nn.Sequential(
            nn.Conv2d(input_channels, 4, kernel_size=(1, 1)),  # Adjusted for HHT
            nn.ReLU()
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(target_shape[0] * target_shape[1] * 12, 128),  # Adjusted for concatenated features
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        time_features = self.time_conv(x)
        freq_features = self.freq_conv(x)
        tf_features = self.tf_conv(x)
        hht_features = self.hht_conv(x)

        # Flatten and concatenate features
        time_features = time_features.view(time_features.size(0), -1)
        freq_features = freq_features.view(freq_features.size(0), -1)
        tf_features = tf_features.view(tf_features.size(0), -1)
        hht_features = hht_features.view(hht_features.size(0), -1)

        concatenated_features = torch.cat((time_features, freq_features, tf_features, hht_features), dim=1)

        # Fully connected layers
        out = self.fc(concatenated_features)
        return out

class EEGEMGTransformedDataset(Dataset):
    def __init__(self, trials, labels, augment=False):
        self.trials = trials
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        trial = self.trials[idx]
        label = self.labels[idx]
        if self.augment:
            trial = augment(trial)
        return torch.tensor(trial, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def create_dataloader(trials, labels, batch_size=9, shuffle=True, augment=False):
    dataset = EEGEMGTransformedDataset(trials, labels, augment=augment)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
def apply_transforms(trials, transform_funcs, target_shape, pca_components=None):
    transformed_trials = []
    for trial in trials:
        trial_transformed = []
        for func in transform_funcs:
            if func == compute_hht:
                transformed_data = [compute_hht(channel) for channel in trial.T]
            else:
                transformed_data = [func(channel) for channel in trial.T]
            transformed_data = [np.pad(td, pad_width=((0, max(0, target_shape[0] - td.shape[0])), 
                                                      (0, max(0, target_shape[1] - td.shape[1]))), 
                                      mode='constant') for td in transformed_data]
            trial_transformed.append(np.stack(transformed_data, axis=0))
        
        # Concatenate all transformed features
        concatenated_transformed = np.concatenate(trial_transformed, axis=0)

        # Reshape for PCA
        n_channels, n_freq_bins, n_time_steps = concatenated_transformed.shape
        concatenated_transformed_reshaped = concatenated_transformed.reshape(n_channels, -1).T  # Shape (samples, channels)

        max_components = min(concatenated_transformed_reshaped.shape)
        if pca_components:
            pca_components = min(pca_components, max_components)  # Adjust n_components
            pca = PCA(n_components=pca_components)
            pca_transformed = pca.fit_transform(concatenated_transformed_reshaped)
            pca_transformed_reshaped = pca_transformed.T.reshape(pca_components, n_freq_bins, n_time_steps)
            transformed_trials.append(pca_transformed_reshaped)
        else:
            transformed_trials.append(concatenated_transformed)

    return np.array(transformed_trials)

def preprocess_data(file_path, num_trials, trial_duration_samples, total_samples_per_trial, sampling_rate):
    data = pd.read_csv(file_path)

    eeg_columns = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']
    emg_columns = ['EMG_Right', 'EMG_Left']

    eeg_trials, labels = extract_trials(data[eeg_columns + ['Trigger', 'Label', 'Planning']], num_trials, trial_duration_samples, total_samples_per_trial)
    emg_trials, _ = extract_trials(data[emg_columns + ['Trigger', 'Label', 'Planning']], num_trials, trial_duration_samples, total_samples_per_trial)

    # Apply notch filter before the bandpass filter
    notch_center_freq = 50.0  # Powerline frequency to be removed
    quality_factor = 30.0
    eeg_trials_notch_filtered = np.array([apply_notch_filter(trial, notch_center_freq, sampling_rate, quality_factor) for trial in eeg_trials])
    emg_trials_notch_filtered = np.array([apply_notch_filter(trial, notch_center_freq, sampling_rate, quality_factor) for trial in emg_trials])

    # Apply bandpass filter
    lowcut = 27.0
    highcut = 90.0
    eeg_trials_filtered = np.array([apply_bandpass_filter(trial, lowcut, highcut, sampling_rate) for trial in eeg_trials_notch_filtered])
    emg_trials_filtered = np.array([apply_bandpass_filter(trial, lowcut, highcut, sampling_rate) for trial in emg_trials_notch_filtered])

    sample_channel = eeg_trials_filtered[0, :, 0]
    stft_shape = compute_stft(sample_channel).shape
    cwt_shape = compute_cwt(sample_channel).shape

    target_shape = (max(stft_shape[0], cwt_shape[0]), max(stft_shape[1], cwt_shape[1]))

    transform_funcs = [compute_stft, compute_cwt]
    pca_components = 20
    eeg_trials_transformed = apply_transforms(eeg_trials_filtered, transform_funcs, target_shape, pca_components=pca_components)
    emg_trials_transformed = apply_transforms(emg_trials_filtered, transform_funcs, target_shape, pca_components=pca_components)

    scaler = StandardScaler()

    eeg_trials_normalized = []
    emg_trials_normalized = []

    for trial in eeg_trials_transformed:
        n_channels, n_freq_bins, n_time_steps = trial.shape
        trial_reshaped = trial.reshape(n_channels, -1).T
        trial_normalized = scaler.fit_transform(trial_reshaped).T  # Transpose back to original shape (n_channels, n_samples)
        trial_normalized = trial_normalized.reshape(n_channels, n_freq_bins, n_time_steps)  # Reshape back to original shape
        eeg_trials_normalized.append(trial_normalized)

    for trial in emg_trials_transformed:
        n_channels, n_freq_bins, n_time_steps = trial.shape
        trial_reshaped = trial.reshape(n_channels, -1).T
        trial_normalized = scaler.fit_transform(trial_reshaped).T  # Transpose back to original shape (n_channels, n_samples)
        trial_normalized = trial_normalized.reshape(n_channels, n_freq_bins, n_time_steps)  # Reshape back to original shape
        emg_trials_normalized.append(trial_normalized)

    eeg_trials_normalized = np.array(eeg_trials_normalized)
    emg_trials_normalized = np.array(emg_trials_normalized)

    # Fuse EEG and sEMG data
    fused_trials = np.concatenate((eeg_trials_normalized, emg_trials_normalized), axis=1)

    return fused_trials, labels, target_shape

def train_and_evaluate_model(train_loader, val_loader, model, criterion, optimizer, scheduler, num_epochs=200):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {running_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Val Accuracy: {100 * correct / total:.2f}%')
def extract_trials(data, num_trials, trial_duration_samples, total_samples_per_trial):
    trials = []
    labels = []

    for i in range(num_trials):
        # Assuming 'Trigger' column marks the start of each trial
        start_idx = data[data['Trigger'] == i].index[0]
        
        # Define the end of the trial
        end_idx = start_idx + trial_duration_samples
        
        if end_idx < len(data):
            trial_data = data.iloc[start_idx:end_idx].values
            trial_label = data.iloc[start_idx]['Label']
            trials.append(trial_data)
            labels.append(trial_label)
    
    # Convert lists to numpy arrays
    trials = np.array(trials)
    labels = np.array(labels)
    
    return trials, labels
# Load and preprocess data
file_path = 'P36_EEG_OpenBCIEMG_IMUADS1220_EEG_EMG250Hz1.csv'  # Replace with your data file path
data = pd.read_csv(file_path)
print(data.head())
print(data.columns)
print(data['Trigger'].unique())
print(data['Label'].unique())
num_trials = 40  # Number of trials
trial_duration_samples = 2000  # Duration of each trial in samples
total_samples_per_trial = 2750  # Total number of samples per trial
sampling_rate = 250  # Sampling rate in Hz
trials_normalized, labels, target_shape = preprocess_data(file_path, num_trials, trial_duration_samples, total_samples_per_trial, sampling_rate)


# Cross-validation setup
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits)
fold = 1
for train_index, val_index in skf.split(trials_normalized, labels):
    print(f'Fold {fold}/{n_splits}')
    X_train, X_val = trials_normalized[train_index], trials_normalized[val_index]
    y_train, y_val = labels[train_index], labels[val_index]
    class_0_indices = np.where(y_train == 0)[0]
    class_1_indices = np.where(y_train == 1)[0]
    balanced_indices = np.hstack([class_0_indices, np.random.choice(class_1_indices, len(class_0_indices), replace=False)])
    np.random.shuffle(balanced_indices)
    X_train_balanced = X_train[balanced_indices]
    y_train_balanced = y_train[balanced_indices]

    train_loader = create_dataloader(X_train_balanced, y_train_balanced, augment=True)
    val_loader = create_dataloader(X_val, y_val, shuffle=True)

    simple_model = SimpleAMPCNet(num_classes=2, input_channels=X_train.shape[1], target_shape=target_shape)
    optimizer = optim.Adam(simple_model.parameters(), lr=0.0007)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    train_and_evaluate_model(train_loader, val_loader, simple_model, nn.CrossEntropyLoss(), optimizer, scheduler)

    fold += 1

torch.save(simple_model.state_dict(), 'simple_model_random.pth')
