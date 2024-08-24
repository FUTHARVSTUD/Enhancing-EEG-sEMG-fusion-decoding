import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import butter, sosfilt, stft
import pywt

# Define custom blocks
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

class TFConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TFConvUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.25)
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

class TemporalConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TemporalConvUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.25)
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
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        self.se = SEBlock(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.se(x)
        return x

class SimpleAMPCNet(nn.Module):
    def __init__(self, num_classes, input_channels, target_shape):
        super(SimpleAMPCNet, self).__init__()

        self.time_conv = TimeConvBlock(input_channels, 8)
        self.freq_conv = FreqConvBlock(input_channels, 8)
        self.tf_conv = TFConvBlock(input_channels, 8)
        self.hht_conv = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=(1, 1), padding=(0, 0)),  # Adjust as needed
            nn.ReLU()
        )

        # Calculate flatten size using dummy input
        dummy_input = torch.randn(1, input_channels, target_shape[0], target_shape[1])
        with torch.no_grad():
            self.flatten_size = self._get_flatten_size(dummy_input)

        # Initialize the classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _get_flatten_size(self, x):
        x_time = self.time_conv(x)
        x_freq = self.freq_conv(x)
        x_tf = self.tf_conv(x)
        x_hht = self.hht_conv(x)
        x_time_flat = x_time.view(x_time.size(0), -1)
        x_freq_flat = x_freq.view(x_freq.size(0), -1)
        x_tf_flat = x_tf.view(x_tf.size(0), -1)
        x_hht_flat = x_hht.view(x_hht.size(0), -1)
        return x_time_flat.shape[1] + x_freq_flat.shape[1] + x_tf_flat.shape[1] + x_hht_flat.shape[1]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_time = self.time_conv(x)
        x_freq = self.freq_conv(x)
        x_tf = self.tf_conv(x)
        x_hht = self.hht_conv(x)
        x_time_flat = x_time.view(x_time.size(0), -1)
        x_freq_flat = x_freq.view(x_freq.size(0), -1)
        x_tf_flat = x_tf.view(x_tf.size(0), -1)
        x_hht_flat = x_hht.view(x_hht.size(0), -1)
        concat_features = torch.cat([x_time_flat, x_freq_flat, x_tf_flat, x_hht_flat], dim=1)
        out = self.classifier(concat_features)
        return out

# Transformation functions
def compute_stft(data, fs=250, nperseg=128, noverlap=64):
    f, t, Zxx = stft(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return np.abs(Zxx)

def compute_cwt(data, wavelet='morl', scales=np.arange(1, 31)):
    coeffs, freqs = pywt.cwt(data, scales, wavelet)
    return np.abs(coeffs)

def compute_hht(data):
    # Placeholder for Hilbert-Huang Transform
    return data  # Replace with actual HHT implementation

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
            pca_n = min(pca_components, max_components)
            pca = PCA(n_components=pca_n)
            pca_transformed = pca.fit_transform(concatenated_transformed_reshaped)
            pca_transformed_reshaped = pca_transformed.T.reshape(pca_n, n_freq_bins, n_time_steps)
            transformed_trials.append(pca_transformed_reshaped)
        else:
            transformed_trials.append(concatenated_transformed)

    return np.array(transformed_trials)

# Filtering functions
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

def butter_notch(center_freq, fs, Q=30):
    w0 = center_freq / (fs / 2)
    sos = butter(2, [w0 - w0 / Q, w0 + w0 / Q], btype='bandstop', analog=False, output='sos')
    return sos

def apply_notch_filter(data, center_freq, fs, Q=30):
    sos = butter_notch(center_freq, fs, Q)
    filtered_data = sosfilt(sos, data, axis=0)
    return filtered_data

# Augmentation functions
def add_noise(data, noise_level=0.01):
    noise = np.random.randn(*data.shape) * noise_level
    return data + noise

def time_shift(data, shift_max=50):
    shift = np.random.randint(-shift_max, shift_max)
    return np.roll(data, shift, axis=0)

def scale(data, scale_min=0.8, scale_max=1.2):
    factor = np.random.uniform(scale_min, scale_max)
    return data * factor

def augment(data):
    data = add_noise(data)
    data = time_shift(data)
    data = scale(data)
    return data

# Data extraction function
def extract_trials(data, num_trials, trial_duration_samples, total_samples_per_trial, eeg_channels, semg_channels):
    trials_eeg = []
    trials_semg = []
    labels = []
    planning_duration_samples = 2 * sampling_rate  # 2 seconds planning duration

    # Find the first trigger index
    trigger_indices = data.index[data['Trigger'] == 1].tolist()
    if not trigger_indices:
        raise ValueError("No trigger found in the data.")
    start_index = trigger_indices[0]
    
    for i in range(num_trials):
        trial_start = start_index + i * total_samples_per_trial
        trial_end = trial_start + trial_duration_samples
        trial_data = data.iloc[trial_start:trial_end]

        
        planning_data = trial_data[trial_data['Planning'] == 1]
        if planning_data.empty:
            raise ValueError(f"No planning data found for trial {i}.")
        planning_start = planning_data.index[0]
        planning_end = planning_start + planning_duration_samples
        planning_data = trial_data.loc[planning_start:planning_end]

        # Separate EEG and sEMG channels
        eeg_data = trial_data.iloc[:, eeg_channels].values  
        semg_data = trial_data.iloc[:, semg_channels].values  

        trials_eeg.append(planning_data.iloc[:, eeg_channels].values)
        trials_semg.append(planning_data.iloc[:, semg_channels].values)
        labels.append(trial_data['Label'].values[0])

    return np.array(trials_eeg), np.array(trials_semg), np.array(labels)

# Custom Dataset class
class EEGEMGTransformedDataset(Dataset):
    def __init__(self, eeg_trials, semg_trials, labels, augment=False):
        self.eeg_trials = eeg_trials
        self.sem_trials = semg_trials
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.eeg_trials)

    def __getitem__(self, idx):
        eeg = self.eeg_trials[idx]
        semg = self.sem_trials[idx]
        label = self.labels[idx]
        if self.augment:
            eeg = augment(eeg)
            semg = augment(semg)
        return torch.tensor(eeg, dtype=torch.float32), torch.tensor(semg, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# DataLoader creation function
def create_dataloader(eeg_trials, semg_trials, labels, batch_size=9, shuffle=True, augment=False):
    dataset = EEGEMGTransformedDataset(eeg_trials, semg_trials, labels, augment=augment)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# Combined Model Definition
class CombinedModel(nn.Module):
    def __init__(self, num_classes, input_channels_eeg, input_channels_semg, target_shape):
        super(CombinedModel, self).__init__()
        self.eeg_model = SimpleAMPCNet(num_classes, input_channels_eeg, target_shape)
        self.sem_model = SimpleAMPCNet(num_classes, input_channels_semg, target_shape)
        # Combined classifier
        combined_flatten_size = self.eeg_model.flatten_size + self.sem_model.flatten_size
        self.classifier = nn.Sequential(
            nn.Linear(combined_flatten_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, eeg, semg):
        eeg_features = self.eeg_model(eeg)
        semg_features = self.sem_model(semg)
        # Concatenate features
        combined_features = torch.cat([eeg_features, semg_features], dim=1)
        out = self.classifier(combined_features)
        return out

# Training and Evaluation Function
def train_and_evaluate_model(train_loader, val_loader, model, criterion, optimizer, scheduler, num_epochs=200):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (eeg, semg, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(eeg, semg)
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
            for eeg, semg, labels in val_loader:
                outputs = model(eeg, semg)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {running_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Val Accuracy: {100 * correct / total:.2f}%')

# Main Function
def main():
    file_path = 'P36_EEG_OpenBCIEMG_IMUADS1220_EEG_EMG250Hz1.csv'
    sampling_rate = 250  
    trial_duration_sec = 8  
    break_duration_sec = 3  
    num_trials = 40
    n_splits = 5  
    
    trial_duration_samples = trial_duration_sec * sampling_rate
    break_duration_samples = break_duration_sec * sampling_rate
    total_samples_per_trial = trial_duration_samples + break_duration_samples
    
    # Load data
    data = pd.read_csv(file_path)

    # Define EEG and sEMG channel indices
    # Adjust these based on your CSV's actual channel layout
    n_eeg_channels = 32  # Example: first 32 channels are EEG
    n_semg_channels = 8  # Example: next 8 channels are sEMG
    eeg_channels = list(range(4, 4 + n_eeg_channels))  # Assuming first 4 columns are metadata
    semg_channels = list(range(4 + n_eeg_channels, 4 + n_eeg_channels + n_semg_channels))

    # Extract trials
    trials_eeg, trials_semg, labels = extract_trials(
        data, num_trials, trial_duration_samples, total_samples_per_trial, eeg_channels, semg_channels
    )
    labels = labels - 1  # Adjust labels if necessary

    # Apply notch filter
    notch_center_freq = 50.0  # Hz
    quality_factor = 30.0
    trials_notch_filtered_eeg = np.array([apply_notch_filter(trial, notch_center_freq, sampling_rate, quality_factor) for trial in trials_eeg])
    trials_notch_filtered_semg = np.array([apply_notch_filter(trial, notch_center_freq, sampling_rate, quality_factor) for trial in trials_semg])

    # Apply bandpass filter
    lowcut = 27.0
    highcut = 90.0
    trials_filtered_eeg = np.array([apply_bandpass_filter(trial, lowcut, highcut, sampling_rate) for trial in trials_notch_filtered_eeg])
    trials_filtered_semg = np.array([apply_bandpass_filter(trial, lowcut, highcut, sampling_rate) for trial in trials_notch_filtered_semg])

    # Compute target shape based on transformations
    sample_eeg_channel = trials_filtered_eeg[0, :, 0]
    sample_semg_channel = trials_filtered_semg[0, :, 0]
    stft_shape_eeg = compute_stft(sample_eeg_channel).shape
    cwt_shape_eeg = compute_cwt(sample_eeg_channel).shape
    stft_shape_semg = compute_stft(sample_semg_channel).shape
    cwt_shape_semg = compute_cwt(sample_semg_channel).shape

    target_shape_eeg = (max(stft_shape_eeg[0], cwt_shape_eeg[0]), max(stft_shape_eeg[1], cwt_shape_eeg[1]))
    target_shape_semg = (max(stft_shape_semg[0], cwt_shape_semg[0]), max(stft_shape_semg[1], cwt_shape_semg[1]))

    # Apply transformations
    transform_funcs = [compute_stft, compute_cwt, compute_hht]
    pca_components = 20
    trials_transformed_eeg = apply_transforms(trials_filtered_eeg, transform_funcs, target_shape_eeg, pca_components=pca_components)
    trials_transformed_semg = apply_transforms(trials_filtered_semg, transform_funcs, target_shape_semg, pca_components=pca_components)

    # Normalize data
    scaler_eeg = StandardScaler()
    scaler_semg = StandardScaler()
    trials_normalized_eeg = []
    trials_normalized_semg = []
    for trial_eeg, trial_semg in zip(trials_transformed_eeg, trials_transformed_semg):
        # Normalize EEG
        n_channels_eeg, n_freq_bins_eeg, n_time_steps_eeg = trial_eeg.shape
        trial_eeg_reshaped = trial_eeg.reshape(n_channels_eeg, -1).T
        trial_eeg_normalized = scaler_eeg.fit_transform(trial_eeg_reshaped).T
        trial_eeg_normalized = trial_eeg_normalized.reshape(n_channels_eeg, n_freq_bins_eeg, n_time_steps_eeg)
        trials_normalized_eeg.append(trial_eeg_normalized)

        # Normalize sEMG
        n_channels_semg, n_freq_bins_semg, n_time_steps_semg = trial_semg.shape
        trial_semg_reshaped = trial_semg.reshape(n_channels_semg, -1).T
        trial_semg_normalized = scaler_semg.fit_transform(trial_semg_reshaped).T
        trial_semg_normalized = trial_semg_normalized.reshape(n_channels_semg, n_freq_bins_semg, n_time_steps_semg)
        trials_normalized_semg.append(trial_semg_normalized)

    trials_normalized_eeg = np.array(trials_normalized_eeg)
    trials_normalized_semg = np.array(trials_normalized_semg)

    # Dimension checks
    print(f'EEG trials shape: {trials_normalized_eeg.shape}')
    print(f'sEMG trials shape: {trials_normalized_semg.shape}')
    if target_shape_eeg != target_shape_semg:
        raise ValueError("EEG and sEMG transformed data have different target shapes. Please ensure they match.")

    target_shape = target_shape_eeg  # Use one target_shape since they match

    # Create datasets and dataloaders with cross-validation
    skf = StratifiedKFold(n_splits=n_splits)
    fold = 1
    for train_index, val_index in skf.split(trials_normalized_eeg, labels):
        print(f'Fold {fold}/{n_splits}')
        X_train_eeg, X_val_eeg = trials_normalized_eeg[train_index], trials_normalized_eeg[val_index]
        X_train_semg, X_val_semg = trials_normalized_semg[train_index], trials_normalized_semg[val_index]
        y_train, y_val = labels[train_index], labels[val_index]

        # Balance the training set
        class_0_indices = np.where(y_train == 0)[0]
        class_1_indices = np.where(y_train == 1)[0]
        if len(class_1_indices) >= len(class_0_indices):
            selected_class_1 = np.random.choice(class_1_indices, len(class_0_indices), replace=False)
            balanced_indices = np.hstack([class_0_indices, selected_class_1])
        else:
            selected_class_0 = np.random.choice(class_0_indices, len(class_1_indices), replace=False)
            balanced_indices = np.hstack([selected_class_0, class_1_indices])
        np.random.shuffle(balanced_indices)
        X_train_balanced_eeg = X_train_eeg[balanced_indices]
        X_train_balanced_semg = X_train_semg[balanced_indices]
        y_train_balanced = y_train[balanced_indices]

        # Create DataLoaders
        train_loader = create_dataloader(X_train_balanced_eeg, X_train_balanced_semg, y_train_balanced, augment=True)
        val_loader = create_dataloader(X_val_eeg, X_val_semg, y_val, shuffle=False)

        # Define model
        input_channels_eeg = X_train_balanced_eeg.shape[1]
        input_channels_semg = X_train_balanced_semg.shape[1]
        num_classes = len(np.unique(labels))
        model = CombinedModel(num_classes=num_classes, input_channels_eeg=input_channels_eeg, input_channels_semg=input_channels_semg, target_shape=target_shape)

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0007)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

        # Train and evaluate
        train_and_evaluate_model(train_loader, val_loader, model, criterion, optimizer, scheduler)

        fold += 1

    # Save the final model
    torch.save(model.state_dict(), 'combined_model_random.pth')

if __name__ == "__main__":
    main()
