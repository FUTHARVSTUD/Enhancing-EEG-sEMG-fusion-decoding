import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from scipy.signal import butter, sosfilt, stft
import pywt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from PyEMD import EMD
from scipy.signal import hilbert

def compute_hht(data):
    emd = EMD()
    imfs = emd(data)
    
    hht_result = []
    for imf in imfs:
        analytic_signal = hilbert(imf)
        amplitude_envelope = np.abs(analytic_signal)
        hht_result.append(amplitude_envelope)
    
    return np.array(hht_result)

# Define necessary classes and functions
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

class SimpleAMPCNet(nn.Module):
    def __init__(self, num_classes, input_channels, target_shape):
        super(SimpleAMPCNet, self).__init__()

        self.time_conv = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=(1, 10), padding=(0, 5)),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU()
        )

        self.freq_conv = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=(10, 1), stride=(2, 1), padding=(5, 0)),
            nn.ReLU()
        )

        self.tf_conv = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU()
        )

        self.hht_conv = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=(1, 1), padding=(0, 0)),  # Example, adjust according to HHT output
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
        x_hht = self.hht_conv(x)  # Add HHT output
        x_time_flat = x_time.view(x_time.size(0), -1)
        x_freq_flat = x_freq.view(x_freq.size(0), -1)
        x_tf_flat = x_tf.view(x_tf.size(0), -1)
        x_hht_flat = x_hht.view(x_hht.size(0), -1)  # Flatten HHT output
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
        x_hht = self.hht_conv(x)  # Apply HHT convolution
        x_time_flat = x_time.view(x_time.size(0), -1)
        x_freq_flat = x_freq.view(x_freq.size(0), -1)
        x_tf_flat = x_tf.view(x_tf.size(0), -1)
        x_hht_flat = x_hht.view(x_hht.size(0), -1)  # Flatten HHT output
        concat_features = torch.cat([x_time_flat, x_freq_flat, x_tf_flat, x_hht_flat], dim=1)
        out = self.classifier(concat_features)
        return out

# Define data transformation functions

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
            if func == compute_hht:
                transformed_data = [compute_hht(channel) for channel in trial.T]
            else:
                transformed_data = [func(channel) for channel in trial.T]
            transformed_data = [np.pad(td, pad_width=((0, max(0, target_shape[0] - td.shape[0])), 
                                                      (0, max(0, target_shape[1] - td.shape[1]))), 
                                      mode='constant') for td in transformed_data]
            trial_transformed.append(np.stack(transformed_data, axis=0))
        transformed_trials.append(np.concatenate(trial_transformed, axis=0))
    return np.array(transformed_trials)


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

# # Load the data
# file_path = 'P36_EEG_EMG250Hz.csv'
# data = pd.read_csv(file_path)

# # Define constants
# sampling_rate = 250  # 250 Hz
# trial_duration_sec = 8  # 8 seconds
# break_duration_sec = 3  # 3 seconds
# num_trials = 40
# n_splits = 5

# trial_duration_samples = trial_duration_sec * sampling_rate
# break_duration_samples = break_duration_sec * sampling_rate
# total_samples_per_trial = trial_duration_samples + break_duration_samples

# # Extract the trials
# def extract_trials(data, num_trials, trial_duration_samples, total_samples_per_trial):
#     trials = []
#     labels = []

#     start_index = data[(data['Trigger'] == 1)].index[0]  # First trigger point
#     for i in range(num_trials):
#         trial_start = start_index + i * total_samples_per_trial
#         trial_end = trial_start + trial_duration_samples
#         trial_data = data.iloc[trial_start:trial_end]
#         trials.append(trial_data.iloc[:, 4:].values)  # Exclude first 4 columns (Time, Trigger, Label, Planning)
#         labels.append(trial_data['Label'].values[0])  # Label is consistent within a trial

#     return np.array(trials), np.array(labels)

# # Extract trials and labels
# trials, labels = extract_trials(data, num_trials, trial_duration_samples, total_samples_per_trial)
# labels = labels - 1

# # Apply bandpass filter
# lowcut = 1.0
# highcut = 40.0
# fs = 250  # Sampling frequency
# trials_filtered = np.array([apply_bandpass_filter(trial, lowcut, highcut, fs) for trial in trials])

# # Determine target shape
# sample_channel = trials_filtered[0, :, 0]
# stft_shape = compute_stft(sample_channel).shape
# cwt_shape = compute_cwt(sample_channel).shape

# target_shape = (max(stft_shape[0], cwt_shape[0]), max(stft_shape[1], cwt_shape[1]))

# # Apply transformations
# transform_funcs = [compute_stft, compute_cwt]
# trials_transformed = apply_transforms(trials_filtered, transform_funcs, target_shape)

# # Normalize the data
# scaler = StandardScaler()
# trials_reshaped = trials_transformed.reshape(-1, trials_transformed.shape[-1])
# trials_normalized = scaler.fit_transform(trials_reshaped).reshape(trials_transformed.shape)

# # Split into training, validation, and testing sets
# X_train, X_temp, y_train, y_temp = train_test_split(trials_normalized, labels, test_size=0.5, random_state=42, stratify=labels)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# # Save X_test and y_test for testing
# np.save('X_test.npy', X_test)
# np.save('y_test.npy', y_test)

# # Custom Dataset class
# class EEGEMGTransformedDataset(Dataset):
#     def __init__(self, trials, labels, augment=False):
#         self.trials = trials
#         self.labels = labels
#         self.augment = augment

#     def __len__(self):
#         return len(self.trials)

#     def __getitem__(self, idx):
#         trial = self.trials[idx]
#         label = self.labels[idx]
#         if self.augment:
#             trial = augment(trial)

#         return torch.tensor(trial, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# # Create DataLoader
# def create_dataloader(trials, labels, batch_size=15, shuffle=True, augment=False):
#     dataset = EEGEMGTransformedDataset(trials, labels, augment=augment)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#     return dataloader

# # Create DataLoaders
# train_loader = create_dataloader(X_train, y_train, augment=True)
# val_loader = create_dataloader(X_val, y_val, shuffle=False)
# test_loader = create_dataloader(X_test, y_test, shuffle=False)

# # Define model, loss, optimizer
# input_channels = trials_transformed.shape[1]
# simple_model = SimpleAMPCNet(num_classes=2, input_channels=input_channels, target_shape=target_shape)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(simple_model.parameters(), lr=0.0007)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# # Training loop with detailed debugging
# num_epochs = 20
# for epoch in range(num_epochs):
#     simple_model.train()  # Set the model to training mode
#     running_loss = 0.0

#     for i, (inputs, labels) in enumerate(train_loader):
#         optimizer.zero_grad()  # Zero the gradients

#         outputs = simple_model(inputs)  # Forward pass
#         loss = criterion(outputs, labels)  # Compute the loss
#         print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')

#         loss.backward()  # Backward pass

#         # Gradient clipping
#         torch.nn.utils.clip_grad_norm_(simple_model.parameters(), max_norm=1.0)

#         # Print gradients for each parameter
#         for name, param in simple_model.named_parameters():
#             if param.grad is not None:
#                 print(f'{name} grad: {param.grad.norm()}')
#                 print(f'{name} value: {param.data.norm()}')
#                 if param.grad.norm().item() == 0:
#                     print(f'Warning: {name} has zero gradient at step {i}')

#         optimizer.step()  # Optimizer step

#         running_loss += loss.item()  # Accumulate the running loss

#     scheduler.step()  # Adjust the learning rate if using a scheduler

#     # Validation
#     simple_model.eval()  # Set the model to evaluation mode
#     val_loss = 0.0
#     correct = 0
#     total = 0

#     with torch.no_grad():  # Disable gradient computation for validation
#         for inputs, labels in val_loader:
#             outputs = simple_model(inputs)  # Forward pass
#             loss = criterion(outputs, labels)  # Compute the loss
#             val_loss += loss.item()  # Accumulate the validation loss

#             _, predicted = torch.max(outputs.data, 1)  # Get the predicted class
#             total += labels.size(0)  # Update the total number of samples
#             correct += (predicted == labels).sum().item()  # Update the number of correct predictions

#     print(f'Epoch [{epoch+1}/{num_epochs}], '
#           f'Train Loss: {running_loss/len(train_loader):.4f}, '
#           f'Val Loss: {val_loss/len(val_loader):.4f}, '
#           f'Val Accuracy: {100 * correct / total:.2f}%')

# # Save the model
# torch.save(simple_model.state_dict(), 'simple_model.pth')
file_path = 'P36_EEG_EMG250Hz.csv'
sampling_rate = 250  # 250 Hz
trial_duration_sec = 8  # 8 seconds
break_duration_sec = 3  # 3 seconds
num_trials = 40
n_splits = 5  # Number of folds for cross-validation

trial_duration_samples = trial_duration_sec * sampling_rate
break_duration_samples = break_duration_sec * sampling_rate
total_samples_per_trial = trial_duration_samples + break_duration_samples

# Extract the trials
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

# Extract trials and labels
data = pd.read_csv(file_path)
trials, labels = extract_trials(data, num_trials, trial_duration_samples, total_samples_per_trial)
labels = labels - 1

# Apply bandpass filter
lowcut = 1.0
highcut = 40.0
fs = 250  # Sampling frequency
trials_filtered = np.array([apply_bandpass_filter(trial, lowcut, highcut, fs) for trial in trials])

# Determine target shape
sample_channel = trials_filtered[0, :, 0]
stft_shape = compute_stft(sample_channel).shape
cwt_shape = compute_cwt(sample_channel).shape

target_shape = (max(stft_shape[0], cwt_shape[0]), max(stft_shape[1], cwt_shape[1]))

# Apply transformations
transform_funcs = [compute_stft, compute_cwt]
trials_transformed = apply_transforms(trials_filtered, transform_funcs, target_shape)

# Normalize the data
scaler = StandardScaler()
trials_reshaped = trials_transformed.reshape(-1, trials_transformed.shape[-1])
trials_normalized = scaler.fit_transform(trials_reshaped).reshape(trials_transformed.shape)

# Custom Dataset class
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

# Create DataLoader
def create_dataloader(trials, labels, batch_size=15, shuffle=True, augment=False):
    dataset = EEGEMGTransformedDataset(trials, labels, augment=augment)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# Define model, loss, optimizer
input_channels = trials_transformed.shape[1]
num_classes = 2
simple_model = SimpleAMPCNet(num_classes=num_classes, input_channels=input_channels, target_shape=target_shape)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(simple_model.parameters(), lr=0.0007)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop with cross-validation
def train_and_evaluate_model(train_loader, val_loader, model, criterion, optimizer, scheduler, num_epochs=20):
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

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=n_splits)
fold = 1
for train_index, val_index in skf.split(trials_normalized, labels):
    print(f'Fold {fold}/{n_splits}')
    X_train, X_val = trials_normalized[train_index], trials_normalized[val_index]
    y_train, y_val = labels[train_index], labels[val_index]

    # Ensure balanced training data
    class_0_indices = np.where(y_train == 0)[0]
    class_1_indices = np.where(y_train == 1)[0]
    balanced_indices = np.hstack([class_0_indices, np.random.choice(class_1_indices, len(class_0_indices), replace=False)])
    np.random.shuffle(balanced_indices)

    X_train_balanced = X_train[balanced_indices]
    y_train_balanced = y_train[balanced_indices]

    # Create DataLoaders
    train_loader = create_dataloader(X_train_balanced, y_train_balanced, augment=True)
    val_loader = create_dataloader(X_val, y_val, shuffle=False)

    # Train and evaluate the model
    simple_model = SimpleAMPCNet(num_classes=num_classes, input_channels=input_channels, target_shape=target_shape)
    optimizer = optim.Adam(simple_model.parameters(), lr=0.0007)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    train_and_evaluate_model(train_loader, val_loader, simple_model, criterion, optimizer, scheduler)

    fold += 1

# Save the model
torch.save(simple_model.state_dict(), 'simple_model.pth')