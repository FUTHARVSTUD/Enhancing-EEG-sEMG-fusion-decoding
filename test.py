# import torch
# from torch.utils.data import DataLoader
from pipeline_9 import EEGEMGTransformedDataset, SimpleAMPCNet, apply_bandpass_filter, compute_stft, compute_cwt, apply_transforms, DataLoader, simple_model

# simple_model = SimpleAMPCNet(num_classes=2, input_channels=input_channels, target_shape=target_shape)

# saved_model_path = 'best_model.pth'
# simple_model.load_state_dict(torch.load(saved_model_path))

# simple_model.eval()

# # Prepare your test data (assuming you have X_test and y_test)
# test_dataset = EEGEMGTransformedDataset(X_test, y_test)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# # Test the model
# correct = 0
# total = 0

# with torch.no_grad():
#     for inputs, labels in test_loader:
#         outputs = simple_model(inputs)  # Forward pass
#         _, predicted = torch.max(outputs.data, 1)  # Get the predicted class
        
#         total += labels.size(0)  # Update total count of samples
#         correct += (predicted == labels).sum().item()  # Count correct predictions

# test_accuracy = 100 * correct / total
# print(f'Test Accuracy: {test_accuracy:.2f}%')
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch

# Load the CSV file
file_path = 'P36_EEG_EMG250Hz.csv'
data = pd.read_csv(file_path)

# Filter out trials meant for classification (Trigger=1 and Planning=1)
test_data = data[(data['Trigger'] == 1) & (data['Planning'] == 1)]

# Extract relevant columns for classification (Fz, C3, Cz, C4, Pz, P07, Oz, P08, EMG_Right, EMG_Left)
test_data = test_data[['Fz', 'C3', 'Cz', 'C4', 'Pz', 'P07', 'Oz', 'P08', 'EMG_Right', 'EMG_Left', 'Label']]

# Optionally, you can convert the data to numpy arrays for easier manipulation
test_features = test_data[['Fz', 'C3', 'Cz', 'C4', 'Pz', 'P07', 'Oz', 'P08', 'EMG_Right', 'EMG_Left']].values
test_labels = test_data['Label'].values - 1  # Adjust labels if necessary (subtracting 1 as per your previous code)

lowcut = 1.0
highcut = 40.0
fs = 250  # Sampling frequency
test_features_filtered = np.array([apply_bandpass_filter(trial, lowcut, highcut, fs) for trial in test_features])

# Determine target shape (using max dimensions from STFT and CWT as before)
sample_channel = test_features_filtered[0]
stft_shape = compute_stft(sample_channel).shape
cwt_shape = compute_cwt(sample_channel).shape
target_shape = (max(stft_shape[0], cwt_shape[0]), max(stft_shape[1], cwt_shape[1]))

# Apply transformations (STFT and CWT)
transform_funcs = [compute_stft, compute_cwt]
test_features_transformed = apply_transforms(test_features_filtered, transform_funcs, target_shape)

# Normalize the data
scaler = StandardScaler()
test_features_reshaped = test_features_transformed.reshape(-1, test_features_transformed.shape[-1])
test_features_normalized = scaler.fit_transform(test_features_reshaped).reshape(test_features_transformed.shape)

# Optionally, you can convert the normalized data back to a Pandas DataFrame for convenience
test_data_normalized = pd.DataFrame(data=test_features_normalized, columns=test_data.columns[:-1])  # Exclude 'Label' column

batch_size = 20

# Ensure test data is in the correct format for PyTorch DataLoader
test_dataset = EEGEMGTransformedDataset(test_data_normalized.values, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

simple_model.eval()

correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = simple_model(inputs)  # Forward pass
        _, predicted = torch.max(outputs.data, 1)  # Get the predicted class
        
        total += labels.size(0)  # Update total count of samples
        correct += (predicted == labels).sum().item()  # Count correct predictions

test_accuracy = 100 * correct / total
print(f'Test Accuracy: {test_accuracy:.2f}%')