import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

# Import custom modules
from data_preprocessing import load_data, extract_trials, apply_bandpass_filter, compute_stft, compute_cwt, apply_transforms, normalize_data
from pipeline_9 import SimpleAMPCNet  # Assuming the model definition is in model_definition.py

# Constants
file_path = 'P36_EEG_EMG250Hz.csv'
sampling_rate = 250
trial_duration_sec = 8
break_duration_sec = 3
num_trials = 40
trial_duration_samples = trial_duration_sec * sampling_rate
break_duration_samples = break_duration_sec * sampling_rate
total_samples_per_trial = trial_duration_samples + break_duration_samples

# Load data
data = load_data(file_path)

# Extract trials and labels
trials, labels = extract_trials(data, num_trials, trial_duration_samples, total_samples_per_trial)
labels = labels - 1

# Apply bandpass filter
lowcut = 1.0
highcut = 40.0
fs = sampling_rate
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
trials_normalized = normalize_data(trials_transformed)

# Save X_test and y_test for testing
np.save('X_test.npy', trials_normalized)
np.save('y_test.npy', labels)

# Load the saved model
model = 'simple_model.pth'
model.eval()

# Prepare the test DataLoader
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test).long())
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Perform inference on the test set
all_predictions = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute evaluation metrics
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average='weighted')
recall = recall_score(all_labels, all_predictions, average='weighted')
f1 = f1_score(all_labels, all_predictions, average='weighted')
conf_matrix = confusion_matrix(all_labels, all_predictions)

print(f'Test Accuracy: {accuracy:.4f}')
print(f'Test Precision: {precision:.4f}')
print(f'Test Recall: {recall:.4f}')
print(f'Test F1 Score: {f1:.4f}')
print('Confusion Matrix:')
print(conf_matrix)
