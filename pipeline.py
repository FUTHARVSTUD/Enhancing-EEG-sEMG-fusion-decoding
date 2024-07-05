import pandas as pd
import numpy as np
import torch 
import torch.utils.data as Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from AMPCNet import AMPCNet


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Read the CSV file into a DataFrame
df = pd.read_csv('P36_EEG_EMG250Hz.csv')

# Identify trigger transitions
df['trigger_change'] = df['Trigger'].diff() 

# Find start and end indices of trials
trial_starts = df[df['trigger_change'] == 1].index
trial_ends = df[df['trigger_change'] == -1].index

# Create a list to store trial data
trials = []
for start, end in zip(trial_starts, trial_ends):
    trial_data = df.iloc[start:end + 1][['Fz', 'C3', 'Cz', 'C4', 'Pz', 'P07', 'Oz', 'P08', 'EMG_Right', 'EMG_Left', 'Label', 'Time(s)']].copy()
    trials.append(trial_data)

# Concatenate trial DataFrames
trial_df = pd.concat(trials, keys=range(len(trials)), names=['Trial'])

# Display the first few rows of the trial DataFrame
#print(trial_df.head().to_markdown(index=False, numalign="left", stralign="left"))

# Calculate the maximum length among all trials
max_length = max(len(trial) for trial in trials)

# Function to pad trials with zeros
def pad_trial(trial, max_length):
    padded_trial = np.zeros((max_length, trial.shape[1]))
    padded_trial[:len(trial)] = trial
    return padded_trial

# Apply padding to each trial
#trial_df['Padded_Trial'] = trials['Trials'].apply(lambda x: pad_trial(x, max_length))   
padded_trials = [pad_trial(trial[['Fz', 'C3', 'Cz', 'C4', 'Pz', 'P07', 'Oz', 'P08', 'EMG_Right', 'EMG_Left']].values, max_length) for trial in trials]

# Create a new DataFrame with the padded trials and original labels
trial_df = pd.DataFrame({
    'Padded_Trial': padded_trials,
    'Label': trial_df.groupby('Trial')['Label'].first(),  # Get the label from the first row of each trial
})

# Convert padded trials and labels to PyTorch tensors
padded_trials_tensor = torch.tensor(np.array(padded_trials), dtype=torch.float32).unsqueeze(-1) 
labels_tensor = torch.tensor([trial['Label'].iloc[0] for trial in trials], dtype=torch.long)  


# # Convert padded trials and labels to PyTorch tensors
# padded_trials_tensor = torch.tensor(np.stack(trial_df['Padded_Trial'].values), dtype=torch.float32).unsqueeze(-1)
# labels_tensor = torch.tensor(trial_df['Label'].values, dtype=torch.long)

# Create a custom PyTorch dataset
class AmpcDataset(Dataset.Dataset):
    def __init__(self, trials, labels):
        self.trials = trials
        self.labels = labels

    def __getitem__(self, index):
        return self.trials[index], self.labels[index]

    def __len__(self):
        return len(self.trials)

# Create the dataset
dataset = AmpcDataset(padded_trials_tensor, labels_tensor)

# Print the shape of the first trial and its label
#print("Shape of first trial:", dataset[0][0].shape)
#print("Label of first trial:", dataset[0][1])

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Adjust labels to 0 and 1
labels_tensor = labels_tensor - 1

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Instantiate the model
model = AMPCNet(num_classes=2)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(num_epochs):
    #Training
    model.train()
    epoch_train_loss, epoch_train_acc = 0, 0
    for X, y in train_loader:
        print("Input shape:", X.shape)
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()
        epoch_train_acc += (y_pred.argmax(1) == y).sum().item() / len(y)

    train_losses.append(epoch_train_loss / len(train_loader))
    train_accs.append(epoch_train_acc / len(train_loader))

    # Validation
    model.eval()
    epoch_val_loss, epoch_val_acc = 0, 0
    with torch.no_grad():
        for X, y in val_loader:
            y_pred = model(X)
            loss = criterion(y_pred, y)
            epoch_val_loss += loss.item()
            epoch_val_acc += (y_pred.argmax(1) == y).sum().item() / len(y)

    val_losses.append(epoch_val_loss / len(val_loader))
    val_accs.append(epoch_val_acc / len(val_loader))

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accs[-1]:.4f}")

# Plotting
plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Training Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.show()