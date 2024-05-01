import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from AMPCNet import AMPCNet
from data_utils import load_trial_data, preprocess_and_extract_features

def main():
    num_classes = 5  
    learning_rate = 0.001
    batch_size = 16
    num_epochs = 20

    model = AMPCNet(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    all_data = []
    all_labels = []

    subj_id = 10
    trial_id = 1 
    for subj_id in range(10, 11):  
        for trial_id in range(1, 2): 
            eeg, semg = load_trial_data(subj_id, trial_id)
            features = preprocess_and_extract_features(eeg, semg)
            all_data.append(features)
            label = ...  
            all_labels.append(label)

    data_tensor = torch.from_numpy(np.array(all_data)).float()
    label_tensor = torch.from_numpy(np.array(all_labels)).long() 
    dataset = TensorDataset(data_tensor, label_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(data.size(0), 16, *data.shape[1:])  # Reshape for AMPCNet 

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
            
            model_save_path = f"model_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, model_save_path)

if __name__ == "__main__":
    main()
