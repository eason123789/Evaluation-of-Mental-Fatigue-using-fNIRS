import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Define a custom dataset class that inherits from PyTorch's Dataset class
class CustomDataset(torch.utils.data.Dataset):
    # Initialize dataset with data and labels
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    # Define the length of the dataset as the number of samples
    def __len__(self):
        return len(self.X)

    # Define how to get individual items from the dataset
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Function to load data from .npz files
def load_data(data_path='/Users/easonpeng/Desktop/University of Nottingham/Evaluation of Mental Fatigue at Multilevel using functional Near Infrared Spectroscopy/Evaluation-of-Mental-Fatigue-using-fNIRS/labeled_data_814'): # data_labeling_R
    X_list = []  # To hold the data
    y_list = []  # To hold the labelsx

    # Loop through each file in the directory
    for filename in os.listdir(data_path):
        # Check if the file is a .npz file
        if filename.endswith('.npz'):
            # Load the data from the file
            data = np.load(os.path.join(data_path, filename))
            # Print the shapes of the arrays
            print(f"File: {filename}")
            print(f"Spectral data shape: {data['spectral_data'].shape}")
            print(f"Temporal data shape: {data['temporal_data'].shape}")
            # Reshape the data and concatenate
            X = np.concatenate([data['spectral_data'].reshape(data['spectral_data'].shape[0], -1),
                                data['temporal_data'].reshape(data['temporal_data'].shape[0], -1)], axis=1)
            y = data['labels']
            X_list.append(X)
            y_list.append(y)

    # Concatenate all data and labels into a single array
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    return X, y

# Function to create dataloaders for training and validation
def create_dataloaders(X, y, test_size=0.2, batch_size=32, random_state=42):
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Create datasets for training and validation
    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)

    # Create dataloaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TransformerModel, self).__init__()
        self.embedding_layer = nn.Linear(input_dim, 512)
        self.dropout = nn.Dropout(0.1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048)
        self.layer_norm = nn.LayerNorm(512)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = self.layer_norm(x)
        x = self.classifier(x)
        return x

# Function to train the model
def train_model(model, train_loader, val_loader, num_epochs=20, lr=0.001, patience=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_acc, f1 = evaluate_model(model, val_loader)

        print(f'Epoch {epoch}, validation accuracy: {val_acc}%, F1 Score: {f1}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_epochs = 0
            torch.save(model.state_dict(), '../../best_model1.pth')
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print("Early stopping!")
                break

# Function to evaluate the model
def evaluate_model(model, val_loader):
    model.eval()
    correct = 0
    all_preds = []
    all_true = []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            all_preds.extend(predicted.tolist())
            all_true.extend(batch_y.tolist())

    val_acc = 100 * correct / len(val_loader.dataset)
    f1 = f1_score(all_true, all_preds, average='macro')

    return val_acc, f1

# Finally,  use these functions to load data, create the model, and start training.
if __name__ == '__main__':
    X, y = load_data()
    train_loader, val_loader = create_dataloaders(X, y)
    model = TransformerModel(X.shape[1], len(np.unique(y)))
    train_model(model, train_loader, val_loader)
