import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Define a custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Load data
X_list = []
y_list = []

# Iterate over the files in the data_labeling_R folder
for filename in os.listdir('../data_labeling_R'):
    if filename.endswith('.npz'):
        data = np.load(os.path.join('../data_labeling_R', filename))

        # Reshape and concatenate data
        reshaped_spectral_data = data['spectral_data'].reshape(data['spectral_data'].shape[0], -1)
        reshaped_temporal_data = data['temporal_data'].reshape(data['temporal_data'].shape[0], -1)
        X = np.concatenate([reshaped_spectral_data, reshaped_temporal_data], axis=1)

        y = data['labels']
        X_list.append(X)
        y_list.append(y)

# Concatenate all data
X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)

np.savez('../combined_data.npz', X=X, y=y)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create DataLoader objects
train_dataset = CustomDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = CustomDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TransformerModel, self).__init__()
        self.embedding_layer = nn.Linear(input_dim, 256)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x)
        return x

# Training code
model = TransformerModel(X.shape[1], len(np.unique(y)))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(20):  # Number of epochs
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = loss_fn(outputs, batch_y)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()


    val_acc = 100 * correct / len(X_val)
    print(f'Epoch {epoch}, validation accuracy: {val_acc}%')
