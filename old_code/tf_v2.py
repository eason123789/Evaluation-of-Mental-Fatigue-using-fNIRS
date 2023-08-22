import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data(data_path='data_labeling_R'):
    X_list = []
    y_list = []

    for filename in os.listdir(data_path):
        if filename.endswith('.npz'):
            data = np.load(os.path.join(data_path, filename))
            reshaped_spectral_data = data['spectral_data'].reshape(data['spectral_data'].shape[0], -1)
            reshaped_temporal_data = data['temporal_data'].reshape(data['temporal_data'].shape[0], -1)
            X = np.concatenate([reshaped_spectral_data, reshaped_temporal_data], axis=1)
            y = data['labels']
            X_list.append(X)
            y_list.append(y)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    return X, y

def create_dataloaders(X, y, test_size=0.2, batch_size=32, random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    train_dataset = CustomDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = CustomDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TransformerModel, self).__init__()
        self.embedding_layer = nn.Linear(input_dim, 256)
        self.dropout = nn.Dropout(0.1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x)
        return x

def train_model(model, train_loader, val_loader, num_epochs=20, lr=0.001, patience=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
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
            torch.save(model.state_dict(), '../best_model.pth')
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print("Early stopping!")
                break

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

# Finally, we use these functions to load data, create the model, and start training.
# Note that this part should be in the main function or in a separate script.
if __name__ == '__main__':
    X, y = load_data()
    train_loader, val_loader = create_dataloaders(X, y)
    model = TransformerModel(X.shape[1], len(np.unique(y)))
    train_model(model, train_loader, val_loader)
