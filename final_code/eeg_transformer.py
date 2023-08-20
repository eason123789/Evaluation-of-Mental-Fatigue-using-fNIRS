import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from collections import deque
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix



def create_dataloaders(X, y, test_size=0.2, batch_size=32, random_state=42):
    # Impute missing values in X
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Use SMOTE for over-sampling
    smote = SMOTE()
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Create datasets for training and validation using the resampled training data
    train_dataset = CustomDataset(X_train_resampled, y_train_resampled)
    val_dataset = CustomDataset(X_val, y_val)

    # Create dataloaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def remove_duplicates(X, y):
    unique_X, unique_indices = np.unique(X, axis=0, return_index=True)
    unique_y = y[unique_indices]
    return unique_X, unique_y


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


def load_data(data_path):
    # Load the data from the file
    data = np.load(data_path)
    print(f"Spectral data shape: {data['spectral_data'].shape}")
    print(f"Temporal data shape: {data['temporal_data'].shape}")

    # Get the minimum size across the two data arrays
    min_size = min(data['spectral_data'].shape[0], data['temporal_data'].shape[0])

    # Extract and reshape the data, but only up to the minimum size
    X = np.concatenate([
        data['spectral_data'][:min_size].reshape(min_size, -1),
        data['temporal_data'][:min_size].reshape(min_size, -1)
    ], axis=1)
    y = data['labels'][:min_size]  # Make sure labels also match the size

    # Standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Remove duplicate data points
    X, y = remove_duplicates(X, y)
    if len(X) != len(np.unique(X, axis=0)):
        print("There are duplicate data points!")

    return X, y



# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TransformerModel, self).__init__()
        self.embedding_layer = nn.Linear(input_dim, 512)
        self.dropout = nn.Dropout(0.2)
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

# Function to train the model with advanced early stopping
def train_model(model, train_loader, val_loader, num_epochs=20, lr=0.0005, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0
    no_improve_epochs = 0
    val_acc_history = deque(maxlen=5)  # Keep track of last 5 accuracy scores for smoothing

    # saved_models = []  # To keep track of saved models

    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        metrics = evaluate_model(model, val_loader, len(np.unique(y)))
        val_acc = metrics['Accuracy']
        val_acc_history.append(val_acc)
        smoothed_val_acc = sum(val_acc_history) / len(val_acc_history)  # Calculate smoothed (average) accuracy
        val_acc = metrics['Accuracy']
        f1 = metrics['F1 Score']
        for key, value in metrics.items():
            print(f"{key}: {value}")

        print(f"Epoch {epoch}, Validation Accuracy: {val_acc}, Smoothed Validation Accuracy: {smoothed_val_acc}")

        if smoothed_val_acc > best_val_acc:
            best_val_acc = smoothed_val_acc
            no_improve_epochs = 0

            model_path = f'../best_model_{epoch}.pth'
            torch.save(model.state_dict(), model_path)
            saved_models.append(model_path)

            # Keep only top 3 best models
            if len(saved_models) > 3:
                try:
                    os.remove(saved_models.pop(0))
                except FileNotFoundError:
                    print("File not found. Skipping the delete operation.")

            print('Saved best model')

        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print("Early stopping!")
            break
# def train_model(model, train_loader, val_loader, num_epochs=20, lr=0.0005, patience=10):
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001)
#     loss_fn = nn.CrossEntropyLoss()
#
#     best_val_acc = 0
#     no_improve_epochs = 0
#
#     for epoch in range(num_epochs):
#         model.train()
#         for batch_X, batch_y in train_loader:
#             optimizer.zero_grad()
#             outputs = model(batch_X)
#             loss = loss_fn(outputs, batch_y)
#             loss.backward()
#             optimizer.step()
#         scheduler.step()
#
#         metrics = evaluate_model(model, val_loader, len(np.unique(y)))
#         val_acc = metrics['Accuracy']
#         f1 = metrics['F1 Score']
#         for key, value in metrics.items():
#             print(f"{key}: {value}")
#
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             no_improve_epochs = 0
#             torch.save(model.state_dict(), '../best_model1.pth')
#             print('save best model')
#         else:
#             no_improve_epochs += 1
#             if no_improve_epochs >= patience:
#                 print("Early stopping!")
#                 break

# Function to evaluate the model


def evaluate_model(model, val_loader, num_classes):
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
    precision = precision_score(all_true, all_preds, average='macro')
    recall = recall_score(all_true, all_preds, average='macro')

    # Check if it's a binary classification task before computing AUC-ROC
    if num_classes == 2:
        try:
            auc_roc = roc_auc_score(all_true, all_preds)
        except ValueError:
            auc_roc = "N/A (Check if only one class is present in the data)"
    else:
        auc_roc = "N/A (AUC-ROC is typically used for binary classification tasks)"

    cm = confusion_matrix(all_true, all_preds)

    return {
        'Accuracy': val_acc,
        'F1 Score': f1,
        'Precision': precision,
        'Recall': recall,
        'AUC-ROC': auc_roc,
        'Confusion Matrix': cm
    }


# use these functions to load data, create the model, and start training.
if __name__ == '__main__':
    data_path = '../labeled_data_821/all_data.npz'
    X, y = load_data(data_path)
    unique, counts = np.unique(y, return_counts=True)
    print(dict(zip(unique, counts)))

    train_loader, val_loader = create_dataloaders(X, y)
    model = TransformerModel(X.shape[1], len(np.unique(y)))

    saved_models = []  # 这个列表应该在train_model函数中被填充
    train_model(model, train_loader, val_loader)

    # Load the best model and evaluate
    if saved_models:
        best_model_path = saved_models[-1]  # Last model is the best model
        model.load_state_dict(torch.load(best_model_path))

        final_metrics = evaluate_model(model, val_loader, len(np.unique(y)))
        for key, value in final_metrics.items():
            print(f"Final {key}: {value}")
    else:
        print("No saved models, something went wrong.")

        # X, y = load_data()
        #
        # unique, counts = np.unique(y, return_counts=True)
        # print(dict(zip(unique, counts)))
        # train_loader, val_loader = create_dataloaders(X, y)
        # model = TransformerModel(X.shape[1], len(np.unique(y)))
        # train_model(model, train_loader, val_loader)
        # metrics = evaluate_model(model, val_loader, len(np.unique(y)))
        # for key, value in metrics.items():
        #     print(f"{key}: {value}")

