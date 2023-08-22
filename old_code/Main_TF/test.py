import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# ----------------------- Data Preprocessing Functions -----------------------

def load_data(data_path):
    """Load and preprocess the dataset."""
    data = np.load(data_path)
    min_size = min(data['spectral_data'].shape[0], data['temporal_data'].shape[0])

    X = np.concatenate([
        data['spectral_data'][:min_size].reshape(min_size, -1),
        data['temporal_data'][:min_size].reshape(min_size, -1)
    ], axis=1)
    y = data['labels'][:min_size]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X, y = remove_duplicates(X, y)
    return X, y

def remove_duplicates(X, y):
    """Remove duplicate samples from the dataset."""
    unique_X, unique_indices = np.unique(X, axis=0, return_index=True)
    unique_y = y[unique_indices]
    return unique_X, unique_y

def create_dataloaders(X, y, test_size=0.2, batch_size=32, random_state=42):
    """Create data loaders for training and validation."""
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    smote = SMOTE()
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    train_dataset = CustomDataset(X_train_resampled, y_train_resampled)
    val_dataset = CustomDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# ----------------------- Custom Dataset Definition -----------------------

class CustomDataset(torch.utils.data.Dataset):
    """Custom dataset for the provided data."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------------------- Model Definition -----------------------

class TransformerModel(nn.Module):
    """Transformer based model for classification."""
    def __init__(self, input_dim, num_classes):
        super(TransformerModel, self).__init__()
        self.embedding_layer = nn.Linear(input_dim, 512)
        self.dropout = nn.Dropout(0.2)
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=4, dim_feedforward=1024)
        self.layer_norm = nn.LayerNorm(512)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = self.layer_norm(x)
        x = self.classifier(x)
        return x

# ----------------------- Training and Evaluation Functions -----------------------

def train_model(model, train_loader, val_loader, num_epochs=20, lr=0.0005, patience=5):
    """Train the provided model using the given data loaders."""
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

        metrics = evaluate_model(model, val_loader, len(np.unique(y)))
        val_acc = metrics['Accuracy']
        for key, value in metrics.items():
            print(f"{key}: {value}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_epochs = 0
            torch.save(model.state_dict(), '../../best_model1.pth')
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print("Early stopping!")
                break

def evaluate_model(model, val_loader, num_classes):
    """Evaluate the model and return performance metrics."""
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

# ----------------------- Main Execution -----------------------

def main():
    # Load data and print label counts
    data_path = '/labeled_data_818/all_data.npz'
    X, y = load_data(data_path)
    unique, counts = np.unique(y, return_counts=True)
    print(dict(zip(unique, counts)))

    # Create data loaders
    train_loader, val_loader = create_dataloaders(X, y)

    # Define and train model
    model = TransformerModel(X.shape[1], len(np.unique(y)))
    train_model(model, train_loader, val_loader)

    # Evaluate and print metrics
    metrics = evaluate_model(model, val_loader, len(np.unique(y)))
    for key, value in metrics.items():
        print(f"{key}: {value}")

if __name__ == '__main__':
    main()
