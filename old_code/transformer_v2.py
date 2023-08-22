import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from data_processing import get_data

class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, num_classes):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(model_dim, num_classes)

    def forward(self, src):
        output = self.transformer_encoder(src)
        output = self.decoder(output[:,-1,:])
        return output

def prepare_data(features, labels, batch_size):
    # Split into train and test
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors and create dataloaders
    train_data = TensorDataset(torch.from_numpy(features_train), torch.from_numpy(labels_train))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    test_data = TensorDataset(torch.from_numpy(features_test), torch.from_numpy(labels_test))
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    return train_loader, test_loader

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    for batch, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return correct / len(test_loader.dataset)

def main():
    # Get data
    features, labels = get_data()

    # Prepare data
    train_loader, test_loader = prepare_data(features, labels, batch_size=64)

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define model
    model = TransformerModel(input_dim=features.shape[-1], model_dim=512, num_heads=8, num_layers=3, num_classes=3).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # Train model
    train(model, train_loader, criterion, optimizer, device)

    # Evaluate model
    accuracy = evaluate(model, test_loader, device)
    print(f'Test accuracy: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    main()
