import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Dummy dataset example (replace with your actual data)
class StockPriceDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# Transformer model definition
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, output_size):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x, x)  # Encoder-Decoder structure, or just encoder
        x = self.fc_out(x)
        return x

# Example dataset preprocessing (replace with your own data)
def load_and_preprocess_data():
    # Replace this with loading your dataset
    raw_data = np.random.rand(1000, 10)  # 1000 samples, 10 features
    labels = np.random.rand(1000, 1)  # 1000 labels
    
    # Normalize the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(raw_data)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(scaled_data, labels, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

# Hyperparameters
input_size = 10  # Number of input features
hidden_size = 64
num_layers = 2
num_heads = 4
output_size = 1  # Predicting a single value (stock price)
batch_size = 64
num_epochs = 100

# Load data
X_train, X_val, y_train, y_val = load_and_preprocess_data()

# Convert to datasets
train_dataset = StockPriceDataset(X_train, y_train)
val_dataset = StockPriceDataset(X_val, y_val)

# Create data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, optimizer, and loss function
model = TransformerModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_heads=num_heads, output_size=output_size)
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Lowered learning rate
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        inputs, targets = batch  # inputs: features, targets: stock prices

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print loss for every epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Optionally, evaluate the model on the validation set
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            val_loss += loss.item()

    print(f"Validation Loss after Epoch [{epoch+1}/{num_epochs}]: {val_loss/len(val_loader):.4f}")
