import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, dropout, input_size, output_size):
        super(TransformerModel, self).__init__()

        # Linear layer to match input size to d_model
        self.embedding = nn.Linear(input_size, d_model)
        
        # Transformer Encoder Layer
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dff, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Fully connected layer to predict output size
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        # Debugging: Print the shape before the embedding layer
        print(f"Input shape before embedding: {x.shape}")
        
        # Ensure the input has the correct shape: (batch_size, sequence_length, input_size)
        if x.dim() == 2:  # If input is (batch_size, input_size), add sequence length dimension
            x = x.unsqueeze(1)  # Shape: (batch_size, sequence_length=1, input_size)
        
        # Print shape after adding sequence length dimension
        print(f"Shape after adding sequence length dimension: {x.shape}")

        # Apply embedding layer (Linear transformation from input_size to d_model)
        x = self.embedding(x)  # Shape becomes (batch_size, sequence_length, d_model)
        
        # Print shape after embedding
        print(f"Shape after embedding: {x.shape}")

        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # Shape: (batch_size, sequence_length, d_model)

        # Take the output from the last time step
        x = x[:, -1, :]  # Shape: (batch_size, d_model)

        # Apply fully connected layer to predict output
        out = self.fc(x)  # Shape: (batch_size, output_size)
        
        return out
