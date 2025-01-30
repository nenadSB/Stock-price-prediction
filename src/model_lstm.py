import torch
import torch.nn as nn

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        """
        LSTM Model for stock price prediction.
        
        Args:
            input_size (int): Number of input features (e.g., number of columns in dataset like Open, Close, etc.)
            hidden_size (int): Number of hidden units in the LSTM layer.
            num_layers (int): Number of LSTM layers.
            output_size (int): Number of output features (e.g., 1 for predicting a single stock price).
            dropout (float, optional): Dropout rate between LSTM layers. Defaults to 0.0.
        """
        super(StockLSTM, self).__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Fully connected layer to map the output of LSTM to the desired output size
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (tensor): Input tensor of shape (batch_size, sequence_length, input_size).
        
        Returns:
            tensor: Output prediction (batch_size, output_size).
        """
        # LSTM layer: output, (h_n, c_n)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Only take the output from the last time step
        last_lstm_output = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        # Pass the output through the fully connected layer
        out = self.fc(last_lstm_output)  # Shape: (batch_size, output_size)
        
        return out
