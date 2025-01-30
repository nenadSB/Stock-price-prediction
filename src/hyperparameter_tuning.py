import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from src.model_lstm import StockLSTM
from src.model_transformer import TransformerModel
from sklearn.metrics import mean_squared_error
from src.data_preprocessing import DataPreprocessor
import yaml

class HyperparameterTuning:
    """
    Hyperparameter tuning for LSTM and Transformer models using Optuna.
    """
    def __init__(self, config_path="config/config.yaml", model_type="lstm"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.df = DataPreprocessor().load_data()

    def load_data(self):
        """
        Load preprocessed data and split into train and test sets.
        """
        train_size = int(len(self.df) * 0.8)
        train_df = self.df[:train_size]
        test_df = self.df[train_size:]

        x_train = train_df.iloc[:, 1:].values
        y_train = train_df.iloc[:, -1].values
        x_test = test_df.iloc[:, 1:].values
        y_test = test_df.iloc[:, -1].values

        return (torch.tensor(x_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32),
                torch.tensor(x_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.float32))

    def objective(self, trial):
        """
        Objective function for Optuna to optimize hyperparameters.
        """
        # Sample hyperparameters
        batch_size = trial.suggest_int('batch_size', 16, 128, step=16)
        num_layers = trial.suggest_int('num_layers', 1, 3)
        hidden_size = trial.suggest_int('hidden_size', 64, 256, step=64)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        dropout = trial.suggest_uniform('dropout', 0.0, 0.5)

        # Choose model based on type
        if self.model_type == "lstm":
            model = StockLSTM(
                input_size=self.config["model"]["lstm"]["input_size"],
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=self.config["model"]["lstm"]["output_size"],
                dropout=dropout
            ).to(self.device)
        elif self.model_type == "transformer":
            model = TransformerModel(
                num_layers=num_layers,
                d_model=self.config["model"]["transformer"]["d_model"],
                num_heads=self.config["model"]["transformer"]["num_heads"],
                dff=self.config["model"]["transformer"]["dff"],
                dropout=dropout,
                input_size=self.config["model"]["lstm"]["input_size"],
                output_size=self.config["model"]["lstm"]["output_size"]
            ).to(self.device)
        else:
            raise ValueError("Invalid model type. Choose 'lstm' or 'transformer'.")

        # Create DataLoader
        x_train, y_train, x_test, y_test = self.load_data()
        train_data = torch.utils.data.TensorDataset(x_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

        # Training loop
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        model.train()
        for epoch in range(self.config["training"]["num_epochs"]):
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                predictions = model(x_batch)
                loss = criterion(predictions, y_batch.view(-1, 1))
                loss.backward()
                optimizer.step()

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            predictions = model(x_test.to(self.device))
            mse = mean_squared_error(y_test, predictions.cpu().numpy())
        
        return mse

    def tune(self):
        """
        Run the Optuna optimization process.
        """
        study = optuna.create_study(direction='minimize')  # Minimize the MSE
        study.optimize(self.objective, n_trials=10)

        print("Best hyperparameters found: ")
        print(study.best_params)
        print(f"Best MSE: {study.best_value:.4f}")

if __name__ == "__main__":
    model_choice = input("Enter model type ('lstm' or 'transformer'): ").strip().lower()
    tuner = HyperparameterTuning(model_type=model_choice)
    tuner.tune()
