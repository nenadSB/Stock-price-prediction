import torch
import torch.nn as nn
import pandas as pd
import yaml
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.model_lstm import StockLSTM
from src.model_transformer import TransformerModel

class Evaluator:
    """
    Evaluate the trained model using test data.
    """
    def __init__(self, config_path="config/config.yaml", model_type="lstm"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type

        if model_type == "lstm":
            self.model = StockLSTM(
                input_size=self.config["model"]["lstm"]["input_size"],
                hidden_size=self.config["model"]["lstm"]["hidden_size"],
                num_layers=self.config["model"]["lstm"]["num_layers"],
                output_size=self.config["model"]["lstm"]["output_size"],
                dropout=self.config["model"]["lstm"]["dropout"]
            ).to(self.device)
            model_path = "models/lstm_model.pth"
        elif model_type == "transformer":
            self.model = TransformerModel(
                num_layers=self.config["model"]["transformer"]["num_layers"],
                d_model=self.config["model"]["transformer"]["d_model"],
                num_heads=self.config["model"]["transformer"]["num_heads"],
                dff=self.config["model"]["transformer"]["dff"],
                dropout=self.config["model"]["transformer"]["dropout"],
                input_size=self.config["model"]["lstm"]["input_size"],
                output_size=self.config["model"]["lstm"]["output_size"]
            ).to(self.device)
            model_path = "models/transformer_model.pth"
        else:
            raise ValueError("Invalid model type. Choose 'lstm' or 'transformer'.")

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def load_test_data(self):
        """
        Load test data and convert it to tensors.
        """
        df = pd.read_csv(self.config["data"]["processed_data_path"])
        test_size = int(len(df) * 0.2)  # 20% of data for testing
        test_df = df.iloc[-test_size:]

        x_test = torch.tensor(test_df.iloc[:, 1:].values, dtype=torch.float32)
        y_test = torch.tensor(test_df.iloc[:, -1].values, dtype=torch.float32).view(-1, 1)

        dataset = TensorDataset(x_test, y_test)
        return DataLoader(dataset, batch_size=1, shuffle=False), y_test.numpy()

    def evaluate(self):
        """
        Compute MSE, RMSE, and MAE for the model.
        """
        dataloader, y_true = self.load_test_data()
        predictions = []

        with torch.no_grad():
            for x_batch, _ in dataloader:
                x_batch = x_batch.to(self.device)
                pred = self.model(x_batch).cpu().numpy()
                predictions.append(pred.flatten()[0])

        mse = mean_squared_error(y_true, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, predictions)

        print(f"Evaluation Results for {self.model_type.upper()} Model:")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")

        return {"mse": mse, "rmse": rmse, "mae": mae}

if __name__ == "__main__":
    model_choice = input("Enter model type ('lstm' or 'transformer'): ").strip().lower()
    evaluator = Evaluator(model_type=model_choice)
    evaluator.evaluate()
