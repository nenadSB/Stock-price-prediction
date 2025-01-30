import torch
import pandas as pd
import yaml
from src.model_lstm import StockLSTM
from src.model_transformer import TransformerModel

class Predictor:
    """
    Make predictions using the trained model.
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

    def predict(self, input_data):
        """
        Predict stock price based on input features.
        """
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_tensor).cpu().numpy()
        
        print(f"Predicted stock price: {prediction[0][0]:.4f}")
        return prediction[0][0]

if __name__ == "__main__":
    model_choice = input("Enter model type ('lstm' or 'transformer'): ").strip().lower()
    predictor = Predictor(model_type=model_choice)

    sample_data = [float(x) for x in input("Enter stock features (comma-separated): ").split(",")]
    predictor.predict(sample_data)
