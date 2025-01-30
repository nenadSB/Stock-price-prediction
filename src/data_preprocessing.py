import pandas as pd
import yaml
import os

class DataPreprocessor:
    """
    Handles loading, cleaning, and preprocessing stock market data.
    """
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.raw_data_path = self.config["data"]["raw_data_path"]
        self.processed_data_path = self.config["data"]["processed_data_path"]

    def load_data(self):
        """
        Load raw stock market data from CSV.
        """
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Raw data file not found: {self.raw_data_path}")
        
        data = pd.read_csv(self.raw_data_path)
        print("Data loaded successfully.")
        return data

    def clean_data(self, data):
        """
        Perform basic data cleaning such as handling missing values.
        """
        data = data.dropna()  # Remove missing values
        data = data.sort_values(by="Date")  # Ensure data is sorted by date
        print("Data cleaned successfully.")
        return data

    def save_processed_data(self, data):
        """
        Save the cleaned dataset to the specified processed data path.
        """
        # Ensure the 'processed' folder exists
        os.makedirs(os.path.dirname(self.processed_data_path), exist_ok=True)
        
        data.to_csv(self.processed_data_path, index=False)
        print(f"Processed data saved to {self.processed_data_path}")

    def preprocess(self):
        """
        Full pipeline: load, clean, and save processed data.
        """
        data = self.load_data()
        cleaned_data = self.clean_data(data)
        self.save_processed_data(cleaned_data)

# Run preprocessing when executed
if __name__ == "__main__":
    processor = DataPreprocessor()
    processor.preprocess()
