from src.data_preprocessing import DataPreprocessor
from src.train import Trainer

if __name__ == "__main__":
    print("Starting Stock Price Prediction Pipeline...")

    # Step 1: Preprocess data
    processor = DataPreprocessor()
    processor.preprocess()

    # Step 2: Select model type
    model_choice = input("Enter model type ('lstm' or 'transformer'): ").strip().lower()

    # Step 3: Train model
    trainer = Trainer(model_type=model_choice)
    trainer.train()

    print("Pipeline completed successfully!")
