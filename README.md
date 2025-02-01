For this project, we used the following tools:

### 1. **Python**  
   - **Version**: Python 3.12.3  
   - **Purpose**: The main programming language for the entire project. It was used for data processing, model development, training, and evaluation.

### 2. **Libraries and Frameworks**:
   - **Pandas**  
     - **Purpose**: Used for data manipulation and cleaning, specifically for loading CSV files, handling missing data, and processing the raw stock data.
   - **NumPy**   
     - **Purpose**: Often used for numerical operations, especially in handling and transforming data in arrays.
   - **PyTorch (Version 2.6.0)**  
     - **Purpose**: Used for building and training deep learning models (LSTM and Transformer).
   - **Matplotlib/Seaborn**  
     - **Purpose**: Potentially used for data visualization, though not explicitly mentioned, it might be useful for plotting during training and evaluation phases.
   - **Scikit-learn**  
     - **Purpose**: Used for tasks like splitting data into training/testing sets and calculating performance metrics (e.g., mean absolute error).
   - **YAML**  
     - **Purpose**: Used to handle configuration files (`config.yaml`) that store paths and hyperparameters for the models and training process.

### 3. **Tools for Model Evaluation and Hyperparameter Tuning**:
   - **PyTorch**: For training deep learning models, evaluating performance, and tuning hyperparameters like dropout rates, number of layers, etc.
   - **Training script (`train.py`)**: Handles the model training process, using data from the processed dataset.
   - **Hyperparameter tuning script (`hyperparameter_tuning.py`)**: Used to optimize model hyperparameters, like learning rate and batch size.

### 4. **Data Management Tools**:
   - **CSV Files**  
     - **Purpose**: Used for storing both raw and processed stock data.
   - **Directories (`data/raw/`, `data/processed/`)**  
     - **Purpose**: Organize raw data, processed data, and model outputs.

### 5. **Other Utility Tools**:
   - **OS library**  
     - **Purpose**: Used to check and handle file paths, directories, and file existence during data loading and processing.
   - **Flask (for future use)**  
     - **Purpose**: If you decide to deploy the model for predictions through an API, Flask can be used to serve the model.

These tools and libraries together support the data processing, model training, and overall workflow for the stock price prediction project.
