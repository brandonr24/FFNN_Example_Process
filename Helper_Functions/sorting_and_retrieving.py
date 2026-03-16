import pandas as pd
import numpy as np
import torch
import ast

def get_data_from_folder(folder_loc):
    train_df = pd.read_csv(f"{folder_loc}train.csv").sort_values(by='x_data')
    val_df = pd.read_csv(f"{folder_loc}val.csv").sort_values(by='x_data')
    test_df = pd.read_csv(f"{folder_loc}test.csv").sort_values(by='x_data')
    
    # Train
    x_train = torch.tensor(train_df['x_data'].values, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(train_df['y_data'].values, dtype=torch.float32).unsqueeze(1)

    # Validation
    x_val = torch.tensor(val_df['x_data'].values, dtype=torch.float32).unsqueeze(1)
    y_val = torch.tensor(val_df['y_data'].values, dtype=torch.float32).unsqueeze(1)

    # Test
    x_test = torch.tensor(test_df['x_data'].values, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(test_df['y_data'].values, dtype=torch.float32).unsqueeze(1)

    return [x_test, x_train, x_val], [y_test, y_train, y_val]