import pandas as pd
import numpy as np
import torch
import ast

def get_data_from_folder(folder_loc):
    train_df = pd.read_excel(f"{folder_loc}train.xlsx")
    val_df = pd.read_excel(f"{folder_loc}val.xlsx")
    test_df = pd.read_excel(f"{folder_loc}test.xlsx")

    def parse_column(series):
        parsed_lists = series.apply(ast.literal_eval).tolist()
        return np.array(parsed_lists)

    # Train
    x_train = torch.tensor(parse_column(train_df['x_data']), dtype=torch.float32)
    y_train = torch.tensor(parse_column(train_df['y_data']), dtype=torch.float32)

    # Validation
    x_val = torch.tensor(parse_column(val_df['x_data']), dtype=torch.float32)
    y_val = torch.tensor(parse_column(val_df['y_data']), dtype=torch.float32)

    # Test
    x_test = torch.tensor(parse_column(test_df['x_data']), dtype=torch.float32)
    y_test = torch.tensor(parse_column(test_df['y_data']), dtype=torch.float32)

    return [x_test, x_train, x_val], [y_test, y_train, y_val]