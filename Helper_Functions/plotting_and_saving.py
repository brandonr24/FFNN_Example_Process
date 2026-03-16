import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy
import torch
import os

model_training_data_path = "Data"
data_path = "Results\Data"
plots_path = "Results\Plots"

def plot_and_save_data(x_data, y_data, file_name = "Data", save_plot = False, save_data = False):
    df = pd.DataFrame({"data": y_data.tolist()}) # Assume data is given as a tensor
    if save_data: df.to_excel(f"{data_path}\{file_name}.xlsx") # Save to Excel Document

    plt.plot(x_data.detach(), y_data.detach()) # Addume data is given as a tensor
    if save_plot: plt.savefig(f"{plots_path}\{file_name}.jpg")
    plt.show()

def plot_multiple_data(x_data, y_data, legend=None, file_name="Plot", save_plot=False, plots_path="."):
    l_train = len(y_data[0])
    l_test = len(y_data[1])
    l_val = len(y_data[2])
    
    x_train_slice = x_data[0 : l_train]
    x_test_slice = x_data[l_train : l_train + l_test]
    x_val_slice = x_data[l_train + l_test : l_train + l_test + l_val]
    
    x_slices = [x_train_slice, x_test_slice, x_val_slice]

    for i in range(len(y_data) - 3):
        x = x_slices[i % 3].detach().squeeze()
        y = y_data[i].detach().squeeze()
        
        x_sorted, indices = torch.sort(x)
        y_sorted = y[indices]
        
        x_np = x_sorted.cpu().numpy()
        y_np = y_sorted.cpu().numpy()
        
        current_label = legend[i] if legend and i < len(legend) else None
        
        plt.plot(x_np, y_np, label=current_label) 
        
    if legend:
        plt.legend()
        
    if save_plot:
        os.makedirs(plots_path, exist_ok=True)
        save_file = os.path.join(plots_path, f"{file_name}.jpg")
        plt.savefig(save_file)
        
    plt.show()

def save_data(y_data, legend = None, file_name = "Data"):
    df = pd.DataFrame({legend if legend else "Data": y_data.detach().numpy().flatten()})
    df.to_excel(f"{data_path}\{file_name}.xlsx") # Save to Excel Document

import pandas as pd
import torch

def save_multiple_data(x_data, y_data_array, legend=None, file_name="Data"):
    x_flat = x_data.detach().cpu().numpy().flatten()
    df = pd.DataFrame({"X_Data": x_flat})
    
    l_train = len(y_data_array[0])
    l_test = len(y_data_array[1])
    l_val = len(y_data_array[2])
    
    idx_train = range(0, l_train)
    idx_test = range(l_train, l_train + l_test)
    idx_val = range(l_train + l_test, l_train + l_test + l_val)
    
    for i, data_tensor in enumerate(y_data_array):
        data_name = legend[i] if legend and i < len(legend) else f"Data{i}"
        np_data = data_tensor.detach().cpu().numpy().flatten()
        
        if i >= len(y_data_array) - 3:
            series = pd.Series(np_data, index=[0]) 
            
        else:
            if i % 3 == 0:
                series = pd.Series(np_data, index=idx_train)
            elif i % 3 == 1:
                series = pd.Series(np_data, index=idx_test)
            else:
                series = pd.Series(np_data, index=idx_val)
                
        df[data_name] = series
        
    df.to_csv(f"{file_name}.csv", index=False)
    
    print(f"Data successfully saved to {file_name}.csv")
    return df

def save_training_data(x_data, y_data, file_name = "Data"):
    df = pd.DataFrame({
        "x_data": x_data.detach().cpu().squeeze().tolist(), 
        "y_data": y_data.detach().cpu().squeeze().tolist()
    })

    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

    train_df.to_csv(f"{model_training_data_path}\{file_name}train.csv", index=False)
    val_df.to_csv(f"{model_training_data_path}\{file_name}val.csv", index=False)
    test_df.to_csv(f"{model_training_data_path}\{file_name}test.csv", index=False)