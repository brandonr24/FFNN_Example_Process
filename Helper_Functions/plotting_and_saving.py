import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy
import torch

model_training_data_path = "Data"
data_path = "Results\Data"
plots_path = "Results\Plots"

def plot_and_save_data(x_data, y_data, file_name = "Data", save_plot = False, save_data = False):
    df = pd.DataFrame({"data": y_data.tolist()}) # Assume data is given as a tensor
    if save_data: df.to_excel(f"{data_path}\{file_name}.xlsx") # Save to Excel Document

    plt.plot(x_data.detach(), y_data.detach()) # Addume data is given as a tensor
    if save_plot: plt.savefig(f"{plots_path}\{file_name}.jpg")
    plt.show()

def plot_multiple_data(xy_data, legend=None, file_name="Plot", save_plot=False, plots_path="."):
    # Iterate through the pairs, ignoring the 3 R_2 values at the end
    for i in range(0, len(xy_data) - 3, 2):
        x = xy_data[i].detach().squeeze()
        y = xy_data[i+1].detach().squeeze()
        
        # Sort values so the plot lines draw cleanly from left to right
        x_sorted, indices = torch.sort(x)
        y_sorted = y[indices]
        
        x_np = x_sorted.cpu().numpy()
        y_np = y_sorted.cpu().numpy()
        
        # FIX: Grab legend[i+1] to get the "Output" label instead of the "Input" label
        current_label = legend[i+1] if legend and (i+1) < len(legend) else None
        
        plt.plot(x_np, y_np, label=current_label) 
        
    if legend:
        plt.legend()
        
    if save_plot: 
        # Note: Ensure plots_path is defined or passed as an argument
        save_file = os.path.join(plots_path, f"{file_name}.jpg")
        plt.savefig(save_file)
        
    plt.show()

def save_data(y_data, legend = None, file_name = "Data"):
    df = pd.DataFrame({legend if legend else "Data": y_data.detach().numpy().flatten()})
    df.to_excel(f"{data_path}\{file_name}.xlsx") # Save to Excel Document

def save_multiple_data(y_data_array, legend = None, file_name = "Data"):
    data_formatted = {}
    for index, data_tensor in enumerate(y_data_array):
        data_name = legend[index] if legend and index < len(legend) else f"Data{index}"
        data_formatted[data_name] = data_tensor.detach().numpy().flatten()

    df = pd.DataFrame.from_dict(data_formatted, orient='index').T
    df.to_excel(f"{data_path}\{file_name}.xlsx") # Save to Excel Document

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