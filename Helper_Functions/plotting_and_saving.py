import matplotlib.pyplot as plt
import pandas as pd
import numpy

data_path = "Results\Data"
plots_path = "Results\Plots"

def plot_and_save_data(x_data, y_data, file_name = "Data", save_plot = False, save_data = False):
    df = pd.DataFrame({"data": y_data.tolist()}) # Assume data is given as a tensor
    if save_data: df.to_excel(f"{data_path}\{file_name}.xlsx") # Save to Excel Document

    plt.plot(x_data.detach(), y_data.detach()) # Addume data is given as a tensor
    if save_plot: plt.savefig(f"{plots_path}\{file_name}.jpg")
    plt.show()

def plot_multiple_data(x_data, y_data, file_name = "Plot", save_plot = False):
    plt.plot(x_data.detach(), y_data.detach()) # Addume data is given as a tensor
    if save_plot: plt.savefig(f"{plots_path}\{file_name}.jpg")
    plt.show()

def save_multiple_data(y_data_array, legend = None, file_name = "Data"):
    data_formatted = {}
    for index, data_tensor in enumerate(y_data_array):
        data_name = legend[index] if legend and index < len(legend) else f"Data{index}"
        data_formatted[data_name] = data_tensor.detach().numpy().flatten()

    df = pd.DataFrame(data_formatted)
    df.to_excel(f"{data_path}\{file_name}.xlsx") # Save to Excel Document