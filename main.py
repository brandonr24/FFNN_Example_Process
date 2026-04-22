import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
import sys

from Neural_Network_Setups.Classes.FFNN import *
from Neural_Network_Setups.model_training import *

from Helper_Functions.plotting_and_saving import *
from Helper_Functions.sorting_and_retrieving import *
from Finite_Differences.finite_differences_method import *

GAMMA = 0.5
OMEGA0 = 2.0
X0 = 1.0
V0 = 0.0
d, w0, s0 = 2, 20, [1, 0] #Initial Conditions

def main():
    model = PINN()
    optimizer = optim.LBFGS(model.parameters(), lr=0.1)  # Lower learning rate
    scheduler = StepLR(optimizer, step_size=300, gamma=0.001)

    x_data, y_data = get_data_from_folder("Data/Central_Diff_0.0001/")

    train_pinn(model, optimizer, scheduler, GAMMA, OMEGA0, X0, V0, epochs=1000)
    model.eval()
    t_plot = torch.linspace(0, 10, 500).view(-1, 1) # Plotting just the first second to match your graphs
    with torch.no_grad():
        x_plot = model(t_plot)
    x_true = oscillator(GAMMA, OMEGA0, t_plot)


    N = x_plot.shape[0]

    train_size = int(0.70 * N)
    val_size = int(0.15 * N)
    test_size = N - train_size - val_size

    indices = torch.randperm(N)

    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    test_idx = indices[train_size + val_size :]

    x_plot_train = torch.full_like(x_plot, float('nan'))
    x_plot_val = torch.full_like(x_plot, float('nan'))
    x_plot_test = torch.full_like(x_plot, float('nan'))

    x_plot_train[train_idx] = x_plot[train_idx]
    x_plot_val[val_idx] = x_plot[val_idx]
    x_plot_test[test_idx] = x_plot[test_idx]

    x_true_train = torch.full_like(x_true, float('nan'))
    x_true_val   = torch.full_like(x_true, float('nan'))
    x_true_test  = torch.full_like(x_true, float('nan'))

    x_true_train[train_idx] = x_true[train_idx]
    x_true_val[val_idx]     = x_true[val_idx]
    x_true_test[test_idx]   = x_true[test_idx]

    df = pd.DataFrame({
        't_plot': t_plot.flatten().numpy(),
        'true_train': x_true_train.flatten().numpy(),
        'pred_train': x_plot_train.flatten().numpy(),
        'true_val': x_true_val.flatten().numpy(),
        'pred_val': x_plot_val.flatten().numpy(),
        'true_test': x_true_test.flatten().numpy(),
        'pred_test': x_plot_test.flatten().numpy()
    })

    csv_filename = "pinn_predictions_split.csv"
    df.to_csv(csv_filename, index=False)

    print(f"Successfully saved 7-column data to {csv_filename}")

    # plt.figure(figsize=(8, 5))

    # # Plot the true computed oscillator as a solid background line
    # plt.plot(t_plot, oscillator(GAMMA, OMEGA0, t_plot), label="Computed (True)", color="black", alpha=0.4, linewidth=2)

    # # Plot the splits using dot markers ('.') so lines don't zigzag across the random gaps
    # plt.plot(t_plot.numpy(), x_plot_train.numpy(), '.', label="Train (70%)", color="green", markersize=5)
    # plt.plot(t_plot.numpy(), x_plot_val.numpy(), '.', label="Val (15%)", color="blue", markersize=5)
    # plt.plot(t_plot.numpy(), x_plot_test.numpy(), '.', label="Test (15%)", color="red", markersize=5)

    # plt.title(f"Damped Oscillator (No Fourier) - ω0={OMEGA0}, γ={GAMMA}")
    # plt.xlabel("Time (t)")
    # plt.ylabel("Position (x)")
    # plt.legend()
    # plt.grid(True)
    # plt.show()







    # plot_multiple_data(x_output, y_outputs, legend = legend_outputs, save_plot=True, file_name="Testing")
    # save_multiple_data(x_output, y_outputs, legend = legend_outputs, save_plot=True, file_name="mainFile")

if __name__ == "__main__":
    main()