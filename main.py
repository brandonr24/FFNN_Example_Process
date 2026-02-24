import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sys

from Neural_Network_Setups.Classes.FFNN import *
from Neural_Network_Setups.model_training import *

from Helper_Functions.plotting_and_saving import *
from Finite_Differences.finite_differences_method import *

def main():
    model = FCN(1, 1, 32, 3)
    d, w0, s0 = 2, 20, [1, 0] #Initial Conditions

    y_data = euler_method_central(d, w0, s0, 0.0001)
    x_data = torch.linspace(0, 1, len(y_data)).view(-1, 1)

    y_outputs, legend_outputs = train_model(model, x_data, y_data, read_parameters(), save_every_epoch_interval = 100)
    save_multiple_data(y_outputs, legend = legend_outputs, file_name = "Example")

if __name__ == "__main__":
    main()