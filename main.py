import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sys

from Neural_Network_Setups.Classes.FFNN import *
from Neural_Network_Setups.model_training import *

from Helper_Functions.plotting_and_saving import *
from Helper_Functions.sorting_and_retrieving import *
from Finite_Differences.finite_differences_method import *

d, w0, s0 = 2, 20, [1, 0] #Initial Conditions

def main():
    model = FCN(1, 1, 256, 3)

    x_data, y_data = get_data_from_folder("Data/Central_Diff_0.0001/")

    all_parameters = read_parameters()
    x_output, y_outputs, legend_outputs = train_model(model, x_data, y_data, all_parameters, save_every_epoch_interval = 500)
    save_multiple_data(x_output, y_outputs, legend = legend_outputs, file_name="Testing")

if __name__ == "__main__":
    main()