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

from main import d, w0, s0

def main():
    x_data, y_data = get_data_from_folder("Data/Central_Diff_0.0001/")

    plt.plot(x_data[0], oscillator(d, w0, x_data[0]), label = "computed")
    plt.plot(x_data[0], y_data[0], label = "numerical")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()