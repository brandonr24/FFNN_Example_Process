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
    # Called to save new training data, not needed most of the time
    y_data = euler_method_central(d, w0, s0, 0.0001) # Example of new y_data
    x_data = torch.linspace(0, 1, len(y_data)).view(-1, 1) # Example of new corresponding x_data
    save_training_data(x_data, y_data, file_name = "Central_Diff_0.0001/")

if __name__ == "__main__":
    main()