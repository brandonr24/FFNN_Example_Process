import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sys

from Neural_Network_Setups.Classes.FFNN import FCN
from Neural_Network_Setups.model_training import train_model, oscillator

from Helper_Functions.plotting_and_saving import plot_and_save_model
from Finite_Differences.forward_difference_method import euler_method

def main():
    model = FCN(1,1,32,3)
    d, w0, s0 = 2, 20, [1, 0] #Initial Conditions

    x_data = torch.linspace(0,1,500).view(-1,1)
    y_data = euler_method(d, w0, s0, 0.002)

    train_model(model, x_data, y_data, optim = "Adam", epochs = 9000)
    plot_and_save_model(x_data, model(x_data), "Example")

if __name__ == "__main__":
    main()