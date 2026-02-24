import numpy as np
import pandas as pd
import torch
from Neural_Network_Setups.parameters import params

def read_parameters():
    params_all_lower = {}
    for next_param in params.keys():
        params_all_lower[next_param.lower()] = \
            params[next_param].lower() if isinstance(params[next_param], str) else params[next_param]

    return params_all_lower

def oscillator(d, w0, x):
    assert d < w0
    w = np.sqrt(w0**2-d**2)
    phi = np.arctan(-d/w)
    A = 1/(2*np.cos(phi))
    cos = torch.cos(phi+w*x)
    sin = torch.sin(phi+w*x)
    exp = torch.exp(-d*x)
    y  = exp*2*A*cos
    return y

def choose_optimizer(model, optim_choice, lr):
    if optim_choice.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr = lr)
    elif optim_choice.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr = lr)

    print("ERROR: Given optimizer was unable to be interpreted, defaulting to Adam")
    return torch.optim.Adam(model.parameters(), lr = lr) # Default to Adam if Given Param Wasn't Valid

def train_model(model, x_data, y_data, model_params, save_every_epoch_interval = -1):
    d, w0 = 2, 20
    x = torch.linspace(0,1,500).view(-1,1)
    y = oscillator(d, w0, x).view(-1,1)

    optimizer = choose_optimizer(model, model_params["optimizer"] if "optimizer" in model_params else "adam", lr = 1e-3)
    y_outputs, legend_outputs = [], []

    for i in range(model_params["epochs"] if "epochs" in model_params else 1000):
        optimizer.zero_grad()
        yh = model(x_data)
        loss = torch.mean((yh-y_data)**2)# use mean squared error
        loss.backward()
        optimizer.step()

        if save_every_epoch_interval != -1 and not (i + 1) % save_every_epoch_interval:
            y_outputs.append(model(x_data))
            legend_outputs.append(f"Epoch {i + 1}")

    return y_outputs, legend_outputs