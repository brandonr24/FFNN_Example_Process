import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from Neural_Network_Setups.parameters import params

optimizer_function_map = {
    "adadelta": "Adadelta",
    "adafactor": "Adafactor",
    "adagrad": "Adagrad",
    "adam": "Adam",
    "adamw": "AdamW",
    "sparseadam": "SparseAdam",
    "adamax": "Adamax",
    "asgd": "ASGD",
    "lbfgs": "LBFGS",
    "muon": "Muon",
    "nadam": "NAdam",
    "adam": "RAdam",
    "rmsprop": "RMSprop",
    "rpop": "RProp",
    "sgd": "SGD",
}

def read_parameters():
    params_all_lower = {}
    for next_param in params.keys():
        params_all_lower[next_param.lower()] = \
            params[next_param].lower() if isinstance(params[next_param], str) else params[next_param]

    print(f"Given Paramters: {params_all_lower}")
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
    try:
        optimizer_class = getattr(optim, optimizer_function_map[optim_choice])
    except AttributeError:
        print("ERROR: Given optimizer was unable to be interpreted, defaulting to Adam")
        return torch.optim.Adam(model.parameters(), lr = lr) # Default to Adam if Given Param Wasn't Valid
    
    return optimizer_class(model.parameters(), lr = lr)

def calculate_r2(y_pred, y_true):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2.item()

def train_model(model, x_data, y_data, model_params, save_every_epoch_interval = -1):
    x_test, y_test = x_data[0], y_data[0]
    x_train, y_train = x_data[1], y_data[1]
    x_val, y_val = x_data[2], y_data[2]

    d, w0 = 2, 20
    x = torch.linspace(0,1,500).view(-1,1)
    y = oscillator(d, w0, x).view(-1,1)

    optimizer = choose_optimizer(model, model_params["optimizer"] if "optimizer" in model_params else "adam", lr = 1e-3)
    y_outputs, legend_outputs = [], []

    for i in range(model_params["epochs"] if "epochs" in model_params else 1000):
        optimizer.zero_grad()
        yh = model(x_train)
        loss = torch.mean((yh-y_train)**2)# use mean squared error
        loss.backward()
        optimizer.step()

        if save_every_epoch_interval != -1 and not (i + 1) % save_every_epoch_interval:
            y_outputs.append(model(x_train))
            legend_outputs.append(f"Train Epoch {i + 1}")
            y_outputs.append(model(x_test))
            legend_outputs.append(f"Test Epoch {i + 1}")
            y_outputs.append(model(x_val))
            legend_outputs.append(f"Val Epoch {i + 1}")

    y_outputs.append(torch.tensor([calculate_r2(model(x_train), y_train)]))
    legend_outputs.append("Final Train R_2")
    y_outputs.append(torch.tensor([calculate_r2(model(x_test), y_test)]))
    legend_outputs.append("Final Test R_2")
    y_outputs.append(torch.tensor([calculate_r2(model(x_val), y_val)]))
    legend_outputs.append("Final Val R_2")

    return y_outputs, legend_outputs