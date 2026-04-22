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

    optimizer = choose_optimizer(model, model_params.get("optimizer", "adam"), lr = 1e-3)
    y_outputs, legend_outputs = [], []
    x_output = torch.cat((x_train, x_test, x_val), dim=0)

    for i in range(model_params["epochs"] if "epochs" in model_params else 1000):
        model.train()
        optimizer.zero_grad()
        yh = model(x_train)
        loss = torch.mean((yh-y_train)**2)# use mean squared error
        loss.backward()
        optimizer.step()

        if save_every_epoch_interval != -1 and not (i + 1) % save_every_epoch_interval:
            model.eval()
            with torch.no_grad():
                pred_train = model(x_train)
                pred_test = model(x_test)
                pred_val = model(x_val)

                y_outputs.extend([pred_train, pred_test, pred_val])
                
                y_outputs.extend([
                    pred_train - y_train,
                    pred_test - y_test,
                    pred_val - y_val
                ])
                
                # Save Legends
                legend_outputs.extend([
                    f"Train Epoch {i + 1}",
                    f"Test Epoch {i + 1}",
                    f"Val Epoch {i + 1}"
                ])

    y_outputs.append(torch.tensor([calculate_r2(model(x_train), y_train)]))
    legend_outputs.append("Final Train R_2")
    y_outputs.append(torch.tensor([calculate_r2(model(x_test), y_test)]))
    legend_outputs.append("Final Test R_2")
    y_outputs.append(torch.tensor([calculate_r2(model(x_val), y_val)]))
    legend_outputs.append("Final Val R_2")

    print(f"Finished Training for Paramters: {model_params}")
    return x_output, y_outputs, legend_outputs

def physics_loss(net, t, gamma, omega0):
    t.requires_grad = True
    x = net(t)
    dx_dt = torch.autograd.grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    d2x_dt2 = torch.autograd.grad(dx_dt, t, grad_outputs=torch.ones_like(dx_dt), create_graph=True)[0]
    residual = d2x_dt2 + 2 * gamma * dx_dt + omega0**2 * x
    return torch.mean(residual**2)

def initial_condition_loss(net, x0, v0):
    t0 = torch.tensor([0.0], requires_grad=True).float()
    x_pred = net(t0)
    dx_pred_dt = torch.autograd.grad(x_pred, t0, create_graph=True)[0]
    loss_ic = (x_pred - x0)**2 + (dx_pred_dt - v0)**2
    return loss_ic

def train_pinn(net, optimizer, scheduler, gamma, omega0, x0, v0, epochs=5000):
    for epoch in range(epochs):
        def closure():
            optimizer.zero_grad()
            t = torch.linspace(0, 10, 1000).view(-1, 1).float() 
            
            loss_physics = physics_loss(net, t, gamma, omega0)
            loss_ic = initial_condition_loss(net, x0, v0)
            loss_total = loss_physics + 10.0 * loss_ic
            loss_total.backward()
            return loss_total
            
        optimizer.step(closure)
        scheduler.step()
        
        if epoch % 100 == 0:
            loss_total = closure()
            print(f"Epoch {epoch}, Loss: {loss_total.item():.6f}")