import numpy as np
import torch

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

def choose_optimizer(model, optim_choice):
    if optim_choice == "Adam":
        return torch.optim.Adam(model.parameters(),lr=1e-3)

    return torch.optim.Adam(model.parameters(),lr=1e-3) # Default to Adam

def train_model(model, x_data, y_data, optim = "Adam", epochs = 1000):
    d, w0 = 2, 20
    x = torch.linspace(0,1,500).view(-1,1)
    y = oscillator(d, w0, x).view(-1,1)

    optimizer = choose_optimizer(model, optim)
    for i in range(epochs):
        optimizer.zero_grad()
        yh = model(x_data)
        loss = torch.mean((yh-y_data)**2)# use mean squared error
        loss.backward()
        optimizer.step()

    return model