import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

d, w0 = 2, 20
s0 = [1, 0] # Initial Condition

class FCN(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

def plot_exact_solution(d, w0):
    x = torch.linspace(0,1,500).view(-1,1)

    assert d < w0
    w = np.sqrt(w0**2-d**2)
    phi = np.arctan(-d/w)
    A = 1/(2*np.cos(phi))
    cos = torch.cos(phi+w*x)
    sin = torch.sin(phi+w*x)
    exp = torch.exp(-d*x)
    y  = exp*2*A*cos
    
    plt.plot(x, y, 'black', label='Exact')
    return y

def plot_FFNN_solution(d, w0, y_benchmark, plot_type = 0):
    torch.manual_seed(321)
    x_full = torch.linspace(0, 1, 500).view(-1, 1)

    indices = np.arange(len(y_benchmark))
    np.random.shuffle(indices)
    train_end = int(0.7 * len(y_benchmark))
    val_end = int(0.85 * len(y_benchmark))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    x_train_data = train_idx / (len(y_benchmark) - 1)
    x_val_data = val_idx / (len(y_benchmark) - 1)
    x_test_data = test_idx / (len(y_benchmark) - 1)

    y_train_data = y_benchmark[train_idx]
    y_val_data = y_benchmark[val_idx]
    y_test_data = y_benchmark[test_idx]
    
    x_train_tensor = torch.tensor(x_train_data, dtype=torch.float32).view(-1, 1)
    y_train_tensor = torch.tensor(y_train_data, dtype=torch.float32).view(-1, 1)
    sort_idx = np.argsort(x_train_data)
    x_train_sorted = torch.tensor(x_train_data[sort_idx], dtype=torch.float32).view(-1, 1)
    
    x_val_tensor = torch.tensor(x_val_data, dtype=torch.float32).view(-1, 1)
    y_val_tensor = torch.tensor(y_val_data, dtype=torch.float32).view(-1, 1)
    sort_idx = np.argsort(x_val_data)
    x_val_sorted = torch.tensor(x_val_data[sort_idx], dtype=torch.float32).view(-1, 1)
    
    x_test_tensor = torch.tensor(x_test_data, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test_data, dtype=torch.float32).view(-1, 1)
    sort_idx = np.argsort(x_test_data)
    x_test_sorted = torch.tensor(x_test_data[sort_idx], dtype=torch.float32).view(-1, 1)

    model = FCN(1,1,32,3)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    
    for i in range(2000):
        optimizer.zero_grad()
        yh = model(x_train_tensor)
        loss = torch.mean((yh - y_train_tensor)**2)
        loss.backward()
        optimizer.step()

        
        if (i + 1) % 500 == 0:
            model.eval()
            with torch.no_grad():
                if plot_type == 0:
                    yh_train = model(x_train_tensor)
                    v_loss = torch.mean((yh_train - y_train_tensor)**2)
                    y_train_pred = model(x_train_sorted).detach()
                    plt.plot(x_train_sorted.numpy(), y_train_pred.numpy(), label=f"FFNN Epoch {i + 1} with loss {v_loss.item():.5f}")
                elif plot_type == 1:
                    yh_val = model(x_val_tensor)
                    v_loss = torch.mean((yh_val - y_val_tensor)**2)
                    y_val_pred = model(x_val_sorted).detach()
                    plt.plot(x_val_sorted.numpy(), y_val_pred.numpy(), label=f"FFNN Epoch {i + 1} with loss {v_loss.item():.5f}")
                elif plot_type == 2:
                    yh_test = model(x_test_tensor)
                    v_loss = torch.mean((yh_test - y_test_tensor)**2)
                    y_test_pred = model(x_test_sorted).detach()
                    plt.plot(x_test_sorted.numpy(), y_test_pred.numpy(), label=f"FFNN Epoch {i + 1} with loss {v_loss.item():.5f}")
    return model(x_full)



plt.figure(figsize = (12, 8))

y_exact = plot_exact_solution(d, w0)
data_file = pd.read_excel('data.xlsx')
y_approx = data_file['step_size_0.0001'].to_numpy()
y_model = plot_FFNN_solution(d, w0, y_approx, plot_type = 2)

plt.xlabel('t')
plt.ylabel('f(t)')
plt.grid()
plt.legend(loc='lower right')
file = "FFNN_test_graph"
plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
plt.show()