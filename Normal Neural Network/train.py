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

def plot_FFNN_solution(d, w0, y_benchmark):
    x = torch.linspace(0,1,500).view(-1,1)
    x_data = x[0:200:20]
    y_data = y_benchmark[0:200:20]

    model = FCN(1,1,32,3)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    
    for i in range(1500):
        optimizer.zero_grad()
        yh = model(x_data)
        loss = torch.mean((yh - y_data)**2)
        loss.backward()
        optimizer.step()

        
        if (i + 1) % 500 == 0: 
            yh = model(x).detach()
            plt.plot(x, yh, label=f"FFNN after {i + 1} epochs")
    return model(x)



plt.figure(figsize = (12, 8))

y_exact = plot_exact_solution(d, w0)
y_model = plot_FFNN_solution(d, w0, y_exact)

plt.ylim(-0.75, 1.05)
plt.xlabel('t')
plt.ylabel('f(t)')
plt.grid()
plt.legend(loc='lower right')
file = "finite_differences_step_sizes"
# plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
plt.show()