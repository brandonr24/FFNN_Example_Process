import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from   torch.utils.data import DataLoader, TensorDataset

# Define the function
def f(z):
    return z**2 + 2*z - 1j

def predict_on_grid(model, z, device="cpu"):
    # Build (N, 2) points in float32 in one go (fast, no Python loop)
    pts = np.column_stack((z.ravel())).astype(np.float32)  # shape (N, 2)

    # Convert once to a tensor on the right device
    t = torch.from_numpy(pts).to(device)  # (N, 2)

    model.eval()
    with torch.no_grad():
        y = model(t).squeeze(1).cpu().numpy()  # (N,)

    # Reshape back to the mesh shape
    return y.reshape(X1.shape)

# Create a grid of x1 and x2 values
x1 = np.linspace(-5, 5, 200)
x2 = np.linspace(-5, 5, 200)
X1, X2 = np.meshgrid(x1, x2)

#Grab current iteration of the model
class FFNN(nn.Module):
    def __init__(self):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(2, 6)
        self.fc2 = nn.Linear(6, 12)
        self.fc3 = nn.Linear(12, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FFNN().to(device)

model.load_state_dict(torch.load('xy_model.pth', map_location=device))

# Compute the function values once
Z = f(X1, X2)
Z_computed = predict_on_grid(model, X1, X2)

def make_figure(X1, X2, Z, plotTitle):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Surface plot
    surf = ax.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='none')

    # Labels and title
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')
    ax.set_title(plotTitle)

    # Fix the viewpoint so both figures look exactly the same
    ax.view_init(elev=30, azim=-60)

    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    return fig, ax

# Make two identical figures
make_figure(X1, X2, Z, r'$f(z) = z^2 + 2z - j$')
model_fig, model_ax = make_figure(X1, X2, Z_computed, r'Modeled $f(z)$ with FFNN Tanh for 64 and Tanh for 64 hidden layers')
model_fig.savefig('Plots/FFNN2_V1.png', dpi=300, bbox_inches='tight', transparent=True)

plt.show()
