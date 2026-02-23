import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

df = pd.read_excel("CDM_data.xlsx", sheet_name="h=0.0001") # using most accurate step size

t_cdm = df.iloc[:,0].to_numpy()
y_cdm = df.iloc[:,1].to_numpy()


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
# Normalize time for better results
t_cdm = (t_cdm - t_cdm.min()) / (t_cdm.max() - t_cdm.min())


def plot_FFNN_solution(y_cdm, t_cdm, plot_type = 0):
    torch.manual_seed(321)
    x_full = torch.tensor(t_cdm, dtype=torch.float32).view(-1,1)


    indices = np.arange(len(y_cdm))
    np.random.shuffle(indices)
    train_end = int(0.7 * len(y_cdm))
    val_end = int(0.85 * len(y_cdm))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    x_train_data = t_cdm[train_idx]
    x_val_data = t_cdm[val_idx]
    x_test_data = t_cdm[test_idx]


    y_train_data = y_cdm[train_idx]
    y_val_data = y_cdm[val_idx]
    y_test_data = y_cdm[test_idx]
    
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

plt.plot(t_cdm, y_cdm, color='black', linewidth=2, label='CDM solution')
y_model = plot_FFNN_solution(y_cdm, t_cdm, plot_type=2)


plt.xlabel('t')
plt.ylabel('y(t)')
plt.grid()
plt.legend(loc='lower right')
file = "FFNN_CDM_test_graph"
plt.savefig("FFNN_CDM_test_graph", bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
plt.show()



