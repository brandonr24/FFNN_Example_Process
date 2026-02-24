import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd

h_ranges = [0.01,0.005,0.001,0.0005,0.0001]
d, w0 = 2, 20
s0 = [1, 0] # Initial Condition

excel_data = []

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

def f(t_i, s_i):
    return [s_i[1], (-2*d*s_i[1]-w0**2*s_i[0])]

def euler_method(h):
    t = np.arange(0, 1 + h, h)

    # Explicit Euler Method
    s = np.zeros((len(t), 2))
    s[0] = s0

    for i in range(0, len(t) - 1):
        s[i + 1] = [s[i][j] + h*f(t[i], s[i])[j] for j in range(2)]
        
    plt.plot(t, [s[i][0] for i in range(len(s))] , '--', label=f'Approximate With {h} Step Size')
    excel_data.append(pd.Series(s[:, 0], name=f'step_size_{h}'))

plt.figure(figsize = (12, 8))

x = torch.linspace(0,1,500).view(-1,1)
y = oscillator(d, w0, x).view(-1,1)
plt.plot(x, y, 'black', label='Exact')
for k in h_ranges:
    euler_method(k)
    
df1 = pd.concat(excel_data, axis=1)
df1.to_excel("output.xlsx")
    
plt.ylim(-0.75, 1.05)
plt.xlabel('t')
plt.ylabel('f(t)')
plt.grid()
plt.legend(loc='lower right')
file = "finite_differences_step_sizes"
# plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
plt.show()