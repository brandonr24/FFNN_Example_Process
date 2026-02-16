import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Analytical Solution
def analytical(d, w0, x):
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
    return np.array([s_i[1], (-2*d*s_i[1]-w0**2*s_i[0])]) # split into two first-order ODEs


# Central Difference Method
h_ranges = [0.01,0.005,0.001,0.0005,0.0001]
d, w0 = 2, 20
s0 = [1, 0] # 1st initial condition


def central_diff_method(h):
    t = np.arange(0, 1 + h, h) # creating array from 0 to 1 and all x values between step size

    s = np.zeros((len(t), 2))
    s[0] = s0 # 0-th column

    s[1] = [s[0][j] + h*f(t[0], s[0])[j] for j in range(2)] # 2nd initial condition

    for i in range(1, len(t) - 1):
        s[i + 1] = s[i - 1] + 2*h*f(t[i], s[i])
        
    plt.plot(t, s[:, 0] , '--', label=f'Approximate With {h} Step Size')

plt.figure(figsize = (12, 8))
for k in h_ranges:
    central_diff_method(k)
x = torch.linspace(0,1,500).view(-1,1)
y = analytical(d, w0, x).view(-1,1)
plt.plot(x, y, 'black', label='Analytical')
plt.ylim(-0.75, 1.05)
plt.xlabel('t')
plt.ylabel('f(t)')
plt.grid()
plt.legend(loc='lower right')
file = "finite_differences_step_sizes"
# plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
plt.show()