import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd

def f(d, w0, t_i, s_i):
    return [s_i[1], (-2*d*s_i[1]-w0**2*s_i[0])]

def euler_method(d, w0, s0, h):
    t = np.arange(0, 1 + h, h)

    # Explicit Euler Method
    s = np.zeros((len(t), 2))
    s[0] = s0

    for i in range(0, len(t) - 1):
        s[i + 1] = [s[i][j] + h*f(d, w0, t[i], s[i])[j] for j in range(2)]
        
    return torch.from_numpy(s[:-1])