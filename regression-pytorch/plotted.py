import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x1, x2):
    return x2 * np.sin(x1) - x1 * np.cos(x2)

# Create a grid of x1 and x2 values
x1 = np.linspace(-5, 5, 200)
x2 = np.linspace(-5, 5, 200)
X1, X2 = np.meshgrid(x1, x2)

# Compute the function values
Z = f(X1, X2)

# Plot the surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Surface plot
surf = ax.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='none')

# Labels and title
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.set_title(r'$f(x_1, x_2) = x_2 \sin(x_1) - x_1 \cos(x_2)$')

# Add color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

plt.show()
