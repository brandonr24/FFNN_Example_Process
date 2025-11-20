import csv
import random
import numpy as np

# Define the function
def f(x1, x2):
    return x2 * np.sin(x1) - x1 * np.cos(x2)

for pointsCount in range(1000, 10000, 1000):
    with open(f'modified_data/{pointsCount}_points_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x1','x2','y'])
        for i in range(pointsCount):
            new_x1, new_x2 = random.uniform(-5, 5), random.uniform(-5, 5)
            writer.writerow([new_x1, new_x2, f(new_x1, new_x2)])