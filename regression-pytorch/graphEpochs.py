import pandas as pd
import matplotlib.pyplot as plt
import sys

files = sys.argv[1:]
plt.figure()

run_id = {
    '1': "Structure 2x64x64x1",
    '2': "Structure 2x64x32x1",
    '3': "Structure 2x64x32x16x1",
    '4': "Structure 2x64x32x16x8x1",
    '5': "Structure 2x64x32x16x8x4x1",
    '10': "lr = 0.01",
    '11': "lr = 0.05",
    '12': "lr = 0.1",
    '13': "lr = 0.3",
    '14': "lr = 0.001"
}

for path in files:
    df = pd.read_csv("EpochsData/epochsRangeLoss" + str(path) + ".txt")
    plt.plot(df["Epoch"], df["Loss"], label=f"Run {run_id[path]}")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss vs Epoch for Sigmoid Activation Function")
plt.grid(True)
plt.legend()
plt.show()