import pandas as pd
import matplotlib.pyplot as plt
import sys

path = sys.argv[1]
plt.figure()

labeling_text = ""
end_labeling_text = ""
if path == "learning_rate_data":
    title = "Training Loss vs. Epochs for Different Learning Rates"
    labeling_text = "lr = "
elif path == "ascending_structures_data":
    title = "Training Loss vs. Epochs for Different Structures"
    labeling_text = "Structure "
elif path == "descending_structures_data":
    title = "Training Loss vs. Epochs for Different Structures"
    labeling_text = "Structure "
elif path == "activation_functions_data":
    title = "Training Loss vs. Epochs for Different Activation Functions"
elif path == "optimizers_data":
    title = "Training Loss vs. Epochs for Different Optimizers"
elif path == "losses_data":
    title = "Training Loss vs. Epochs for Different Loss Equations"
elif path == "data_sizes_data":
    title = "Training Loss vs. Epochs for Different Data Sizes"
    end_labeling_text = " random points"
    
df = pd.read_csv("EpochsData/" + str(path) + ".txt")
for data in [c for c in df.columns][1:]:
    plt.plot(df["Epoch"], df[data], label=f"{labeling_text}{data}{end_labeling_text}")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(title)
plt.grid(True)
plt.legend()
plt.savefig(f"plots/{path}.png", dpi=300, bbox_inches="tight")
plt.show()