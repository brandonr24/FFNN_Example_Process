############################################################
# Install the following dependencies:
# > pip install numpy pandas scikit-learn torch
############################################################

import pandas as pd
import numpy as np
import math, random
from sklearn.model_selection import train_test_split
import torch
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
from   torch.utils.data import DataLoader, TensorDataset

## check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

############################################################
# The CSV specifies the function to predict,
# It contains 3 columns:
# - column 1: `x1`, float between -5.0 & 5.0
# - column 2: `x2`, float between -5.0 & 5.0
# - column 3: `y`,  float y = x2*sin(x1) - x1*cos(x2)
############################################################

df = pd.read_csv("sample_data.csv")

X = df[["x1", "x2"]].values   # the input columns
y = df[["y"]].values          # the output column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# convert to PyTorch tensors and move to GPU if available
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# create PyTorch datasets and dataloaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)

############################################################
# Model definition
############################################################

class FFNN(nn.Module):
    def __init__(self):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, 1)
        # self.fc4 = nn.Linear(16, 8)
        # self.fc5 = nn.Linear(8, 4)
        # self.fc6 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        # x = self.tanh(self.fc3(x))
        # x = self.tanh(self.fc4(x))
        # x = self.tanh(self.fc5(x))
        x = self.fc3(x)
        return x
    
def quantile_loss(q):
    def loss_fn(pred, target):
        e = target - pred
        return torch.mean(torch.maximum(q * e, (q - 1) * e))
    return loss_fn

def mape_loss(pred, target, eps=1e-6):
    return torch.mean(torch.abs((target - pred) / (target + eps)))

def log_cosh_loss(pred, target):
    diff = pred - target
    return torch.mean(torch.log(torch.cosh(diff)))

############################################################
# Training
############################################################

totalEpochs = 500
run_mode = 2    # 1: train & save the model into a folder
                # 2: train but don't save the model
                # 3: load the saved model

model = None

dataSlot = 1
while True:
    path = Path("EpochsData\epochsRangeLoss" + str(dataSlot) + ".txt")
    if not path.exists():
        break
    dataSlot += 1


propertyRangeList = ["1000", "2000", "3000", "4000", "5000", "6000", "7000", "8000", "9000"]
propertyAppend = []
for changingProperty in propertyRangeList:
    df = pd.read_csv(f"modified_data\{changingProperty}_points_data.csv")

    X = df[["x1", "x2"]].values   # the input columns
    y = df[["y"]].values          # the output column

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # convert to PyTorch tensors and move to GPU if available
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    # create PyTorch datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)
    ## instantiate the model and move it to GPU if available
    model = FFNN().to(device)
    if run_mode == 1 or run_mode == 2:
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        for epoch in range(totalEpochs):  # number of epochs
            model.train()
            running_loss = 0.0
            for i, (inputs, targets) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if len(propertyAppend) == epoch:
                propertyAppend.append(f"{running_loss / len(train_loader):.4f}")
            else:
                propertyAppend[epoch] = f"{propertyAppend[epoch]},{running_loss / len(train_loader):.4f}"

            #print(f"Epoch [{epoch + 1}/{totalEpochs}], Loss: {propertyAppend[epoch]}")

        if run_mode == 1:
            torch.save(model.state_dict(), 'xy_model.pth')  # save the model

        print(f"Training completed for property: {changingProperty}")

    elif run_mode == 3:
        print("\nLoading stored model...")
        model.load_state_dict(torch.load('xy_model.pth', map_location=device))
        model.eval()


with open(r"EpochsData\data_sizes_data.txt", "w") as f:
    f.write(f"Epoch,{','.join(str(x) for x in propertyRangeList)}\n")
    for epoch in range(totalEpochs):
        f.write(str(epoch + 1) + "," + str(propertyAppend[epoch]) + "\n")

############################################################
# Show model performance
############################################################

model.eval()
test_loss = 0.0
criterion = nn.MSELoss()
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()

print("\nModel performance:")
print(f"> Loss on test set: {test_loss / len(test_loader):.3f}")

############################################################
# Further testing
############################################################

def rand_input():
    '''input range of the function'''
    return random.random() * 10 - 5

def function_to_predict(x1, x2):
    '''the function to predict'''
    return x2 * math.sin(x1) - x1 * math.cos(x2)

x_hat = []
y_true = []
for _ in range(5):
    x1, x2 = rand_input(), rand_input()
    x_hat.append([x1, x2])
    y_true.append(function_to_predict(x1, x2))

## convert to tensor and move to GPU if available
x_hat_tensor = torch.tensor(x_hat, dtype=torch.float32).to(device)
y_hat_tensor = model(x_hat_tensor).cpu().detach().numpy()  # move back to CPU for printing

print("\nMore test results using random inputs:")
for i in range(len(y_true)):
    print(f"> x1,x2 = {x_hat[i][0]:+.2f}, {x_hat[i][1]:+.2f}; ", end='')
    print(f"predicted = {y_hat_tensor[i][0]:+.2f}; actual={y_true[i]:+.2f}; "
          f"diff={abs(y_true[i] - y_hat_tensor[i][0]):.2f}")