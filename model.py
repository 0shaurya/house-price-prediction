import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time


LEARNING_RATE = 0.001
SPLIT = (70, 20, 10) # train, validation, test
NUM_EPOCHS = 200
SEED = 0
WEIGHT_DECAY = 0
BATCH_SIZE = 64

torch.manual_seed(SEED)
start_time = time.time()

df = pd.read_csv('boston.csv')

Xy = torch.tensor(df.values, dtype=torch.float32)
Xy = Xy[torch.randperm(Xy.shape[0])]
X, y = torch.split(Xy, [13, 1], dim=1)
NUM_INPUT_LAYERS = X.shape[1]


class RealEstateDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.y.shape[0]

split1 = int(X.shape[0]*SPLIT[0]/100)
split2 = int(X.shape[0]*(SPLIT[0] + SPLIT[1])/100)

mu, sigma = X[:split1].mean(), X[:split1].std()
X_train = (X[:split1] - mu) / sigma
X_valid = (X[split1:split2] - mu) / sigma
X_test = (X[split2:] - mu) / sigma

y_train = y[:split1]
y_valid = y[split1:split2]
y_test  = y[split2:]

train_dataset = RealEstateDataset(X_train, y_train)
valid_dataset = RealEstateDataset(X_valid, y_valid)
test_dataset = RealEstateDataset(X_test, y_test)

train_loader = DataLoader(dataset=train_dataset,
                        batch_size=BATCH_SIZE,
                        drop_last=False,
                        shuffle=True,
                        num_workers=0)
valid_loader = DataLoader(dataset=valid_dataset,
                        batch_size=BATCH_SIZE,
                        drop_last=False,
                        shuffle=True,
                        num_workers=0)
test_loader = DataLoader(dataset=test_dataset,
                        batch_size=BATCH_SIZE,
                        drop_last=False,
                        shuffle=True,
                        num_workers=0)



class network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(NUM_INPUT_LAYERS, 16),
            torch.nn.Linear(16, 20),
            torch.nn.Linear(20, 20),
            torch.nn.Linear(20, 16),
            torch.nn.Linear(16, 1))

    def forward(self, x):
        return self.model(x)


model = network()

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

start_time = time.time()
epochs = NUM_EPOCHS
losses = []
train_rmses = []
valid_rmses = []

for epoch in range(epochs):
    for (x_batch, y_batch) in train_loader:
        model.train()
        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        model.eval()

        logits_train = model(X_train)
        loss_train = loss_fn(logits_train, y_train)
        train_rmses.append(np.sqrt(loss_train.item()))

        logits_valid = model(X_valid)
        loss_valid = loss_fn(logits_valid, y_valid)
        valid_rmses.append(np.sqrt(loss_valid.item()))

        print(f"Epoch [{epoch+1}/{epochs}] | Train RMSE: {train_rmses[-1]:.4f} | Valid RMSE: {valid_rmses[-1]:.4f} | Time Elapsed: {(time.time() - start_time):.1f}s")

model.eval()
logits = model(X_test)
loss = loss_fn(logits, y_test)
print('Test Loss: ', np.sqrt(loss.item()))


plt.plot(train_rmses[1:], label='Training')
plt.plot(valid_rmses[1:], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.show()