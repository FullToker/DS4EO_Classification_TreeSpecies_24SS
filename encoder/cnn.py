import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from autoencoder import Encoder

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("You are using the following device: ", device)
encoder = Encoder().to(device)
encoder.load_state_dict(torch.load("./encoder/save/encoder.pth"))

# load the dataset
train_data = torch.load("./encoder/save/train_argu.pth")
test_data = torch.load("./encoder/save/test_argu.pth")

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

for data in train_loader:
    data = data.to(device)
for data in test_loader:
    data = data.to(device)


# the classifier class
class clf(nn.Module):
    def __init__(self, encoded):
        super().__init__()
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        self.model = nn.Sequential(
            nn.Linear(encoded, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 10),
            nn.Softmax(),
        )

    def forward(self, x):
        return self.model(x)
