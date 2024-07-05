import numpy as np
from sklearn.base import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from autoencoder import AutoEncoder, Encoder, Decoder

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("You are using the following device: ", device)

# load the dataset from NPY ----------------------/////////////////-------------////////
batch_size = 64
input = np.load("./data/np_data/X_ndvi.npy")
print(input.shape)
input = torch.tensor(input.reshape(-1, 33, 5, 5), dtype=torch.float32)
label = torch.tensor(np.load("./data/np_data/y_ndvi.npy"), dtype=torch.float32)

dataset = TensorDataset(input, label)
train_data, test_data = random_split(
    dataset, [0.85, 0.15], generator=torch.Generator().manual_seed(42)
)
print(f"shape of train : {len(train_data)}")
print(f"shape of test : {len(test_data)}")
torch.save(train_data, "./encoder/save/train.pth")
torch.save(test_data, "./encoder/save/test.pth")

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

encoder = Encoder().to(device)
decoder = Decoder().to(device)
autoencoder = AutoEncoder(encoder, decoder, batch_size).to(device)

epochs = 30
for epoch in range(epochs):
    total_loss = 0
    for batch in train_loader:
        loss = autoencoder.train_step(batch)
        total_loss += loss.item()
    print(f"the epoch {epoch + 1}/{epochs} loss: {total_loss}")


encoder_path = "./encoder/save/encoder.pth"
autoencoder_path = "./encoder/save/autoencoder.pth"
torch.save(encoder.state_dict(), encoder_path)
torch.save(autoencoder.state_dict(), autoencoder_path)

loss = 0
for batch in test_loader:
    images, labels = batch
    images, labels = images.to(device), labels.to(device)
    reconstruction = autoencoder.forward(images)
    func = nn.MSELoss()
    loss += func(reconstruction, images).item()
print(f"loss in test data: {loss}")
