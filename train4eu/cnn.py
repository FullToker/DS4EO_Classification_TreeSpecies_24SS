import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter


X = np.load("../dataset/X_all_outnull.npy")
y = np.load("../dataset/y_all_outnull.npy")

X = np.load("./test_model/data/X_norm.npy")
y = np.load("./test_model/data/y_norm.npy")

device = torch.device("mps")
batch_size = 128

# transfer the array to tensor
X = torch.tensor(X.reshape(-1, 30, 5, 5), dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
dataset = TensorDataset(X, y)
generator = torch.Generator().manual_seed(42)
train_set, val_set = random_split(dataset, [0.9, 0.1], generator=generator)
print(f"num in trian: {len(train_set)}")
# print(f"num in val: {len(val_set)}")

# build the data loader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# 修正val_loader
X = np.load("./test_model/data/X_norm.npy")
y = np.load("./test_model/data/y_norm.npy")
device = torch.device("mps")

# transfer the array to tensor
X = torch.tensor(X.reshape(-1, 30, 5, 5), dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
dataset = TensorDataset(X, y)
generator = torch.Generator().manual_seed(42)
train_set, val_set = random_split(dataset, [0.7, 0.3], generator=generator)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
print(f"num in val: {len(val_set)}")


# class of cnn model
class Simple_model(nn.Module):
    def __init__(self, input_size=30 * 5 * 5, loss_func=nn.CrossEntropyLoss(), lr=1e-3):
        super().__init__()

        self.device = torch.device("mps")

        self.conv = nn.Sequential(
            nn.Conv2d(30, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 10, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(10 * 2 * 2, 10),
            # nn.ReLU(),
            # nn.Linear(20, 10),
            nn.Sigmoid(),
        )

        self.loss_func = loss_func
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def train_step(self, batch):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()

        output = self.forward(images)
        loss = self.loss_func(output, labels)
        loss.backward()
        self.optimizer.step()

        return loss


num_epoches = 200
cnn_model = Simple_model().to(device)

print("Begin to train")

writer = SummaryWriter("./train4eu/runs/norm_500ep_drop")
for epoch in range(num_epoches):
    current_loss = 0.0

    for i, batch in enumerate(train_loader):
        loss = cnn_model.train_step(batch)
        current_loss += loss

    print(
        f"Epoch [{epoch+1}/{num_epoches}], Loss: {current_loss / len(train_loader):.4f}"
    )
    writer.add_scalar("Training Loss", current_loss / len(train_loader), epoch)

    # use the val to evaluate
    cnn_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = cnn_model.forward(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the model on the test images: {100 * correct / total:.2f}%")
    writer.add_scalar("Validation Accuracy", 100 * correct / total, epoch)


torch.save(cnn_model.state_dict(), "./test_model/data/cnn_norm_drop.pth")
