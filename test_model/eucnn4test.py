import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


X = np.load("./test_model/data/X_norm.npy")
y = np.load("./test_model/data/y_norm.npy")

device = torch.device("mps")
batch_size = 64

# transfer the array to tensor
X = torch.tensor(X.reshape(-1, 30, 5, 5), dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
dataset = TensorDataset(X, y)
generator = torch.Generator().manual_seed(42)
train_set, val_set = random_split(dataset, [0.8, 0.2], generator=generator)
print(f"num in trian: {len(train_set)}")
print(f"num in val: {len(val_set)}")

# build the data loader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)


class Simple_model(nn.Module):
    def __init__(self, input_size=30 * 5 * 5, loss_func=nn.CrossEntropyLoss(), lr=1e-3):
        super().__init__()

        self.device = torch.device("mps")

        self.conv = nn.Sequential(
            nn.Conv2d(30, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
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


num_epoches = 400
cnn_model = Simple_model()
# load the trained data
# cnn_model.load_state_dict(torch.load("./test_model/data/cnn_eu.pth"))
cnn_model.load_state_dict(torch.load("./test_model/data/cnn_norm_drop.pth"))

cnn_model.to(device)

# load the test
X_test = np.load("./test_model/data/X_eval_new.npy")
y_test = np.load("./test_model/data/y_eval_new.npy")

batch_size = 64
# transfer the array to tensor
X_test = torch.tensor(X_test.reshape(-1, 30, 5, 5), dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
dataset_test = TensorDataset(X_test, y_test)
print(f"the length of test: {len(X_test)}")
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

print("Begin to train")
cnn_model.eval()
correct = 0
total = 0

all_labels = []
all_preds = []
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = cnn_model.forward(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

    print(f"Accuracy of the model on the Val Set: {100 * correct / total:.2f}%")
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 7))
ax = sns.heatmap(cm, annot=False, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix of CNN(Validation Set)")
# plt.savefig(f"confusion_matrix_{dataset_name}.png")
plt.show()


# do the test on Test set
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = cnn_model.forward(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

    print(f"Accuracy of the model on the Test Set: {100 * correct / total:.2f}%")
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 7))
ax = sns.heatmap(cm, annot=False, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix of CNN(Test Set)")
# plt.savefig(f"confusion_matrix_{dataset_name}.png")
plt.show()


"""

writer = SummaryWriter("./test_model/runs/norm_800ep_withTest")
# tensorboard --logdir=runs

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

    print(f"Accuracy of the model on the val images: {100 * correct / total:.2f}%")
    writer.add_scalar("Validation Accuracy", 100 * correct / total, epoch)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = cnn_model.forward(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the model on the test images: {100 * correct / total:.2f}%")
    writer.add_scalar("Validation Accuracy", 100 * correct / total, epoch)


"""
