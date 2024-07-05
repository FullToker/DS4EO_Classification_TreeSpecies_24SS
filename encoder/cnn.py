import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from autoencoder import Encoder

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("You are using the following device: ", device)
encoder = Encoder().to(device)
encoder.load_state_dict(torch.load("./encoder/save/encoder.pth"))

# load the dataset
train_data = torch.load("./encoder/save/train_ci.pth")
test_data = torch.load("./encoder/save/test_ci.pth")

train_loader = DataLoader(train_data, batch_size=128, shuffle=False)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)


# the classifier class
class clf(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        encoder = encoder.to(device)
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.encoder = encoder
        self.model = nn.Sequential(
            nn.Linear(5 * 5 * 16, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 10),
            nn.ReLU(),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)

    def forward(self, x):
        self.model = self.model.to(device)
        return self.model(x)

    def train_step(self, batch, loss_func=nn.CrossEntropyLoss()):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        images = self.encoded(images)

        flat = nn.Flatten()
        images = flat(images)
        outputs = self.forward(images)
        loss = loss_func(outputs, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def encoded(self, image):
        return self.encoder.forward(image)


# begin to classifier
model = clf(encoder)
num_epochs = 50

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, batch in enumerate(train_loader):
        loss = model.train_step(batch)
        running_loss += loss

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}"
    )

    model.eval()
    correct = 0
    total = 0
    flaten = nn.Flatten()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            encoded_images = model.encoded(images)
            encoded_images = flaten(encoded_images)

            outputs = model.forward(encoded_images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the model on the test images: {100 * correct / total:.2f}%")
