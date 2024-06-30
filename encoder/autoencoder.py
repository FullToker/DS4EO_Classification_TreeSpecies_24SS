import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size=5 * 5 * 33):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(33, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, output_size=5 * 5 * 33):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 33, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, batch_size):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.size = batch_size

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        latent = self.encoder.forward(x)
        reconstruction = self.decoder.forward(latent)
        return reconstruction

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def train_step(self, batch, loss_func=nn.MSELoss()):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        flatten_images = images
        self.optimizer.zero_grad()
        reconstruction = self.forward(flatten_images)
        loss = loss_func(reconstruction, flatten_images)
        loss.backward()
        self.optimizer.step()

        return loss
