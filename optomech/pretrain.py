import torch
from torch.utils.data import DataLoader, Dataset
import gym
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor, Normalize, Compose

# Define the convolutional autoencoder model
class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # Assuming 3-channel input
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, latent_dim)  # Adjust based on input image size
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 4 * 4),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Assuming pixel values are normalized between 0 and 1
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

# Custom dataset to sample from the gym environment
class OptomechDataset(Dataset):
    def __init__(self, env_name, num_samples, transform=None):
        self.env = gym.make(env_name)
        self.num_samples = num_samples
        self.transform = transform
        self.data = self._generate_data()

    def _generate_data(self):
        observations = []
        for _ in range(self.num_samples):
            obs = self.env.reset()
            if self.transform:
                obs = self.transform(obs)
            observations.append(obs)
        return torch.stack(observations)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Training function
def train_autoencoder(model, dataloader, criterion, optimizer, device, epochs=10):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed, _ = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")

# Main script
if __name__ == "__main__":
    # Parameters
    env_name = "YourOptomechGymEnv-v0"  # Replace with your gym environment name
    image_size = (3, 32, 32)  # Replace with the size of your environment's image observations
    latent_dim = 128
    num_samples = 10000
    batch_size = 64
    learning_rate = 1e-3
    epochs = 20
    export_path = "./encoder.pth"

    # Dataset and DataLoader
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])  # Normalize images
    dataset = OptomechDataset(env_name, num_samples, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, loss, and optimizer
    model = ConvAutoencoder(latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the autoencoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_autoencoder(model, dataloader, criterion, optimizer, device, epochs)

    # Export the encoder
    torch.save(model.encoder.state_dict(), export_path)
    print(f"Encoder saved to {export_path}")