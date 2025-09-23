import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time

'''
input data is a numpy array with shape (10,5,1000,2,128,128):
10: 10 repeated trajectories with the same parameter but different initial conditions
5: 5 activity values (activity is the parameter)
1000: 1000 time frames
2: Q_xx and Q-xy
128x128: space grid
'''

# ---------------------------
# Dataset
# ---------------------------
class QFieldDataset(Dataset):
    """
    Dataset for Q-field snapshots.

    Parameters
    ----------
    data : np.ndarray
        Input shape (n_traj, n_param, n_time, 2, Nx, Ny).
        Will be reshaped into (N, 2, Nx, Ny).
    """
    def __init__(self, data: np.ndarray):
        # merge traj, param, time → total number of frames
        n_traj, n_param, n_time, n_chan, Nx, Ny = data.shape
        reshaped = data.reshape(-1, n_chan, Nx, Ny)  # (N, 2, 128, 128)
        self.data = reshaped.astype(np.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])

# ---------------------------
# Convolutional Autoencoder
# ---------------------------
class QFieldAutoencoder(nn.Module):
    def __init__(self, in_channels=2, latent_dim=128, filter_sizes=[32, 64, 128, 256], stride=2):
        super(QFieldAutoencoder, self).__init__()

        # -----------------------------
        # Encoder
        # -----------------------------
        enc_layers = []
        prev_channels = in_channels
        for fs in filter_sizes:
            enc_layers.append(
                nn.Conv2d(prev_channels, fs, kernel_size=3, stride=stride, padding=1)
            )
            enc_layers.append(nn.ReLU(True))
            prev_channels = fs
        self.encoder = nn.Sequential(*enc_layers)

        # Flatten
        last_spatial_size = 128 // (stride ** len(filter_sizes))
        print(last_spatial_size)
        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(filter_sizes[-1] * last_spatial_size * last_spatial_size,
                                latent_dim)

        # -----------------------------
        # Latent to feature map
        # -----------------------------
        self.fc_dec = nn.Linear(latent_dim,
                                filter_sizes[-1] * last_spatial_size * last_spatial_size)

        # -----------------------------
        # Decoder
        # -----------------------------
        dec_layers = []
        reversed_sizes = list(reversed(filter_sizes))
        for i in range(len(reversed_sizes) - 1):
            dec_layers.append(
                nn.ConvTranspose2d(
                    reversed_sizes[i],
                    reversed_sizes[i+1],
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    output_padding=1,
                )
            )
            dec_layers.append(nn.ReLU(True))

        # final layer: map back to input channels
        dec_layers.append(
            nn.ConvTranspose2d(
                reversed_sizes[-1],
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                output_padding=1,
            )
        )
        dec_layers.append(nn.Tanh())  # keep output in [-1,1]

        self.decoder = nn.Sequential(*dec_layers)

    # forward pass
    def encode(self, x):
        h = self.encoder(x)
        h = self.flatten(h)
        z = self.fc_enc(h)
        return z

    def decode(self, z, stride=2):
        h = self.fc_dec(z)
        last_spatial_size = 128 // (stride ** (len(self.encoder) // 2))
        h = h.view(-1, self.decoder[0].in_channels,
                   last_spatial_size, last_spatial_size)
        x_rec = self.decoder(h)
        return x_rec

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

# ---------------------------
# Training Loop
# ---------------------------
def train_autoencoder(qfield_data, epochs=20, batch_size=32, lr=1e-3, latent_dim=64, 
                      filter_sizes=[32, 64, 128, 256], stride=2):
    
    dataset = QFieldDataset(qfield_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QFieldAutoencoder(in_channels=2, latent_dim=latent_dim, 
                              filter_sizes=filter_sizes, stride=stride).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    return model

start = time.time()
q_data = np.load("data/nematic_data.npy")
print(time.time()-start)
start = time.time()
q_data = q_data.astype(np.float16)
print(time.time()-start)

filter_sizes = [16, 32, 64, 128]
stride = 2

model = train_autoencoder(q_data, epochs=50, batch_size=64, lr=1e-3, latent_dim=128, 
                          filter_sizes=filter_sizes, stride=stride)

# extract latent representations
dataset = QFieldDataset(q_data)
loader = DataLoader(dataset, batch_size=128)
latents = []
with torch.no_grad():
    for index, batch in enumerate(loader):
        start = time.time()
        batch = batch.cuda()
        z = model.encode(batch)
        latents.append(z.cpu().numpy())
        print(time.time()-start)
latents = np.concatenate(latents, axis=0) 
