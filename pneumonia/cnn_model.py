import torch
import torch.nn as nn
import torch.nn.functional as F

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class Encoder(nn.Module):
    def __init__(self, background_latent_size, salient_latent_size):
        super(Encoder, self).__init__()

        # encoder
        hidden_dim = 256

        self.bg_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, hidden_dim, 4, 1),
            nn.ReLU(True),
            View((-1, hidden_dim * 1 * 1)),
            nn.Linear(hidden_dim, background_latent_size * 2),
        )

        self.bg_locs = nn.Linear(background_latent_size * 2, background_latent_size)
        self.bg_scales = nn.Linear(background_latent_size * 2, background_latent_size)

        self.tg_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, hidden_dim, 4, 1),
            nn.ReLU(True),
            View((-1, hidden_dim * 1 * 1)),
            nn.Linear(hidden_dim, salient_latent_size * 2),
        )

        self.tg_locs = nn.Linear(salient_latent_size * 2, salient_latent_size)
        self.tg_scales = nn.Linear(salient_latent_size * 2, salient_latent_size)

    def forward(self, x):
        hz = F.relu(self.bg_encoder(x))
        hs = F.relu(self.tg_encoder(x))

        return self.bg_locs(hz), self.bg_scales(hz), self.tg_locs(hs), self.tg_scales(hs)


class Decoder(nn.Module):
    def __init__(self, background_latent_size, salient_latent_size):
        super(Decoder, self).__init__()

        hidden_dim = 256
        self.decoder = nn.Sequential(
            nn.Linear(background_latent_size + salient_latent_size, hidden_dim),
            View((-1, hidden_dim, 1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim, 128, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.decoder(z)


class Classifier(nn.Module):
    def __init__(self, latent_dim):
        super(Classifier, self).__init__()
        self.z_dim = latent_dim
        self.fc = nn.Sequential(nn.Linear(self.z_dim, self.z_dim), nn.ReLU(), nn.Linear(self.z_dim, 1))

    def forward(self, latent):
        latent = latent.view(-1, self.z_dim)
        h = self.fc(latent)
        pred = torch.sigmoid(h)
        return pred

class FactorClassifier(nn.Module):
    def __init__(self, latent_dim):
        super(FactorClassifier, self).__init__()
        self.z_dim = latent_dim
        self.fc = nn.Sequential(nn.Linear(self.z_dim, self.z_dim), nn.ReLU(), nn.Linear(self.z_dim, self.z_dim), nn.ReLU(), nn.Linear(self.z_dim, 1))

    def forward(self, latent):
        latent = latent.view(-1, self.z_dim)
        h = self.fc(latent)
        pred = torch.sigmoid(h)
        return pred
