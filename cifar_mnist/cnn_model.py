import torch
import torch.nn as nn
import torch.nn.functional as F

# define encoder, decoder and classifier
kernel_size = 4 # (4, 4) kernel
init_channels = 32 # initial number of filters
image_channels = 3
hidden = 256

class Encoder(nn.Module):
    def __init__(self, common_dim, salient_dim, hidden):
        super(Encoder, self).__init__()

        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size, stride=2, padding=1)
        self.enc2 = nn.Conv2d(
            in_channels=init_channels, out_channels=init_channels, kernel_size=kernel_size, stride=2, padding=1)
        self.enc3 = nn.Conv2d(
            in_channels=init_channels, out_channels=init_channels, kernel_size=kernel_size, stride=2, padding=1)
        self.enc4 = nn.Conv2d(
            in_channels=init_channels, out_channels=256, kernel_size=kernel_size, stride=2, padding=0)
        # fully connected layers for learning representations
        self.fc1 = nn.Linear(hidden, hidden)

        self.fc_mu = nn.Linear(hidden, common_dim)
        self.fc_log_var = nn.Linear(hidden, common_dim)

        # encoder
        self.specific_enc1 = nn.Conv2d(
            in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size, stride=2, padding=1)
        self.specific_enc2 = nn.Conv2d(
            in_channels=init_channels, out_channels=init_channels, kernel_size=kernel_size, stride=2, padding=1)
        self.specific_enc3 = nn.Conv2d(
            in_channels=init_channels, out_channels=init_channels, kernel_size=kernel_size, stride=2, padding=1)
        self.specific_enc4 = nn.Conv2d(
            in_channels=init_channels, out_channels=256, kernel_size=kernel_size, stride=2, padding=0)
        # fully connected layers for learning representations
        self.specific_fc1 = nn.Linear(hidden, hidden)

        self.specific_fc_mu = nn.Linear(hidden, salient_dim)
        self.specific_fc_log_var = nn.Linear(hidden, salient_dim)

    def forward(self, x, _return_activations=False):
        batch_size = x.size()[0]

        # encoding
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        h = F.relu(self.enc3(h))
        h = F.relu(self.enc4(h))
        batch, _, _, _ = x.shape
        h = F.adaptive_avg_pool2d(h, 1).reshape(batch, -1)
        hidden = F.relu(self.fc1(h))
        # get `mu` and `log_var`
        mean = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)

        # encoding
        h = F.relu(self.specific_enc1(x))
        h = F.relu(self.specific_enc2(h))
        h = F.relu(self.specific_enc3(h))
        h = F.relu(self.specific_enc4(h))
        batch, _, _, _ = x.shape
        h = F.adaptive_avg_pool2d(h, 1).reshape(batch, -1)
        hidden = F.relu(self.specific_fc1(h))
        # get `mu` and `log_var`
        specific_mean = self.specific_fc_mu(hidden)
        specific_logvar = self.specific_fc_log_var(hidden)

        return mean, log_var, specific_mean, specific_logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden):
        super(Decoder, self).__init__()

        self.fc2 = nn.Linear(latent_dim, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        # decoder
        self.dec1 = nn.ConvTranspose2d(
            in_channels=hidden, out_channels=init_channels, kernel_size=kernel_size,
            stride=2, padding=0
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_channels, out_channels=init_channels, kernel_size=kernel_size,
            stride=2, padding=1
        )
        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_channels, out_channels=init_channels, kernel_size=kernel_size,
            stride=2, padding=1
        )
        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_channels, out_channels=image_channels, kernel_size=kernel_size,
            stride=2, padding=1
        )

    def forward(self, latent):
        z = F.relu(self.fc2(latent))
        z = F.relu(self.fc3(z))
        z = z.view(-1, hidden, 1, 1)
        # decoding
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        reconstruction = torch.sigmoid(self.dec4(x))
        return reconstruction


class Classifier(nn.Module):
    def __init__(self, latent_dim):
        super(Classifier, self).__init__()
        self.z_dim = latent_dim
        self.fc = nn.Sequential(nn.Linear(self.z_dim, self.z_dim), nn.ReLU(), nn.Linear(self.z_dim, 1))
        # self.fc = nn.Linear(self.z_dim, 1)

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