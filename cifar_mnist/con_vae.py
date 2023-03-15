# import functions
import numpy as np
import tensorflow.compat.v2 as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from cnn_model import Encoder, Decoder, Classifier, FactorClassifier
from dataset import CifarMNISTDataset

# Dataset de CIFAR
from keras.datasets import cifar10

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

# build datasets
(X_train_cifar, y_train_cif), (X_test_cifar, y_test_cif) = cifar10.load_data()
X_train_cifar = X_train_cifar[:50000]
X_test_cifar = X_test_cifar[:1000]

mnist = tf.keras.datasets.mnist
(X_train_mnist, y_train_mni), (X_test_mnist, y_test_mni) = mnist.load_data()
X_train_mnist = X_train_mnist[:50000]
X_train_mnist = np.array([np.pad(img, (2, 2)) for img in X_train_mnist])
X_test_mnist = X_test_mnist[:1000]
X_test_mnist = np.array([np.pad(img, (2, 2)) for img in X_test_mnist])

# build train dataset
X_train = []
y_train = []
y_train_mnist = []
y_train_cifar = []
for i in range(len(X_train_mnist)):
    if i < 25000:
        X_train.append(0.5 * X_train_cifar[i] / 255.)
        y_train.append(0)
        y_train_mnist.append(-1)
        y_train_cifar.append(y_train_cif[i][0])
    else:
        X_train.append((0.5 * X_train_cifar[i] + 0.5 * np.repeat(X_train_mnist[i][:, :, None], 3, axis=2)) / 255.)
        y_train.append(1)
        y_train_mnist.append(y_train_mni[i])
        y_train_cifar.append(y_train_cif[i][0])
X_train = np.array(X_train)
y_train = np.array(y_train)
y_train_mnist = np.array(y_train_mnist)
y_train_cifar = np.array(y_train_cifar)

# build test dataset
X_test = []
y_test = []
y_test_mnist = []
y_test_cifar = []
for i in range(len(X_test_mnist)):
    if i < 500:
        X_test.append(0.5 * X_test_cifar[i] / 255.)
        y_test.append(0)
        y_test_mnist.append(-1)
        y_test_cifar.append(y_test_cif[i])
    else:
        X_test.append((0.5 * X_test_cifar[i] + 0.5 * np.repeat(X_test_mnist[i][:, :, None], 3, axis=2)) / 255.)
        y_test.append(1)
        y_test_mnist.append(y_test_mni[i])
        y_test_cifar.append(y_test_cif[i])
X_test = np.array(X_test)
y_test = np.array(y_test)
y_test_mnist = np.array(y_test_mnist)
y_test_cifar = np.array(y_test_cifar)

# Instantiate Dataset and Data Loader
train_dataset = CifarMNISTDataset(X_train, y_train_cifar, y_train_mnist, y_train)
test_dataset = CifarMNISTDataset(X_test, y_test_cifar, y_test_mnist, y_test)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=512, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=512)

class ShallowDisVAE(nn.Module):
    """ The VAE architecture.
    """

    def __init__(self, latent_dim, hidden_dim=256, sigma_q=0.1):
        super(ShallowDisVAE, self).__init__()
        self.z_dim = latent_dim
        self.total_latent_size = latent_dim
        self.common_size = latent_dim // 2
        self.salient_size = latent_dim // 2
        self.h_dim = hidden_dim
        self.sigma_q = sigma_q
        self.encoder = Encoder(self.common_size, self.salient_size, hidden_dim)
        self.decoder = Decoder(latent_dim, hidden_dim)
        self.classifier = Classifier(self.common_size)

    def forward(self, x, y):
        device = x.get_device()
        batch_size = x.size()[0]

        # inference
        mean_qzx, logvar_qzx, specific_mean_qzx, specific_logvar_qzx = self.encoder(x)

        # common and specific latent vectors
        reparameterized_latent = torch.randn((batch_size, self.total_latent_size), device=device)

        # z latent space
        mean = torch.cat([mean_qzx, specific_mean_qzx], dim=1)
        logvar = torch.cat([logvar_qzx, specific_logvar_qzx], dim=1)
        std = (logvar * 0.5).exp()

        # sample latents
        z = mean + std * reparameterized_latent

        # classify on the latents
        y_pred = self.classifier(z[:, self.common_size:])

        # compute log probs for x
        z_zeroed = torch.clone(z)
        z_zeroed[y == 0, self.common_size:] = 0.0
        reconstructed_x = self.decoder(z_zeroed)

        return reconstructed_x, mean, logvar, y_pred, z

    def inference(self, x):
        # inference
        mean_qzx, logvar_qzx, specific_mean_qzx, specific_logvar_qzx = self.encoder(x)

        # z latent space
        mean = torch.cat([mean_qzx, specific_mean_qzx], dim=1)
        logvar = torch.cat([logvar_qzx, specific_logvar_qzx], dim=1)

        y_pred = self.classifier(specific_mean_qzx)

        return mean, logvar, y_pred

# hyperparameters
common_size = 64
salient_size = 64
alpha = 1/0.025
beta_c = 0.5
beta_s = 0.5
kappa = 1
gamma = 0 # 1e-10

## create loss function
bce = nn.BCELoss(reduction="sum")
def compute_loss(x_reconstructed, x, z_mean, z_log_var, y, y_pred):
    # elbo loss
    reconstruction_loss = (x[y==1] - x_reconstructed[y==1]).pow(2).sum()
    reconstruction_loss += (x[y==0] - x_reconstructed[y==0]).pow(2).sum()

    kl_div_loss = - beta_c*0.5 * torch.sum(1 + z_log_var[y==0, :common_size] - z_mean[y==0, :common_size].pow(2) - z_log_var[y==0, :common_size].exp(), dim=-1).sum()
    kl_div_loss += - beta_c*0.5 * torch.sum(1 + z_log_var[y==1, :common_size] - z_mean[y==1, :common_size].pow(2) - z_log_var[y==1, :common_size].exp(), dim=-1).sum()
    kl_div_loss += - beta_s * 0.5 * torch.sum(1 + z_log_var[y==1, common_size:] - z_mean[y==1, common_size:].pow(2) - z_log_var[y==1, common_size:].exp(), dim=-1).sum()
    # kl_div_loss += beta_s * alpha * 0.5 * z_mean[y==0, common_size:].pow(2).sum()

    clsf_loss = bce(y_pred, y[:, None])

    return reconstruction_loss, kl_div_loss, clsf_loss

def train(epoch, vae, optimizer, factor_optimizer):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _, _, y) in enumerate(train_loader):
        data = data.cuda()
        y = y.cuda()

        # independent optimizer training
        factor_optimizer.zero_grad()
        _, _, _, _, z = vae(data, y)

        # common classifiacation training
        y_pred_common = common_classifier(z[:, :common_size])
        fader_clsf_loss = bce(y_pred_common, y[:, None].float())

        # total correlation discriminator training
        joint_predictions = factor_classifier(z)
        product_of_marginals_predictions = factor_classifier(torch.cat((z[:, :common_size], torch.cat((z[1:, common_size:], z[0, common_size:][None]), dim=0)), dim=1))
        factor_input = torch.cat((joint_predictions[:, 0], product_of_marginals_predictions[:, 0]), dim=0)
        factor_target = torch.cat((torch.ones_like(joint_predictions[:, 0]), torch.zeros_like(product_of_marginals_predictions[:, 0])), dim=0)
        factor_clsf_loss = bce(factor_input, factor_target)

        # parameters update
        loss = factor_clsf_loss + fader_clsf_loss
        loss.backward()
        factor_optimizer.step()

        # training of the vae model
        optimizer.zero_grad()
        reconstructed_x, z_mean, z_log_var, y_pred, z = vae(data, y)
        reconstruction_loss, kl_div_loss, clsf_loss = compute_loss(reconstructed_x, data, z_mean, z_log_var, y.float(), y_pred)

        joint_predictions = factor_classifier(z)
        factor_clsf_loss = F.relu(torch.log(joint_predictions / (1 - joint_predictions))).sum()

        # parameters training
        loss = reconstruction_loss + kl_div_loss + kappa*factor_clsf_loss + gamma*clsf_loss
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

def test(vae):
    y_s = []
    y_preds = []
    y_pred_commons = []
    factor_s = []
    factor_pred = []
    vae.eval()
    with torch.no_grad():
        for data, _, _, y in test_loader:
            data = data.cuda()
            y = y.cuda()

            reconstructed_x, z_mean, z_log_var, y_pred, z = vae(data, y)

            y_pred_common = common_classifier(z_mean[:, :common_size])
            y_pred_commons.append(y_pred_common[:, 0].round().int())

            joint_predictions = factor_classifier(z_mean)
            product_of_marginals_predictions = factor_classifier(torch.cat((z_mean[:, :common_size], torch.cat((z_mean[1:, common_size:], z_mean[0, common_size:][None]), dim=0)), dim=1))
            factor_pred.append(joint_predictions[:, 0].round().int())
            factor_pred.append(product_of_marginals_predictions[:, 0].round().int())
            factor_s.append(torch.ones_like(joint_predictions[:, 0]))
            factor_s.append(torch.zeros_like(joint_predictions[:, 0]))

            y_preds.append(y_pred[:, 0].round().int())
            y_s.append(y.int())

    y_preds = torch.cat(y_preds, dim=0).cpu().numpy()
    y_s = torch.cat(y_s, dim=0).cpu().numpy()
    bacc = roc_auc_score(y_s, y_preds)
    print("TEST B-ACC : ", bacc)

    y_pred_commons = torch.cat(y_pred_commons, dim=0).cpu().numpy()
    bacc = roc_auc_score(y_s, y_pred_commons)
    print("TEST COMMON B-ACC : ", bacc)

    factor_pred = torch.cat(factor_pred, dim=0)
    factor_s = torch.cat(factor_s, dim=0)
    bacc = (factor_s == factor_pred).float().mean()
    print("TEST FACTOR B-ACC : ", bacc.item())

# test linear probes
def test_linear_probe(vae):
    # compute the representation of the normal set
    with torch.no_grad():
        X_train_mu = []
        X_test_mu = []
        y_train = []
        y_test = []
        y_digit_train = []
        y_digit_test = []
        for data, y, y_digit, _ in train_loader:
            data = data.cuda()
            mean, _, _ = vae.inference(data)
            X_train_mu.extend(mean.cpu().numpy())
            y_train.extend(y.cpu().numpy())
            y_digit_train.extend(y_digit.cpu().numpy())
        for data, y, y_digit, _ in test_loader:
            data = data.cuda()
            mean, _, _ = vae.inference(data)
            X_test_mu.extend(mean.cpu().numpy())
            y_test.extend(y.cpu().numpy())
            y_digit_test.extend(y_digit.cpu().numpy())
        X_train_mu = np.array(X_train_mu)
        X_test_mu = np.array(X_test_mu)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        y_digit_train = np.array(y_digit_train)
        y_digit_test = np.array(y_digit_test)

    scaler = StandardScaler()
    X_train_mu = scaler.fit_transform(X_train_mu)
    X_test_mu = scaler.transform(X_test_mu)

    # compute cifar labels classification on specific latents
    log_reg = LogisticRegression(max_iter=100).fit(X_train_mu[y_digit_train > -1, common_size:], y_train[y_digit_train > -1])
    log_reg_score = log_reg.score(X_test_mu[y_digit_test > -1, common_size:], y_test[y_digit_test > -1])
    print("Linear probe trained on cifar labels, specific latents : ", log_reg_score)

    # compute mnist labels classification on specific latents
    log_reg = LogisticRegression(max_iter=100).fit(X_train_mu[y_digit_train > -1, common_size:], y_digit_train[y_digit_train > -1,])
    log_reg_score = log_reg.score(X_test_mu[y_digit_test > -1, common_size:], y_digit_test[y_digit_test > -1])
    print("Linear probe trained on mnist labels, specific latents : ", log_reg_score)

    # compute cifar labels classification on common latents
    log_reg = LogisticRegression(max_iter=100).fit(X_train_mu[y_digit_train > -1, :common_size], y_train[y_digit_train > -1])
    log_reg_score = log_reg.score(X_test_mu[y_digit_test > -1, :common_size], y_test[y_digit_test > -1])
    print("Linear probe trained on cifar labels, common latents : ", log_reg_score)

    # compute mnist labels classification on common latents
    log_reg = LogisticRegression(max_iter=100).fit(X_train_mu[y_digit_train > -1, :common_size], y_digit_train[y_digit_train > -1,])
    log_reg_score = log_reg.score(X_test_mu[y_digit_test > -1, :common_size], y_digit_test[y_digit_test > -1])
    print("Linear probe trained on mnist labels, common latents : ", log_reg_score)

# Training
for run in range(5) :
    print(run)
    # define the models
    factor_classifier = FactorClassifier(common_size + salient_size).float().cuda()
    common_classifier = Classifier(common_size).float().cuda()
    vae = ShallowDisVAE(common_size+salient_size).float().cuda()

    for epoch in range(1, 251):
        # redefine optimizers at each epoch lead to better results
        # (provably because it re-init internal states at each epoch)
        optimizer = torch.optim.Adam(vae.parameters())
        factor_optimizer = torch.optim.Adam(list(factor_classifier.parameters()) + list(common_classifier.parameters()))
        train(epoch, vae, optimizer, factor_optimizer)

        # if epoch % 25 == 0:
    test(vae)
    test_linear_probe(vae)
    print("----------")
