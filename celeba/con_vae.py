# import functions
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from cnn_model import Encoder, Decoder, Classifier, FactorClassifier
from dataset import CelebaDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# ignore warning
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

# load data
X_train = np.load('./celeba/X_train_celeba.npy').transpose(0, 3, 1, 2) / 255.
X_test = np.load('./celeba/X_test_celeba.npy').transpose(0, 3, 1, 2) / 255.
y_train = np.load('./celeba/y_train_subtype.npy')
y_test = np.load('./celeba/y_test_subtype.npy')

# hyper-parameter
common_size=16
salient_size=16
alpha = 1/0.025
beta_c = 0.5
beta_s = 0.5
kappa = 2
gamma = 0

## create loss function
bce = nn.BCELoss(reduction="sum")
def compute_loss(x_reconstructed, x, z_mean, z_log_var, y, y_pred):
    # elbo loss
    reconstruction_loss = (x[y==1] - x_reconstructed[y==1]).pow(2).sum()
    reconstruction_loss += (x[y==0] - x_reconstructed[y==0]).pow(2).sum()

    kl_div_loss = - beta_c * 0.5 * torch.sum(1 + z_log_var[y==0, :common_size] - z_mean[y==0, :common_size].pow(2) - z_log_var[y==0, :common_size].exp(), dim=-1).sum()
    kl_div_loss += - beta_c * 0.5 * torch.sum(1 + z_log_var[y==1, :common_size] - z_mean[y==1, :common_size].pow(2) - z_log_var[y==1, :common_size].exp(), dim=-1).sum()
    kl_div_loss += - beta_s * 0.5 * torch.sum(1 + z_log_var[y==1, common_size:] - z_mean[y==1, common_size:].pow(2) - z_log_var[y==1, common_size:].exp(), dim=-1).sum()
    # kl_div_loss += beta_s * alpha * 0.5 * z_mean[y==0, common_size:].pow(2).sum()

    clsf_loss = bce(y_pred, y[:, None])

    return reconstruction_loss, kl_div_loss, clsf_loss


class ShallowDisVAE(nn.Module):
    """ The VAE architecture.
    """

    def __init__(self, background_latent_size, salient_latent_size, sigma_q=0.1):
        super(ShallowDisVAE, self).__init__()

        total_latent_size = background_latent_size + salient_latent_size
        self.common_size = background_latent_size
        self.salient_size = salient_latent_size
        self.total_latent_size = total_latent_size
        self.sigma_q=sigma_q

        self.encoder = Encoder(background_latent_size, salient_latent_size).float()
        self.decoder = Decoder(background_latent_size, salient_latent_size).float()
        self.classifier = Classifier(salient_latent_size).float()

    def forward(self, x, y):
        device = x.get_device()
        batch_size = x.size()[0]

        # compute the parameters of the posterior distribution
        mean_qzx, logvar_qzx, specific_mean_qzx, specific_logvar_qzx = self.encoder(x)
        # specific_logvar_qzx[y==0] = torch.log(0*torch.ones_like(specific_logvar_qzx[y==0]))

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


def train(epoch, vae, optimizer, factor_optimizer):
    vae.train()
    train_loss = 0
    for batch_idx, (data, y) in enumerate(train_loader):
        data = data.cuda()
        y = y.cuda()
        y=(y>0.5).float()

        # independent optimizer training
        factor_optimizer.zero_grad()
        _, _, _, _, z = vae(data, y)

        # common classifiacation training
        y_pred_common = common_classifier(z[:, :common_size])
        fader_clsf_loss = bce(y_pred_common, y[:, None].float())

        # specific classification training
        y_pred = specific_classifier(z[:, common_size:])
        clsf_loss = bce(y_pred, y[:, None].float())

        # total correlation discriminator training
        joint_predictions = factor_classifier(z)
        product_of_marginals_predictions = factor_classifier(torch.cat((z[:, :common_size], torch.cat((z[1:, common_size:], z[0, common_size:][None]), dim=0)), dim=1))
        factor_input = torch.cat((joint_predictions[:, 0], product_of_marginals_predictions[:, 0]), dim=0)
        factor_target = torch.cat((torch.ones_like(joint_predictions[:, 0]), torch.zeros_like(product_of_marginals_predictions[:, 0])), dim=0)
        factor_clsf_loss = bce(factor_input, factor_target)

        # parameters update
        loss = factor_clsf_loss + fader_clsf_loss + clsf_loss
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
        for data, y in test_loader:
            data = data.cuda()
            y = y.cuda()
            y=(y>0.5).float()

            z_mean, z_log_var, y_pred = vae.inference(data)

            y_pred = specific_classifier(z_mean[:, common_size:])

            y_pred_common = common_classifier(z_mean[:, :common_size])
            y_pred_commons.append(y_pred_common[:, 0])

            joint_predictions = factor_classifier(z_mean)
            product_of_marginals_predictions = factor_classifier(torch.cat((z_mean[:, :common_size], torch.cat((z_mean[1:, common_size:], z_mean[0, common_size:][None]), dim=0)), dim=1))
            factor_pred.append(joint_predictions[:, 0].round().int())
            factor_pred.append(product_of_marginals_predictions[:, 0].round().int())
            factor_s.append(torch.ones_like(joint_predictions[:, 0]))
            factor_s.append(torch.zeros_like(joint_predictions[:, 0]))

            y_preds.append(y_pred)
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


def test_linear_probe(vae, train_loader, test_loader):
    # compute the representation of the normal set
    with torch.no_grad():
        X_train_mu = []
        X_test_mu = []
        y_train_subtype = []
        y_test_subtype = []
        for data, target in train_loader:
            data = data.cuda()
            mean, _, _ = vae.inference(data)
            X_train_mu.extend(mean.cpu().numpy())
            y_train_subtype.extend(target.cpu().numpy())
        for data, target in test_loader:
            data = data.cuda()
            mean, _, _ = vae.inference(data)
            X_test_mu.extend(mean.cpu().numpy())
            y_test_subtype.extend(target.cpu().numpy())
        X_train_mu = np.array(X_train_mu)
        X_test_mu = np.array(X_test_mu)
        y_train_subtype = np.array(y_train_subtype)
        y_test_subtype = np.array(y_test_subtype)

    scaler = StandardScaler()
    X_train_mu = scaler.fit_transform(X_train_mu)
    X_test_mu = scaler.transform(X_test_mu)

    # compute hats vs glasses performances
    log_reg = LogisticRegression(max_iter=100).fit(X_train_mu[y_train_subtype>0.5, common_size:], y_train_subtype[y_train_subtype>0.5])
    log_reg_score = log_reg.score(X_test_mu[y_test_subtype>0.5, common_size:], y_test_subtype[y_test_subtype>0.5])
    print("Linear probe trained on subtype labels, specific latents : ", log_reg_score)
    log_reg = LogisticRegression(max_iter=100).fit(X_train_mu[y_train_subtype>0.5, :common_size], y_train_subtype[y_train_subtype>0.5])
    log_reg_score = log_reg.score(X_test_mu[y_test_subtype>0.5, :common_size], y_test_subtype[y_test_subtype>0.5])
    print("Linear probe trained on subtype labels, common latents : ", log_reg_score)


# Training
for run in range(5) :
    print(run)
    # Instantiate Dataset and Data Loader
    train_dataset = CelebaDataset(X_train, y_train)
    test_dataset = CelebaDataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=512, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=512)

    # define the models
    factor_classifier = FactorClassifier(common_size + salient_size).float().cuda()
    common_classifier = Classifier(common_size).float().cuda()
    specific_classifier = Classifier(salient_size).float().cuda()
    vae = ShallowDisVAE(common_size, salient_size).float().cuda()

    for epoch in range(1,201):
        # redefine optimizers at each epoch lead to better results
        # (provably because it re-init internal states at each epoch)
        optimizer = torch.optim.Adam(vae.parameters())
        factor_optimizer = torch.optim.Adam(list(factor_classifier.parameters()) + list(common_classifier.parameters()) + list(specific_classifier.parameters()))
        train(epoch, vae, optimizer, factor_optimizer)

    test(vae)
    test_linear_probe(vae, train_loader, test_loader)
    print("----------")
