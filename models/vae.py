import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, dim, device):
        super().__init__()
        self.device = device

        # encoder
        self.enc_conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=0)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0)
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=0)

        self.act = nn.ReLU(inplace=True)

        # latent z
        self.mu = nn.Linear(1024, dim)
        self.logvar = nn.Linear(1024, dim)

        # decoder
        self.dec_conv1 = nn.ConvTranspose2d(dim, 128, kernel_size=5, stride=2, padding=0)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=0)
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2, padding=0)
        self.dec_conv4 = nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2, padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.latent(mu, logvar)
        o = self.decoder(z)

        return o, mu, logvar

    def encoder(self, x):
        b, c, h, w = x.shape
        y = self.act(self.enc_conv1(x))
        y = self.act(self.enc_conv2(y))
        y = self.act(self.enc_conv3(y))
        y = self.act(self.enc_conv4(y))

        y = y.view(b, 1024)

        mu = self.mu(y)
        logvar = self.logvar(y)

        return mu, logvar

    def decoder(self, z):
        b, c = z.shape
        z = z.view(b, c, 1, 1)
        o = self.act(self.dec_conv1(z))
        o = self.act(self.dec_conv2(o))
        o = self.act(self.dec_conv3(o))
        o = self.sigmoid(self.dec_conv4(o))

        return o

    def latent(self, mu, logvar):
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(logvar).to(self.device)
        z = mu + eps * sigma

        return z


