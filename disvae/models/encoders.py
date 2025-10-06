import numpy as np
import torch
from torch import nn

def get_encoder(model_type):
    model_type = model_type.lower().capitalize()
    return eval("Encoder{}".format(model_type))


def add_gaussian_noise(x, mean=0.0, std=0.1):
    """
    Add Gaussian noise to a tensor.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor, e.g., feature maps [B, C, H, W]
    mean : float
        Mean of Gaussian noise
    std : float
        Standard deviation of Gaussian noise
    """
    noise = torch.randn_like(x) * std + mean
    return x + noise

class EncoderBurgess(nn.Module):
    def __init__(self, img_size,
                 latent_dim=10):
        r"""
        Encoder for 512x512 images (or other sizes >= 512).
        """

        super(EncoderBurgess, self).__init__()

        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.latent_dim = latent_dim
        self.img_size = img_size  # (C, H, W)
        n_chan = self.img_size[0]

        cnn_kwargs = dict(stride=2, padding=1)

        # For 512x512 input, add 8 conv layers with stride=2, each halving spatial size:
        # 512 -> 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2

        self.conv1 = nn.Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs)  # 512 -> 256
        self.conv2 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)  # 256 -> 128
        self.conv3 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)  # 128 -> 64
        self.conv4 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)  # 64 -> 32
        self.conv5 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)  # 32 -> 16
        #self.conv6 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)  # 16 -> 8
        #self.conv7 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)  # 8 -> 4
        #self.conv8 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)  # 4 -> 2

        # Final feature map size: (hid_channels, 2, 2)
        self.final_feature_shape = (hid_channels, 16, 16)
        flattened_size = hid_channels * 16 * 16

        # Fully connected layers
        #self.lin1 = nn.Linear(flattened_size, hidden_dim)
        #self.lin2 = nn.Linear(hidden_dim, hidden_dim)

        # Latent mean and logvar
        #self.mu_logvar_gen = nn.Linear(hidden_dim, self.latent_dim * 2)

        # Final 1x1 conv layers to generate μ and logvar maps
        self.mu_conv = nn.Conv2d(hid_channels, latent_dim, kernel_size=1)
        self.logvar_conv = nn.Conv2d(hid_channels, latent_dim, kernel_size=1)

    def forward(self, x, noise_std = 0.0, inject_noise = False):
        """
        Forward pass: encode image to latent distribution (mu, logvar),
        then optionally sample using reparameterization trick.
        
        Parameters
        ----------
        x : torch.Tensor
            Input image batch [B, C, H, W]
        inject_noise : bool
            If True, sample z from N(mu, sigma^2)
        """
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))

        # Inject noise if noise_std > 0
        if noise_std > 0.0:
            x = add_gaussian_noise(x, std=noise_std)

        mu = self.mu_conv(x)
        logvar = self.logvar_conv(x)

        if inject_noise:
            z = self.reparameterize(mu, logvar)
            return mu, logvar, z
        else:
            return mu, logvar  # return deterministic latent if no noise

    @staticmethod
    def reparameterize(mu, logvar):
        """
        Reparameterization trick: sample from N(mu, sigma^2)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # sample from standard normal
        return mu + eps * std


"""
    def forward(self, x):
        #batch_size = x.size(0)

        # Conv layers with ReLU activations
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        #x = torch.relu(self.conv6(x))
        #x = torch.relu(self.conv7(x))
        #x = torch.relu(self.conv8(x))

        # Flatten
        #x = x.view(batch_size, -1)

        # Fully connected layers
        #x = torch.relu(self.lin1(x))
        #x = torch.relu(self.lin2(x))

        # Latent mean and log variance
        #mu_logvar = self.mu_logvar_gen(x)
        #mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)
        # μ and logvar feature maps, shape [B, latent_dim, 16, 16]
        mu = self.mu_conv(x)
        logvar = self.logvar_conv(x)

        return mu, logvar
"""