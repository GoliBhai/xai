import numpy as np
import torch
from torch import nn

def get_decoder(model_type):
    model_type = model_type.lower().capitalize()
    return eval("Decoder{}".format(model_type))


class DecoderBurgess(nn.Module):
    def __init__(self, img_size, latent_dim=10):
        super(DecoderBurgess, self).__init__()

        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.img_size = img_size
        n_chan = self.img_size[0]

        # Start from (hid_channels, 2, 2), matching encoder output
        #self.reshape = (hid_channels, 2, 2)

        # Fully connected layers to expand latent vector to feature map
        #self.lin1 = nn.Linear(latent_dim, hidden_dim)
        #self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        #self.lin3 = nn.Linear(hidden_dim, np.prod(self.reshape))

        cnn_kwargs = dict(stride=2, padding=1)

        self.initial = nn.Conv2d(latent_dim, hid_channels, kernel_size=3, padding=1)

        # 8 ConvTranspose2d layers to upsample from 2x2 to 512x512
        self.convT1 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)  # 2 -> 4
        self.convT2 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)  # 4 -> 8
        self.convT3 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)  # 8 -> 16
        self.convT4 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)  # 16 -> 32
        self.convT5 = nn.ConvTranspose2d(hid_channels, n_chan, kernel_size, **cnn_kwargs)  # 32 -> 64
        #self.convT6 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)  # 64 -> 128
        #self.convT7 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)  # 128 -> 256
        #self.convT8 = nn.ConvTranspose2d(hid_channels, n_chan, kernel_size, **cnn_kwargs)        # 256 -> 512

    def forward(self, z):
        #batch_size = z.size(0)

        # Fully connected layers with ReLU
        #x = torch.relu(self.lin1(z))
        #x = torch.relu(self.lin2(x))
        #x = torch.relu(self.lin3(x))

        x = torch.relu(self.initial(z))

        # Reshape to feature map
        #x = x.view(batch_size, *self.reshape)

        # Upsample step-by-step with ReLU except last layer
        x = torch.relu(self.convT1(x))
        x = torch.relu(self.convT2(x))
        x = torch.relu(self.convT3(x))
        x = torch.relu(self.convT4(x))
        #x = torch.relu(self.convT5(x))
        #x = torch.relu(self.convT6(x))
        #x = torch.relu(self.convT7(x))
        x = torch.sigmoid(self.convT5(x))  # Final layer with sigmoid activation

        return x
