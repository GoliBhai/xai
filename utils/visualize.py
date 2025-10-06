import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from PIL import Image


# -------------------------------
# VAE Explainability (Traversals + Heatmaps)
# -------------------------------
class VAE_XAI_Burgess:
    def __init__(self, encoder, decoder, device, output_dir="xai_results"):
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.output_dir = output_dir
        os.makedirs(os.path.join(output_dir, "traversals"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "heatmaps"), exist_ok=True)

    def latent_traversals(self, x_batch, epoch, n_steps=5, latent_range=(-3, 3)):
        """Generate latent traversals for inspection."""
        self.encoder.eval()
        self.decoder.eval()
        x_batch = x_batch.to(self.device)

        with torch.no_grad():
            mu, logvar, z = self.encoder(x_batch, inject_noise=True)

        if z.ndim == 4:
            B, latent_dim, H, W = z.shape
        else:
            B, latent_dim = z.shape
            H = W = None

        for i in range(B):
            fig, axs = plt.subplots(latent_dim, n_steps,
                                    figsize=(n_steps * 1.5, latent_dim * 1.5))
            for dim in range(latent_dim):
                for j, val in enumerate(np.linspace(latent_range[0], latent_range[1], n_steps)):
                    z_temp = z[i].clone()
                    if H is not None:
                        z_temp[dim, :, :] += val
                    else:
                        z_temp[dim] += val
                    z_input = z_temp.unsqueeze(0).float().to(self.device)
                    x_recon = self.decoder(z_input).detach().cpu().squeeze().numpy()

                    if x_recon.ndim == 3:
                        x_recon = np.transpose(x_recon, (1, 2, 0))
                        axs[dim, j].imshow(x_recon)
                    else:
                        axs[dim, j].imshow(x_recon, cmap="gray")

                    axs[dim, j].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "traversals", f"epoch{epoch}_img{i}.png"))
            plt.close()

    def reconstruction_heatmaps(self, x_batch, epoch):
        """Generate heatmaps of reconstruction error."""
        self.encoder.eval()
        self.decoder.eval()
        x_batch = x_batch.to(self.device)

        with torch.no_grad():
            mu, logvar, z = self.encoder(x_batch, inject_noise=True)
            x_recon = self.decoder(z)

            # ensure same scale as input
            x_recon = torch.clamp(x_recon, 0, 1)

            # per-pixel squared error
            error_map = ((x_batch - x_recon) ** 2).squeeze(1).detach().cpu().numpy()

        for i, emap in enumerate(error_map):
            plt.imshow(emap, cmap="hot", vmin=0, vmax=1)
            plt.axis("off")
            plt.colorbar()
            plt.savefig(os.path.join(self.output_dir, "heatmaps", f"epoch{epoch}_img{i}.png"))
            plt.close()

    def run_all(self, x_batch, epoch):
        self.latent_traversals(x_batch, epoch)
        self.reconstruction_heatmaps(x_batch, epoch)


# -------------------------------
# Epoch-level Explainability Wrapper
# -------------------------------
class EpochXAI:
    def __init__(self, encoder, decoder, save_dir, device, n_images=5, run_every=1):
        self.xai = VAE_XAI_Burgess(encoder, decoder, device, output_dir=save_dir)
        self.n_images = n_images
        self.run_every = run_every

    def save_epoch_xai(self, epoch, data_loader, force=False):
        if not force and epoch % self.run_every != 0:
            return
        x_batch, _ = next(iter(data_loader))
        x_batch = x_batch[:self.n_images]
        self.xai.run_all(x_batch, epoch)


# -------------------------------
# Simple Visualizer
# -------------------------------
class Visualizer:
    def __init__(self, model):
        self.model = model

    def reconstruct(self, x, size=(5, 1), is_original=False, is_force_return=False):
        self.model.eval()
        x = x.to(next(self.model.parameters()).device)
        recon, _, _ = self.model(x)
        return recon.cpu()


# -------------------------------
# Epoch Image Saver (Input + Recon)
# -------------------------------
class EpochImageSaver:
    """
    Saves input and reconstructed images for each epoch.
    """

    def __init__(self, visualizer, save_dir, n_images=5):
        self.visualizer = visualizer
        self.save_dir = save_dir
        self.n_images = n_images
        os.makedirs(self.save_dir, exist_ok=True)

    def save_epoch_images(self, data, epoch):
        epoch_dir = os.path.join(self.save_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)

        n_images_to_save = min(self.n_images, data.size(0))
        input_images = data[:n_images_to_save]
        reconstructed_images = self.visualizer.reconstruct(
            input_images, size=(self.n_images, 1), is_original=False, is_force_return=True
        )

        for i in range(n_images_to_save):
            save_image(input_images[i], os.path.join(epoch_dir, f"input_{i+1}.png"))
            save_image(reconstructed_images[i], os.path.join(epoch_dir, f"recon_{i+1}.png"))

# -------------------------------
# DiCE Counterfactual Explanation
# -------------------------------
class DiceCounterfactuals:
    """
    Generate and save counterfactuals per epoch for a fixed input.
    """

    def __init__(self, model, device, data_loader, output_dir="xai/dice", num_cf=3):
        self.model = model
        self.device = device
        self.data_loader = data_loader
        self.num_cf = num_cf
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.model.eval()
        self.input_image = None  # fixed input for all epochs

    def prepare_input(self, n_samples=1):
        """Pick the first batch and store the input image."""
        x_batch, _ = next(iter(self.data_loader))
        self.input_image = x_batch[:n_samples].to(self.device)
        save_image(self.input_image[0].cpu(), os.path.join(self.output_dir, "input.png"))

    def run_epoch_cf(self, epoch, n_samples=1):
        """Run counterfactual generation for a small batch after each epoch."""
        if self.input_image is None:
            x_batch, _ = next(iter(self.data_loader))
            self.input_image = x_batch[:n_samples].to(self.device)
            save_image(self.input_image[0].cpu(),
                       os.path.join(self.output_dir, "input.png"))
            self.input_saved = True

        with torch.no_grad():
            enc_out = self.model.encoder(self.input_image, inject_noise=False)
            if len(enc_out) == 3:
                _, _, z = enc_out
            elif len(enc_out) == 2:
                mu, logvar = enc_out
                z = mu
            else:
                raise ValueError(f"Unexpected encoder output: {len(enc_out)} values")

        # Generate 3 CFs
            z_cf = z.unsqueeze(1).repeat(1, self.num_cf, *([1]* (z.ndim-1)))
            z_cf += torch.randn_like(z_cf) * 0.5

        # Decode and save immediately
            x_cf_flat = self.model.decoder(z_cf.view(-1, *z_cf.shape[2:])).cpu()
            epoch_dir = os.path.join(self.output_dir, f"epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)

            for i in range(self.num_cf):
                save_image(x_cf_flat[i],
                           os.path.join(epoch_dir, f"cf_{i+1}.png"))




