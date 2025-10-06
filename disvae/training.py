import imageio
import logging
import os
from timeit import default_timer
from collections import defaultdict

from tqdm import trange
import torch
from torch.nn import functional as F

from disvae.utils.modelIO import save_model


TRAIN_LOSSES_LOGFILE = "train_losses.log"

import os
import torch

def compute_iou_error_epoch(model, data_loader, device, epoch, category_name="", save_file="results/iue.txt"):
    """
    Compute average IoU error for one validation set and append to a single file.

    Args:
        model: VAE model
        data_loader: DataLoader for validation set
        device: torch.device
        epoch: current epoch
        category_name: str, optional, category label
        save_file: path to the single txt file

    Returns:
        avg_iou_error (float)
    """
    model.eval()
    iou_errors = []

    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            recon, _, _ = model(x)

            # Binarize images for IoU
            x_bin = (x > 0.5).float()
            recon_bin = (recon > 0.5).float()

            intersection = (x_bin * recon_bin).sum(dim=(1,2,3))
            union = (x_bin + recon_bin - x_bin * recon_bin).sum(dim=(1,2,3)) + 1e-8
            iou = intersection / union
            iou_errors.append(1 - iou)  # IoU error

    avg_iou_error = torch.cat(iou_errors).mean().item()

    # Append to single txt file
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    with open(save_file, "a") as f:
        f.write(f"Epoch {epoch}, Category {category_name}, IoU Error: {avg_iou_error:.6f}\n")

    return avg_iou_error

class Trainer():
    """
    Class to handle training of model.

    Parameters
    ----------
    model: disvae.vae.VAE

    optimizer: torch.optim.Optimizer

    loss_f: disvae.models.BaseLoss
        Loss function.

    device: torch.device, optional
        Device on which to run the code.

    logger: logging.Logger, optional
        Logger.

    save_dir : str, optional
        Directory for saving logs.

    gif_visualizer : viz.Visualizer, optional
        Gif Visualizer that should return samples at every epochs.

    is_progress_bar: bool, optional
        Whether to use a progress bar for training.
    """

    def __init__(self, model, optimizer, loss_f,
                 device=torch.device("cpu"),
                 logger=logging.getLogger(__name__),
                 save_dir="results",
                 #gif_visualizer=None,
                 saver = None,
                 xai_saver=None,
                 is_progress_bar=True,
                 val_loaders=None,
                 test_loader=None):
        self.xai_saver = xai_saver
        # Store CF images per epoch for visualization
        if hasattr(self, "dice_explainer") and self.dice_explainer is not None:
            self.dice_explainer.epoch_cf_images = []  # list of lists

        self.val_loaders = val_loaders
        self.test_loader = test_loader

        self.device = device
        self.model = model.to(self.device)
        self.loss_f = loss_f
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.is_progress_bar = is_progress_bar
        self.logger = logger
        self.losses_logger = LossesLogger(os.path.join(self.save_dir, TRAIN_LOSSES_LOGFILE))
        #self.gif_visualizer = gif_visualizer
        self.saver = saver
        self.xai_saver = xai_saver
        self.logger.info("Training Device: {}".format(self.device))

    def __call__(self, data_loader,
                 epochs=10,
                 checkpoint_every=10, val_loaders=None):
        from utils.visualize import EpochXAI
        """
        Trains the model.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        epochs: int, optional
            Number of epochs to train the model for.

        checkpoint_every: int, optional
            Save a checkpoint of the trained model every n epoch.
        """
        start = default_timer()
        self.model.train()
        for epoch in range(1, epochs+1):
            storer = defaultdict(list)
            mean_epoch_loss = self._train_epoch(data_loader, storer, epoch)
            self.logger.info('Epoch: {} Average loss per image: {:.2f}'.format(epoch,
                                                                               mean_epoch_loss))
            self.losses_logger.log(epoch, storer)

            # create 'model' folder if it doesn't exist
            model_dir = os.path.join(self.save_dir, "model")
            os.makedirs(model_dir, exist_ok=True)
            save_model(self.model, model_dir, filename=f"model-{epoch}.pt", metadata=None)

            # save every epoch
            if epoch % checkpoint_every == 0 or epoch == epochs:
                save_model(self.model, model_dir, filename=f"model-{epoch}.pt")
                self.logger.info(f"Saved checkpoint in: {model_dir} as model-{epoch}.pt")


            if self.saver is not None:
                first_batch, _ = next(iter(data_loader))
                self.saver.save_epoch_images(first_batch, epoch)

                # Call XAI visualizations
            if self.xai_saver is not None:
                force_save = (epoch == epochs - 1)
                self.xai_saver.save_epoch_xai(epoch, data_loader, force=force_save)

            # --- Run DiCE CFs per epoch ---
            if hasattr(self, "dice_explainer") and self.dice_explainer is not None:
                self.logger.info(f"Generating DiCE counterfactuals for epoch {epoch}...")
                try:
                    self.dice_explainer.run_epoch_cf(epoch=epoch)
                    self.logger.info(f"Saved CFs for epoch {epoch}.")
                except Exception as e:
                    self.logger.warning(f"DiCE generation failed at epoch {epoch}: {e}")

                # --- Compute IoU on all validation sets ---
            if self.val_loaders is not None:
                save_file = os.path.join(self.save_dir, "iue.txt")
                avg_iou_err = compute_iou_error_epoch(
                    self.model, self.val_loaders, self.device, epoch,
                    category_name="val",
                    save_file=save_file
                )
                self.logger.info(f"Epoch {epoch} Validation IoU Error: {avg_iou_err:.6f}")

        self.model.eval()

        delta_time = (default_timer() - start) / 60
        self.logger.info('Finished training after {:.1f} min.'.format(delta_time))

    def _train_epoch(self, data_loader, storer, epoch):
        """
        Trains the model for one epoch.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        storer: dict
            Dictionary in which to store important variables for vizualisation.

        epoch: int
            Epoch number

        Return
        ------
        mean_epoch_loss: float
            Mean loss per image
        """
        epoch_loss = 0.
        kwargs = dict(desc="Epoch {}".format(epoch + 1), leave=False,
                      disable=not self.is_progress_bar)
        with trange(len(data_loader), **kwargs) as t:
            for _, (data, _) in enumerate(data_loader):
                iter_loss = self._train_iteration(data, storer)
                epoch_loss += iter_loss
 
                t.set_postfix(loss=iter_loss)
                t.update()

        mean_epoch_loss = epoch_loss / len(data_loader)

        return mean_epoch_loss

    def _train_iteration(self, data, storer, noise_std = 0.5):
        """
        Trains the model for one iteration on a batch of data.

        Parameters
        ----------
        data: torch.Tensor
            A batch of data. Shape : (batch_size, channel, height, width).

        storer: dict
            Dictionary in which to store important variables for vizualisation.
        """
        batch_size, channel, height, width = data.size()
        data = data.to(self.device)

        try:
            recon_batch, latent_dist, latent_sample = self.model(data, noise_std = noise_std)
            loss = self.loss_f(data, recon_batch, latent_dist, self.model.training,
                               storer, latent_sample=latent_sample)

            mse_loss = F.mse_loss(recon_batch, data, reduction='mean')  # per image MSE
            if storer is not None:
                storer['mse_loss'].append(mse_loss.item())  # log it          

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        except ValueError:
            # for losses that use multiple optimizers (e.g. Factor)
            loss = self.loss_f.call_optimize(data, self.model, self.optimizer, storer)

        return loss.item()


class LossesLogger(object):
    """Class definition for objects to write data to log files in a
    form which is then easy to be plotted.
    """

    def __init__(self, file_path_name):
        """ Create a logger to store information for plotting. """
        if os.path.isfile(file_path_name):
            os.remove(file_path_name)

        self.logger = logging.getLogger("losses_logger")
        self.logger.setLevel(1)  # always store
        file_handler = logging.FileHandler(file_path_name)
        file_handler.setLevel(1)
        self.logger.addHandler(file_handler)

        header = ",".join(["Epoch", "Loss", "Value"])
        self.logger.debug(header)

    def log(self, epoch, losses_storer):
        """Write to the log file """
        for k, v in losses_storer.items():
            log_string = ",".join(str(item) for item in [epoch, k, mean(v)])
            self.logger.debug(log_string)


# HELPERS
def mean(l):
    """Compute the mean of a list"""
    return sum(l) / len(l)
