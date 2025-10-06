import argparse
import logging
import sys
import os
import torch
torch.cuda.empty_cache()

from torch import optim

from disvae import init_specific_model, Trainer, Evaluator
from disvae.utils.modelIO import save_model, load_model, load_metadata
from disvae.models.losses import LOSSES, RECON_DIST, get_loss_f
from disvae.models.vae import MODELS
from utils.datasets import get_dataloaders, get_img_size, get_val_dataloader, get_test_dataloader
from utils.helpers import (create_safe_directory, get_device, set_seed, get_n_param,
                           FormatterNoDuplicate)
from utils.visualize import EpochImageSaver, Visualizer, EpochXAI

CONFIG_FILE = "hyperparam.ini"
RES_DIR = "results"


def parse_arguments(args_to_parse):
    parser = argparse.ArgumentParser(description="Train VAE on custom dataset",
                                     formatter_class=FormatterNoDuplicate)

    parser.add_argument('name', type=str, help="Name of the experiment folder")
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--latent-dim', type=int, default=10, help='Latent dimension')
    parser.add_argument('--model-type', default='Burgess', choices=MODELS, help='Model architecture')
    parser.add_argument('--loss', default='B', choices=LOSSES, help='Loss function')
    parser.add_argument('--rec-dist', default='bernoulli', choices=RECON_DIST, help='Reconstruction distribution')
    parser.add_argument('--reg-anneal', type=float, default=0.0, help='Annealing steps')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--is-eval-only', action='store_true')
    parser.add_argument('--is-metrics', action='store_true')
    parser.add_argument('--no-test', action='store_true')
    parser.add_argument('--eval-batchsize', type=int, default=64)
    parser.add_argument('--log-level', default='INFO')
    return parser.parse_args(args_to_parse)


def setup_logger(log_level):
    logging.basicConfig(
        level=log_level.upper(),
        format='%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger()


def main(args):
    logger = setup_logger(args.log_level)

    logger.info("Starting main function")
    set_seed(args.seed)

    device = get_device(is_gpu=not args.no_cuda)
    logger.info(f"Using device: {device}")

    exp_dir = os.path.join(RES_DIR, args.name)
    logger.info(f"Experiment directory: {exp_dir}")

    if not args.is_eval_only:
        create_safe_directory(exp_dir, logger=logger)
        logger.info("Created experiment directory")

        train_loader = get_dataloaders(batch_size=args.batch_size)
        logger.info(f"Loaded training data: {len(train_loader.dataset)} samples, {len(train_loader)} batches")

        args.img_size = get_img_size()
        logger.info(f"Image size: {args.img_size}")

        model = init_specific_model(args.model_type, args.img_size, args.latent_dim)
        logger.info(f"Initialized model '{args.model_type}' with {get_n_param(model)} parameters")

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        model = model.to(device)
        logger.info(f"Model moved to device")

        visualizer = Visualizer(model)
        #gif_visualizer = GifTraversalsTraining(model, "disvae", exp_dir)
        saver = EpochImageSaver(visualizer, save_dir=os.path.join(exp_dir, "epoch_images"), n_images=5)

        xai_saver = EpochXAI(
            model.encoder,
            model.decoder,
            save_dir=os.path.join(exp_dir, "xai"),
            device=device,
            n_images=5,
            run_every=1
        )

        # Initialize DiCE Counterfactuals (optional)
        from utils.visualize import DiceCounterfactuals
        dice_explainer = DiceCounterfactuals(
            model=model,
            device=device,
            data_loader=train_loader,
            output_dir=os.path.join(exp_dir, "xai", "dice"),
            num_cf=3
        )

        loss_f = get_loss_f(args.loss, n_data=len(train_loader.dataset), device=device, **vars(args))

        # Load all validation datasets
        val_loaders = get_val_dataloader(root_dir="data", batch_size=args.eval_batchsize)
        logger.info(f"Loaded validation datasets: {len(val_loaders.dataset)} samples")

        trainer = Trainer(model, optimizer, loss_f, device=device, logger=logger,
                          save_dir=exp_dir, saver= saver, xai_saver=xai_saver, is_progress_bar=True, val_loaders=val_loaders)
        
        # Attach lightweight Dice counterfactual generator
        trainer.dice_explainer = dice_explainer

        logger.info(f"Starting training for {args.epochs} epochs")
        trainer(train_loader, epochs=args.epochs, checkpoint_every=1)
        logger.info("Training completed")

        save_model(trainer.model, exp_dir, metadata=vars(args))
        logger.info("Model saved")

    if args.is_metrics or not args.no_test:
        logger.info("Starting evaluation phase")

        model = load_model(exp_dir, is_gpu=not args.no_cuda)
        logger.info("Model loaded for evaluation")

        metadata = load_metadata(exp_dir)

        test_loader = get_test_dataloader(root_dir="data", batch_size=args.eval_batchsize)
        logger.info(f"Loaded test data: {len(test_loader.dataset)} samples, {len(test_loader)} batches")

        loss_f = get_loss_f(args.loss, n_data=len(test_loader.dataset), device=device, **vars(args))
        evaluator = Evaluator(model, loss_f, device=device, logger=logger, save_dir=exp_dir, is_progress_bar=True)

        evaluator(test_loader, is_metrics=args.is_metrics, is_losses=not args.no_test)
        logger.info("Evaluation completed")


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
