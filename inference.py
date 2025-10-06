import os
import torch
from torchvision import transforms
from PIL import Image
from disvae.utils.modelIO import load_model, load_metadata
from utils.datasets import get_test_dataloader
from utils.helpers import get_device


def reconstruct_and_save(model, data_loader, device, save_dir="results/recon-test", max_batches=None):
    """
    Reconstruct images from test data and save input and reconstruction separately in the same folder.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    to_pil = transforms.ToPILImage()
    img_counter = 1

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(data_loader):
            if max_batches and batch_idx >= max_batches:
                break

            x = x.to(device)
            recon, _, _ = model(x)

            for i in range(x.size(0)):
                input_img = to_pil(x[i].cpu())
                recon_img = to_pil(recon[i].cpu())

                input_path = os.path.join(save_dir, f"input{img_counter}.png")
                recon_path = os.path.join(save_dir, f"recon{img_counter}.png")

                input_img.save(input_path)
                recon_img.save(recon_path)

                img_counter += 1

            print(f"✅ Saved batch {batch_idx + 1}/{len(data_loader)}")

    print(f"\n🎯 Reconstructions saved to: {save_dir}")


def main():
    # --- Configuration ---
    model_path = r"C:/depro/results/dicef/model/model.pt"  # your checkpoint
    root_dir = r"C:/depro/data"  # dataset root (must contain test folder)
    batch_size = 8
    save_dir = "results/testrecon"

    # --- Device ---
    device = get_device(is_gpu=True)
    print(f"Using device: {device}")

    # --- Load metadata and model ---
    model_dir = os.path.dirname(model_path)        
    metadata_dir = os.path.dirname(model_dir)      

    metadata = load_metadata(metadata_dir)
    print(f"✅ Loaded metadata from: {metadata_dir}")

    model = load_model(model_dir, is_gpu=True)
    model = model.to(device)
    print("✅ Model loaded successfully.")

    # --- Load test data ---
    test_loader = get_test_dataloader(root_dir=root_dir, batch_size=batch_size)
    print(f"Loaded test data: {len(test_loader.dataset)} images.")

    # --- Reconstruct and save results ---
    reconstruct_and_save(model, test_loader, device, save_dir=save_dir, max_batches=None)


if __name__ == "__main__":
    main()
