import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Default dataset root (change if needed for GPU/local runs)
root_dir = "C:/depro/data"


class DisVaeDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        """
        Args:
            root_dir (str): Path to dataset root.
            split (str): "train" or "val". Used to ensure correct folder is loaded.
        """
        self.root_dir = root_dir
        self.split = split
        self.image_files = []
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

        # Recursively search for .bmp images of size 512x512
        for root, _, files in os.walk(root_dir):
            if split not in root:  # only pick images from the correct split
                continue
            for f in files:
                if f.lower().endswith('.bmp'):
                    path = os.path.join(root, f)
                    try:
                        with Image.open(path) as img:
                            if img.size == (512, 512):
                                self.image_files.append(path)
                    except Exception as e:
                        print(f"Warning: Skipping {path} due to error: {e}")

        if len(self.image_files) == 0:
            print(f"⚠️ Warning: No valid 512x512 .bmp images found in {root_dir}/{split}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('L')
        image = self.transform(image)
        return image, 0  # dummy label for compatibility


def get_dataloaders(batch_size=64, shuffle=True, logger=None, root_dir=root_dir):
    """Return DataLoader for training set only."""
    dataset = DisVaeDataset(root_dir=root_dir, split="train")
    if logger:
        logger.info(f"Loaded training dataset with {len(dataset)} samples.")
    else:
        print(f"Loaded training dataset with {len(dataset)} samples.")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


def get_val_dataloader(batch_size=64, shuffle=False, logger=None, root_dir=root_dir):
    """Return DataLoader for validation set only."""
    dataset = DisVaeDataset(root_dir=root_dir, split="val")
    if logger:
        logger.info(f"Loaded validation dataset with {len(dataset)} samples.")
    else:
        print(f"Loaded validation dataset with {len(dataset)} samples.")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

def get_test_dataloader(batch_size=64, shuffle=False, logger=None, root_dir=root_dir):
    dataset = DisVaeDataset(root_dir=root_dir, split="test")
    if logger:
        logger.info(f"Loaded test dataset with {len(dataset)} samples.")
    else:
        print(f"Loaded test dataset with {len(dataset)} samples.")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


def get_img_size(root_dir=root_dir):
    """Detect image size from first sample in train set."""
    dataset = DisVaeDataset(root_dir=root_dir, split="train")
    if len(dataset) == 0:
        raise RuntimeError("❌ Could not detect image size (no training images found).")
    sample, _ = dataset[0]
    return sample.shape  # (C, H, W)
