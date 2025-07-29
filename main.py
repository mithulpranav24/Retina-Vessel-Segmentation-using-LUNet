import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import v2 as transforms
from tqdm import tqdm

#
# ----------------- 1. CONFIGURATION -----------------
#
# --- SET THESE VALUES BEFORE RUNNING ---
# Make sure this path points to the root of your DRIVE dataset folder
DATASET_PATH = "./DRIVE"
MODEL_NAME = "my_drive_unet"
EPOCHS = 100
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
IMG_SIZE = 512
# ---

#
# ----------------- 2. REPRODUCIBILITY -----------------
#
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


#
# ----------------- 3. DATASET CLASS -----------------
#
class DriveDataset(Dataset):
    """
    Custom Dataset for the DRIVE database.
    The dataset is expected to be in the structure provided in the prompt.
    """

    def __init__(self, root_dir, split='train', img_size=512):
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split

        if split == 'train' or split == 'val':
            self.image_dir = os.path.join(root_dir, 'training', 'images')
            self.mask_dir = os.path.join(root_dir, 'training', '1st_manual')
            self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.tif')])
        elif split == 'test':
            self.image_dir = os.path.join(root_dir, 'test', 'images')
            self.mask_dir = os.path.join(root_dir, 'test', 'mask')
            self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.tif')])
        else:
            raise ValueError(f"Invalid split '{split}'. Choose from 'train', 'val', 'test'.")

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Construct mask name based on split
        if self.split in ['train', 'val']:
            mask_name = img_name.split('_')[0] + "_manual1.gif"
        else:  # test
            mask_name = img_name.split('_')[0] + "_test_mask.gif"
        mask_path = os.path.join(self.mask_dir, mask_name)

        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")  # Grayscale
        except FileNotFoundError:
            print(f"Error: Could not find image or mask for index {idx}")
            print(f"Image path: {img_path}")
            print(f"Mask path: {mask_path}")
            return None, None

        image = self.transform(image)
        mask = self.mask_transform(mask)

        # Binarize mask to ensure it's {0, 1}
        mask = (mask > 0.5).float()

        return image, mask


#
# ----------------- 4. U-NET MODEL ARCHITECTURE -----------------
#
class DoubleConv(nn.Module):
    """(Convolution -> BatchNorm -> ReLU -> Dropout) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, dropout_p=0.1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        padding = kernel_size // 2
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=7, dropout_p=0.1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel_size=kernel_size, dropout_p=dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=7, dropout_p=0.1, transpose_stride=2):
        super().__init__()
        # THE FIX IS APPLIED ON THE LINE BELOW
        self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=transpose_stride)
        self.conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size, dropout_p=dropout_p)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 to match x2 size if necessary
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    PyTorch U-Net model adapted from the Keras script parameters.
    - init_n_filters = 18
    - kernel_size = 7
    - keep_prob = 0.9 (dropout_p = 0.1)
    """

    def __init__(self, n_channels=3, n_classes=1, init_n_filters=18, kernel_size=7, dropout_p=0.1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        f = init_n_filters
        self.inc = DoubleConv(n_channels, f, kernel_size=kernel_size, dropout_p=dropout_p)
        self.down1 = Down(f, f * 2, kernel_size=kernel_size, dropout_p=dropout_p)
        self.down2 = Down(f * 2, f * 4, kernel_size=kernel_size, dropout_p=dropout_p)
        self.down3 = Down(f * 4, f * 8, kernel_size=kernel_size, dropout_p=dropout_p)
        factor = 2
        self.down4 = Down(f * 8, f * 16 // factor, kernel_size=kernel_size, dropout_p=dropout_p)
        self.up1 = Up(f * 16, f * 8 // factor, kernel_size=kernel_size, dropout_p=dropout_p)
        self.up2 = Up(f * 8, f * 4 // factor, kernel_size=kernel_size, dropout_p=dropout_p)
        self.up3 = Up(f * 4, f * 2 // factor, kernel_size=kernel_size, dropout_p=dropout_p)
        self.up4 = Up(f * 2, f, kernel_size=kernel_size, dropout_p=dropout_p)
        self.outc = OutConv(f, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


#
# ----------------- 5. LOSS & METRICS -----------------
#
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1 - dice


def dice_coefficient(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


# Combined Loss
def criterion(pred, target):
    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()
    return bce(pred, target) + dice(pred, target)


# --- SCRIPT ENTRY POINT GUARD ---
if __name__ == '__main__':
    print('*' * 50)
    print(f'Starting PyTorch experiment with hardcoded settings.')
    print(f'Dataset Path: {DATASET_PATH}')
    print(f'Model will be saved as: {MODEL_NAME}.pth')
    print('*' * 50)

    #
    # ----------------- 6. TRAINING SETUP -----------------
    #
    # DataLoaders
    full_dataset = DriveDataset(DATASET_PATH, split='train', img_size=IMG_SIZE)
    # Splitting training data into train and validation (e.g., 18 for train, 2 for val)
    train_size = len(full_dataset) - 2
    val_size = 2
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Device, Model, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        n_channels=3,
        n_classes=1,
        init_n_filters=18,
        kernel_size=7,
        dropout_p=0.1
    )
    model.to(device)

    # Handle multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    #
    # ----------------- 7. TRAINING & VALIDATION LOOP -----------------
    #
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_dice': []}

    print(f"\nTraining on {device}...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        # Training loop
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")
        for images, masks in pbar_train:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            pbar_train.set_postfix({'loss': loss.item()})

        epoch_train_loss = running_loss / len(train_dataset)
        history['train_loss'].append(epoch_train_loss)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]")
        with torch.no_grad():
            for images, masks in pbar_val:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item() * images.size(0)
                val_dice += dice_coefficient(outputs, masks).item() * images.size(0)
                pbar_val.set_postfix({'val_loss': loss.item(), 'val_dice': dice_coefficient(outputs, masks).item()})

        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_dice = val_dice / len(val_dataset)
        history['val_loss'].append(epoch_val_loss)
        history['val_dice'].append(epoch_val_dice)

        print(f"Epoch {epoch + 1}/{EPOCHS} -> "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, "
              f"Val Dice: {epoch_val_dice:.4f}")

        # Save the best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_path = MODEL_NAME + '_best.pth'
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path}")

    #
    # ----------------- 8. SAVE FINAL RESULTS -----------------
    #
    # Save the final model
    final_model_path = MODEL_NAME + '_final.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")

    # Save training history
    history_path = MODEL_NAME + '_history.npy'
    np.save(history_path, history)
    print(f"Saved training history to {history_path}")

    print("\nTraining finished.")