import torch
import os
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import the model and dataset classes from your training script
# Make sure this script is in the same directory as your training script
from main import UNet, DriveDataset

# --- CONFIGURATION ---
MODEL_PATH = "test_predictions/my_drive_unet_best.pth"  # Use the best model saved during training
DATASET_PATH = "./DRIVE"
OUTPUT_DIR = "./test_predictions"
BATCH_SIZE = 4
IMG_SIZE = 512


# ---

def predict():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Setup device, model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=3, n_classes=1, init_n_filters=18, kernel_size=7, dropout_p=0.1)

    # Load the trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Setup the test dataset and dataloader
    test_dataset = DriveDataset(DATASET_PATH, split='test', img_size=IMG_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Running inference on {len(test_dataset)} test images...")

    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(test_loader, desc="Predicting")):
            images = images.to(device)

            # Get model predictions
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()

            # Save each prediction in the batch
            for j in range(preds.shape[0]):
                # Get the original filename to name the output
                original_file_name = test_dataset.image_files[i * BATCH_SIZE + j]
                output_file_name = original_file_name.replace(".tif", "_pred.png")

                # Convert tensor to numpy array and scale to 0-255
                pred_mask = preds[j].cpu().numpy().squeeze()
                pred_mask = (pred_mask * 255).astype(np.uint8)

                # Save as an image
                img = Image.fromarray(pred_mask)
                img.save(os.path.join(OUTPUT_DIR, output_file_name))

    print(f"\nInference complete. Predictions saved to '{OUTPUT_DIR}' folder.")


if __name__ == '__main__':
    # To run this, save it as `predict.py` and execute `python predict.py`
    predict()