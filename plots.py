import numpy as np
import matplotlib.pyplot as plt

# Load the history file
history = np.load('test_predictions/my_drive_unet_history.npy', allow_pickle=True).item()

plt.figure(figsize=(12, 5))

# Plot training & validation loss values
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'])
plt.plot(history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.grid(True)

# Plot training & validation dice scores
plt.subplot(1, 2, 2)
plt.plot(history['val_dice'])
plt.title('Model Dice Score')
plt.ylabel('Dice Score')
plt.xlabel('Epoch')
plt.legend(['Validation'], loc='upper left')
plt.grid(True)

plt.tight_layout()
plt.show()