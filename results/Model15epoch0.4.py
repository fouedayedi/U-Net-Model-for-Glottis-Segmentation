import numpy as np
import os
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.metrics import MeanIoU

def iou_metric(y_true, y_pred, smooth=1e-6):
    y_pred = tf.cast(y_pred > 0.5, dtype=tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_pred = tf.cast(y_pred > 0.5, dtype=tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_coefficient(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)


model_path = '/home/foued/GlotissSegmentation/unet_model_batchsize_8_epochs_10.hdf5'
model = load_model(model_path, custom_objects={"iou_metric": iou_metric, "dice_loss": dice_loss})


test_dir = "/home/foued/GlotissSegmentation/data/test"
test_image_files = [f for f in os.listdir(test_dir) if f.endswith('.png') and not '_seg.png' in f]
test_mask_files = [f.replace('.png', '_seg.png') for f in test_image_files]

test_images = []
test_masks = []

for img_file in test_image_files:
    img_path = os.path.join(test_dir, img_file)
    img = Image.open(img_path).convert("L")
    img = img.resize((256, 256))
    test_images.append(np.array(img))
    
for mask_file in test_mask_files:
    mask_path = os.path.join(test_dir, mask_file)
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((256, 256))
    test_masks.append(np.array(mask))
    
test_images = np.array(test_images).astype(np.float32) / 255.0
test_images = np.expand_dims(test_images, axis=-1)

test_masks = np.array(test_masks).astype(np.float32) / 255.0
test_masks = np.expand_dims(test_masks, axis=-1)

# Predict on test data
predictions = model.predict(test_images)
binary_predictions = (predictions > 0.5).astype(int)


loss,  iou, dice = model.evaluate(test_images, test_masks)
print(f"IOU: {iou}")
print(f"Dice Coefficient: {dice}")


num_samples = 5
for i in range(min(num_samples, len(test_images))):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    ax[0].imshow(test_images[i].squeeze(), cmap='gray')
    ax[0].set_title(f"Original Image {i}")

    ax[1].imshow(test_masks[i].squeeze(), cmap='gray')
    ax[1].set_title(f"Ground Truth {i}")

    ax[2].imshow(binary_predictions[i].squeeze(), cmap='gray')
    ax[2].set_title(f"Prediction {i}")

    plt.show()