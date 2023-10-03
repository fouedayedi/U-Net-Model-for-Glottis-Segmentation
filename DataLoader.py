import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from PIL import Image

class DataLoader:
    def __init__(self, directory_path,subset):
        self.directory_path = directory_path
        self.image_dataset = []
        self.mask_dataset = []
        self.subset=subset
        self.load_data()

    def load_data(self):
        all_files = [f for f in os.listdir(self.directory_path) if f.endswith('.png')]
        all_files.sort()
        
        subset_length = int(self.subset * len(all_files))
        all_files_subset = all_files[:subset_length]

        image_files = [f for f in all_files_subset if not '_seg.png' in f]
        mask_files = [f.replace('.png', '_seg.png') for f in image_files]

        self.image_dataset = self._load_images_or_masks(image_files)
        self.mask_dataset = self._load_images_or_masks(mask_files)

    def _load_images_or_masks(self, files_list):
        dataset = []
        for file in files_list:
            file_path = os.path.join(self.directory_path, file)
            image_or_mask = Image.open(file_path).convert("L")
            image_or_mask = image_or_mask.resize((256, 256))
            dataset.append(np.array(image_or_mask))
        return np.array(dataset)

    def display_samples(self, num_samples=5):
        for i in range(num_samples):
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))

            ax[0].imshow(self.image_dataset[i].squeeze(), cmap='gray')
            ax[0].set_title(f"Original Image {i}")

            ax[1].imshow(self.mask_dataset[i].squeeze(), cmap='gray')
            ax[1].set_title(f"Mask {i}")

            plt.show()

    def get_normalized_data(self):
        image_dataset_normalized = self.image_dataset.astype(np.float32) / 255.0
        image_dataset_vecteur = np.expand_dims(image_dataset_normalized, axis=-1)

        mask_dataset_normalized = self.mask_dataset.astype(np.float32) / 255.0 
        mask_dataset_vecteur = np.expand_dims(mask_dataset_normalized, axis=-1)

        return image_dataset_vecteur, mask_dataset_vecteur

""" 
directory_path = "/home/foued/GlotissSegmentation/data/test"
data_loader = DataLoader(directory_path)
data_loader.display_samples() """