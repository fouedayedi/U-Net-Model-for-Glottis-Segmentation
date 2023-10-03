from DataLoader import DataLoader
from UNetTrainer import UNetTrainer
from models.unet import simple_unet_model
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils.metrics import iou_metric,dice_coefficient
import time


if __name__ == "__main__":
    directory_path = "/home/foued/GlotissSegmentation/data/training"
    batch_size=16
    epochs=25
    subset=0.5
  
    data_loader = DataLoader(directory_path, subset)
    data_loader.display_samples()
    
    image_dataset_vecteur, mask_dataset_vecteur = data_loader.get_normalized_data()
    print(image_dataset_vecteur.shape)

    model = simple_unet_model(256, 256, 1) 
    
    start_time = time.time() 
    #loaded_model = load_model('model_checkpoint.hdf5', custom_objects={'iou_metric': iou_metric, 'dice_loss': dice_loss})
    # trainer = UNetTrainer(loaded_model, image_dataset, mask_dataset)
    trainer = UNetTrainer(model, image_dataset_vecteur, mask_dataset_vecteur)
    trainer.compile_model()  
    trainer.train(batch_size, epochs)
    
    end_time = time.time() 
    total_time = end_time - start_time
    print(f"Total training time for epochs was {total_time:.2f} seconds.")
    
    trainer.save(batch_size, epochs,subset) 
