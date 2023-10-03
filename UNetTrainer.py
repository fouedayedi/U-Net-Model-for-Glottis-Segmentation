from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from utils.metrics import iou_metric, dice_loss
import pickle

class UNetTrainer:
    def __init__(self, model, image_dataset, mask_dataset):
        self.model = model
        self.image_dataset = image_dataset
        self.mask_dataset = mask_dataset

    def compile_model(self, optimizer=Adam(learning_rate=1e-3)):
        self.model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=[iou_metric, dice_loss])

    def train(self, batch_size=8, epochs=25):
        train_images, test_images, train_masks, test_masks = train_test_split(self.image_dataset, self.mask_dataset, train_size=0.9, shuffle=True)

        early_stopping = EarlyStopping(monitor='val_dice_loss', patience=5, verbose=1, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_dice_loss', factor=0.1, patience=3, verbose=1, min_delta=1e-7, min_lr=1e-6)
        checkpoint = ModelCheckpoint('model_checkpoint.hdf5', monitor='val_dice_loss', verbose=1, save_best_only=True, mode='min', save_freq='epoch')

        history = self.model.fit(train_images, train_masks, 
                             batch_size=batch_size, 
                             epochs=epochs, 
                             verbose=1, 
                             validation_data=(test_images, test_masks),
                             callbacks=[early_stopping, reduce_lr, checkpoint])  
    
        with open('training_history_batchsize_{batch_size}_epochs_{epochs}_subset.pkl', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        
        return history

    def save(self, batch_size, epochs, subset):
        model_filename = f"unet_model_batchsize_{batch_size}_epochs_{epochs}_subset_{subset}.hdf5"
        self.model.save(model_filename)
        print(f"Model saved as {model_filename}")



