from keras.layers import Input, Conv2D, MaxPooling2D, concatenate
from keras.layers import Conv2DTranspose, Dropout
from keras.models import Model

class Encoder:
    @staticmethod
    def conv_block(inputs, num_filters):
        """A helper function to create a convolutional block"""
        x = Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
        x = Dropout(0.1)(x)
        x = Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
        return x

    @staticmethod
    def encode(inputs):
        c1 = Encoder.conv_block(inputs, 16)
        p1 = MaxPooling2D((2, 2))(c1)
        
        c2 = Encoder.conv_block(p1, 32)
        p2 = MaxPooling2D((2, 2))(c2)
        
        c3 = Encoder.conv_block(p2, 64)
        p3 = MaxPooling2D((2, 2))(c3)
        
        c4 = Encoder.conv_block(p3, 128)
        p4 = MaxPooling2D((2, 2))(c4)
        
        c5 = Encoder.conv_block(p4, 256)
        
        return c1, c2, c3, c4, c5

class Decoder:
    @staticmethod
    def decode(c1, c2, c3, c4, c5):
        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        c6 = Encoder.conv_block(u6, 128)
        
        u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = Encoder.conv_block(u7, 64)
        
        u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = Encoder.conv_block(u8, 32)
        
        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1])
        c9 = Encoder.conv_block(u9, 16)
        
        return c9

def simple_unet_model(img_height, img_width, img_channels):
    inputs = Input((img_height, img_width, img_channels))

    c1, c2, c3, c4, c5 = Encoder.encode(inputs)
    c9 = Decoder.decode(c1, c2, c3, c4, c5)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model
