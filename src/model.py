
import numpy as np
import tensorflow as tf
import os
import cv2
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers import LeakyReLU, ReLU
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam


#from ..cfg.config import debug, info, warning, log_config


IMG_SIZE = 256
# revoie la boucle et verifier si on construit pas deux fois les layers
class Encoder(Model):
    def __init__(self,hid_dim = 512, init_fm = 16, max_filter = 512 ):
        super(Encoder, self).__init__()
        self.nb_layers = int(np.log2(hid_dim/init_fm))
        layer_filter = [init_fm]
        for i in range (self.nb_layers):
            layer_filter.append(2*layer_filter[-1])
        #print(layer_filter)
        self.input_layer =  Conv2D(init_fm, (4,4),strides=(2,2),padding='same', input_shape = (IMG_SIZE, IMG_SIZE, 3))
        self.hid_layer = []

        for i in layer_filter[1:]:
            self.hid_layer.append(Conv2D(i, (4,4),strides=(2,2),padding='same'))

        self.output_layer = Conv2D(max_filter, (4,4),strides=(2,2),padding='same')

    def call(self, inputs, training=None, **kwargs):
        x = self.input_layer(inputs)
        x=BatchNormalization()(x)
        #BatchNormalization()
        x=LeakyReLU(alpha=0.2)(x)
        for i in range(self.nb_layers):
            x = self.hid_layer[i](x)
            x=BatchNormalization()(x)
            #BatchNormalization()
            x=LeakyReLU(alpha=0.2)(x)

        x = self.output_layer(x)
        x=BatchNormalization()(x)
        #BatchNormalization()
        x=LeakyReLU(alpha=0.2)(x)
        return x



class Discriminator(Model):
    def __init__(self,n_attr = 1 ):
        super(Discriminator, self).__init__()
        self.inputs_shape = (2,2,512)
        self.layer1 = Conv2DTranspose(512,(4,4),strides=(2,2),padding='same', input_shape = self.inputs_shape)
        self.layer2 = Dense(512, input_shape=(512,), activation=None)
        self.layer3 = Dense(n_attr,input_shape=(512,), activation=None)

    def call(self, inputs, training=None, **kwargs):
        x = self.layer1(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.3)(x)
        x = Flatten()(x)
        x = self.layer2(x)
        x = LeakyReLU(alpha = 0.2)(x)
        x = self.layer3(x)
        x = Activation('sigmoid')(x)
        return x



class Decoder(Model):
    def __init__(self,nbr_attr = 1, img_fm = 3, max_filter = 512,init_fm = 16, disc = False ):
        super(Decoder, self).__init__()

        self.nb_layers = int(np.log2(max_filter/init_fm))
        if disc:
            self.latent_dim = (2,2,max_filter + nbr_attr) 
            #self.latent_dim = (2,2,max_filter + 2*nbr_attr)
        else:
            self.latent_dim = (2,2,max_filter)
            
        
        filter_layer = [max_filter]
        for i in range (self.nb_layers):
            filter_layer.append(filter_layer[-1]/2)
        self.input_layer = Conv2DTranspose(max_filter, (4,4),strides=(1,1),padding='same', input_shape = self.latent_dim)
        self.hid_layer = []
        for i in filter_layer:   
            self.hid_layer.append(Conv2DTranspose(i, (4,4),strides=(2,2),padding='same'))
        self.output_layer = Conv2DTranspose(img_fm, (4,4),strides=(2,2),padding='same')


    def call(self, inputs, training=None, **kwargs):

        x = self.input_layer(inputs)
        x=BatchNormalization()(x)
        #BatchNormalization()
        x=ReLU()(x)
        for i in range(self.nb_layers+1):
            x = self.hid_layer[i](x)
            x=BatchNormalization()(x)
            #BatchNormalization()
            x=ReLU()(x)
            x=Dropout(0.3)(x)
        x = self.output_layer(x)
        #vérifier que x2 est de dimension (256, 256)

        #valeur de l'image entre -1 et 1
        x = tf.math.tanh(x)        # a verifier 
        return x

# #################  revoir  ##################
# def input_decode(z,y):
#     y=tf.keras.utils.to_categorical(y, num_classes=2)
#     y=tf.transpose(tf.stack([y]*4),[1, 0, 2])

#     xx=2
#     yy=2
#     y=tf.reshape(y, [-1, xx, yy, 2*40])
    
#     z=tf.concat([z,y],3)
#     return z
################################################
def input_decode(z, y):
    z = tf.cast(z, tf.float32)
    y = tf.cast(y, tf.float32)
    y_ = np.reshape(y, (y.shape[0],-1))
    n_attr = y_.shape[1]
    y = tf.expand_dims(y, axis = -1)
    y = tf.expand_dims(y, axis = -1)
    y = tf.repeat(y, 2, axis = -1)
    y = tf.repeat(y, 2, axis = -2)
    z = tf.reshape(z,(-1,512,2,2))
    zy = tf.concat((z,y), axis = 1)
    zy = tf.reshape(zy,(-1,2,2,512+n_attr))

    return zy


# Simple Autoencoder
class AutoEncoder(Model):
    def __init__ (self, encoder, decoder, iSdis = True):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    
    def call(self, input, att = None):
        '''
        Model forward pass, when we use our model
        args:
            inputs : Model inputs
        return:
            x_reconstruct : Output of the model 
        '''
        z = self.encoder(input)
        if att == None:
            return z
        else:
            dec_input = input_decode(z,att)
            x_reconstruct = self.decoder(dec_input)
            return x_reconstruct,z

    
    def train_step(self, input):
        '''
        Implementation of the training update.
        Receive an input, compute loss, get gradient, update weights and return metrics.
        Here, our metrics is the loss reconstruction.
        args:
            inputs : Model inputs
        return:
            r_loss  : Reconstruction loss
            
        '''
        
        # ---- Get the input we need, specified in the .fit()
        #
        if isinstance(input, tuple):
            input = input[0]
        
        #
        with tf.GradientTape() as tape:
            
            # ---- Get encoder outputs
            
            z = self.encoder(input)
            
            # ---- Get reconstruction from decoder
            #
            reconstruction       = self.decoder(z)
         
            # ---- Compute loss
            #
            reconstruction_loss  = tf.keras.losses.binary_crossentropy(input, reconstruction)

        grads = tape.gradient(reconstruction_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        return {
            "r_loss":   reconstruction_loss,
        }


    def save(self,filename):
        '''Save model'''
        filename, extension = os.path.splitext(filename)
        self.encoder.save(f'{filename}-encoder.h5')
        self.decoder.save(f'{filename}-decoder.h5')

    def reload(self,filename):
        '''Reload the two parts of AutoEncoder'''
        filename, extension = os.path.splitext(filename)
        self.encoder = load_model(f'{filename}-encoder.h5')
        self.decoder = load_model(f'{filename}-decoder.h5')
        print('Reloaded.')


#if __name__ == '__main__':

    # tmp = np.load('D:\M2\ML\Projet\Fader_N\Fader_Network\src\Fader_Network.npy')
    # enc = Encoder()
    # dec = Decoder()
    # dec.compile(optimizer=Adam())
    # ae = AutoEncoder(enc, dec)
    # ae.compile(optimizer=Adam())
    # #data  = tmp
    # # # le rajout d'un reshape (-1, 256,256,3) n'est obligatoire que lorsqu'on donne une suele image à model
    # #data = data.reshape(-1, 256,256,3)
    # i = cv2.imread('D:/M2/ML/Projet/Fader_N/Fader_Network/data/train/000001.jpg')
    # print(i)
    # # cv2.imshow('test',i)
    # # cv2.waitKey(0)
    # print(i / 127.5 - 1)
    #print(enc(tmp[0:3]).shape)
    # print(data.shape)
    # enc.compile()
    # print(enc(data))
    # print(enc.get_weights())
    # dis = Discriminator()
    # print(dis(np.ones((2,2,2,512))))
    # print(dis.get_weights())
    #print(dec(np.ones((2,2,2,512))))
    #history = ae.fit(data, epochs=1, batch_size=16)
