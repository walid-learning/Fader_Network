
"""
Created on Mon Oct 31 09:35:49 2022

@author: bouinsalome
"""

import numpy as np
import tensorflow as tf
import os
from keras.models import Sequential, Model, load_model
from model import Encoder, Decoder, Discriminator,AutoEncoder, input_decode


class GAN(Model):
    '''
    Our model, built from given encoder and decoder and discriminator
    '''
    def __init__(self, encoder=Encoder(), decoder=Decoder(), discriminator = Discriminator(), **kwargs):
        '''
        GAN instantiation with encoder, decoder and discriminator
        args :
            encoder : Encoder model
            decoder : Decoder model
            discriminator : Discriminator model
        return:
            None
        '''
        super(GAN, self).__init__(**kwargs)
        self.encoder       = encoder
        self.decoder       = decoder
        self.discriminator = discriminator
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
        self.ae = AutoEncoder(self.encoder, self.decoder)


    def combine_model(self,img, att):
        """ Args:
                img : image
                att: attribut
            return : 
                z : latent space
                y_predict: attribut predict for discriminator
                x_reconstruct: image reconstruct for decoder
        """
        #img = img.reshape(-1,256,256,3)
        z = self.encoder(img)
        y_predict = self.discriminator(self.gaussian_noise(z))
        # y_predict = self.discriminator(z)
        attr = 1 - att
        z_ = input_decode(z, attr)
        x_reconstruct = self.decoder(z_)


        return z, y_predict, x_reconstruct

    def get_loss(self):
        loss_ae = tf.keras.losses.MeanSquaredError()
        loss_discrimintor = tf.keras.losses.BinaryCrossentropy()                      # pttr à modifier
        # loss_discrimintor = tf.keras.losses.CategoricalCrossentropy()
        return loss_ae, loss_discrimintor


    def compile(self):
        self.discriminator.compile(
            optimizer= tf.keras.optimizers.Adam(),
            loss = self.get_loss()[0])
        
        self.ae.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=self.get_loss()[1]
        )

    # def compile(self, weights=""):

    #     if weights =="":
    #         self.discriminator.compile(
    #             optimizer= tf.keras.optimizers.Adam(),
    #             loss = self.get_loss()[0])
            
    #         self.ae.compile(
    #             optimizer=tf.keras.optimizers.Adam(),
    #             loss=self.get_loss()[1]
    #         )

    #     else:
    #         # self.discriminator.built = True
    #         self.discriminator.load_weights(weights[0])
    #         # self.ae.built = True
    #         self.ae.load_weights(weights[1])
    

    def train_step(self, img, att, lamda_e):

        loss_ae, loss_discriminator = self.get_loss()

        # ---- Train the discriminator ----------------------------------------
        self.discriminator.trainable = True
        self.ae.trainable = False

        with tf.GradientTape() as Tape:
            z, y_predict, x_reconstruct = self.combine_model(img, att)
            
            loss_diss = loss_discriminator(att, y_predict)
            loss_diss2 = 1 - loss_diss

        # ---- Backward pass
        #
        # Retrieve gradients from gradient_tape and run one step 
        # of gradient descent to optimize trainable weights

        gradient_diss = Tape.gradient(loss_diss, self.discriminator.trainable_weights)
        # Update discriminator weights
        self.discriminator.optimizer.apply_gradients(zip(gradient_diss, self.discriminator.trainable_weights))

        # ---- Train the generator --------------------------------------------
        self.discriminator.trainable = False
        self.ae.trainable = True

        with tf.GradientTape() as Tape:
            z, y_predict, x_reconstruct = self.combine_model(img, att)
            flipt_attr = 1 - att
            loss_reconstruct = loss_ae(img, x_reconstruct)
            loss_model = loss_reconstruct + lamda_e*loss_discriminator(flipt_attr, y_predict)

        # ---- Backward pass
        #
        # Retrieve gradients from gradient_tape and run one step 
        # of gradient descent to optimize trainable weights
        gradient_rec = Tape.gradient(loss_model, self.ae.trainable_weights)

        # Update autoencoder weights
        self.ae.optimizer.apply_gradients(zip(gradient_rec, self.ae.trainable_weights))

    
        
        return loss_model, loss_diss, loss_reconstruct, x_reconstruct


    def gaussian_noise(self, z):
        shape = z.shape
        gauss = np.random.normal(0, 0.05, shape)
        nLatent_R = z + gauss
        return nLatent_R
    
    # def save(self,filename):
    #         '''Save model in 2 part'''
    #         save_dir             = os.path.dirname(filename)
    #         filename, _extension = os.path.splitext(filename)
    #         # ---- Create directory if needed
    #         os.makedirs(save_dir, mode=0o750, exist_ok=True)
    #         # ---- Save models
    #         self.discriminator.save_weights( f'{filename}-discriminator.h5')
    #         self.ae.save_weights(     f'{filename}-ae.h5'     )


    def reload(self,filename):
        '''Reload a 2 part saved model.
        Note : to train it, you need to .compile() it...'''
        # filename, extension = os.path.splitext(filename)
        self.discriminator.load_weights(f'{filename}')
        # gan_model-discriminator.h5') , compile=False
        self.ae.load_weights(f'{filename}gan_model-ae.h5'    ) # , compile=False
        print('Reloaded.')



