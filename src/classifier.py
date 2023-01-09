import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense
from keras.layers import BatchNormalization, Activation
from keras.layers import LeakyReLU, Conv2D
import keras.layers 
from keras.models import Sequential, Model
from keras.optimizers import Adam
import os
import matplotlib as plt
from model import Encoder
import argparse
from train import Loader
from config import debug, info, warning, log_config, PATH, save_loss
import glob
import cv2


class Classifier(Model):
    def __init__(self,attributes):

        self.shape = (256,256,3)
        self.nbr_attributes = len(attributes)
        self.image_path = glob.glob(PATH + '\\data\\train' + '\\*.jpg')
        self.attributs = attributes
        

        super(Classifier,self).__init__()

        self.encoder = Encoder()
        self.conv2d_layer = tf.keras.layers.Conv2D(512, 4, 2, 'same', activation=LeakyReLU(0.2)) 
        self.bach_normalisation = tf.keras.layers.BatchNormalization()
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layer1 = tf.keras.layers.Dense(512, activation = LeakyReLU(0.2))
        self.dense_layer2 = tf.keras.layers.Dense(self.nbr_attributes,activation='sigmoid')

        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0002)


    def call(self):
            
        inputs = tf.keras.layers.Input(shape=self.shape)
        x = self.encoder(inputs)
        x = self.conv2d_layer(x)
        x = self.bach_normalisation(x)
        # x = self.flatten_layer(x)
        x = self.dense_layer1(x)
        x = self.dense_layer2(x)
        x = tf.keras.layers.Reshape((self.nbr_attributes,))(x)
        return x

    def attributs(self):
        return self.attributs

class classification(Model):
    def __init__(self, classifier=Classifier(["Smiling"])):
        super(classification, self).__init__()
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0002)
        self.accuracy = tf.keras.metrics.BinaryAccuracy()
        # super(classification, self).__init__()
        self.classifier = classifier
        self.attributs = self.classifier.attributs
        

    def get_metrics(self):
        metric_optimizer = self.optimizer
        metric_loss = self.loss_fn
        metric_accuracy = self.accuracy
        return metric_optimizer, metric_accuracy, metric_loss

    def train_step(self, images,attributs):

        Optimizer, Accuracy, Loss = self.get_metrics()
        # inputs is our batch images
        # batch_size=tf.shape(images)[0]
        print(images.shape)
        with tf.GradientTape() as tape:
            predictions = self.classifier(images)
            print('prediction',predictions)
            print('attributs',attributs)
            c_loss = Loss(attributs,predictions)

        #Backward
        grads = tape.gradient(c_loss, self.classifier.trainable_weights)
        Optimizer.apply_gradients( zip(grads, self.classifier.trainable_weights) )

        # self.c_loss.update_state(c_loss)

        # return {"c_loss": self.c_loss_metric.result()}


    def training(self, epochs, batch_size):
        name_attributs = self.attributs
        ld = Loader()
        nbr_itr_per_epoch = 200 #int(len(self.image_path)/batch_size)

        # info('start training') 
        for epoch in range (epochs):
         
            for i in range (nbr_itr_per_epoch):
                lambda_e = 0.0001 * (epoch*nbr_itr_per_epoch + i)/(nbr_itr_per_epoch*epochs)

                imgs, atts = ld.Load_Data(batch_size,i, name_attributs)
                self.train_step(im gs, atts)
        
        return


if __name__ == '__main__':


    # parser = argparse.ArgumentParser(description='Classifier')
    # parser.add_argument("--img_path", type = str, default = "data/img_align_celeba_resized", help= "Path to images")
    # parser.add_argument("--attr_path" ,type = str, default = "data/attributes.npz", help = "path to attributes")
    # parser.add_argument("--batch_size", type = int, default = 32, help= "Size of the batch used during the training")
    # parser.add_argument("--attr", type = str, default= "*", help= "Considered attributes to train the network with")
    # parser.add_argument("--n_epoch", type = int, default = 5, help = "Numbers of epochs")
    # parser.add_argument("--epoch_size", type = int, default = 50000, help = "Number of images seen at each epoch")
    # parser.add_argument("--n_images", type = int, default = 202599, help = "Number of images")

    classi = classification()
    classi.training(4,32)
