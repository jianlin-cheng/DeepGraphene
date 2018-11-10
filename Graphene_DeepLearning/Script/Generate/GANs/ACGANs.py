# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 13:25:35 2018

@author: Herman Wu
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#                                      

from __future__ import print_function

from collections import defaultdict

from six.moves import range
from keras import layers
import keras.backend as K
from keras.layers import Input, Dense, Activation,Reshape,BatchNormalization,Flatten, Embedding, merge, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv2D,Conv2DTranspose
from keras.layers.convolutional import UpSampling2D
from keras.models import Sequential, Model
from keras import optimizers
from keras.utils.generic_utils import Progbar
from keras.utils import np_utils   
from keras.layers.noise import GaussianNoise
import numpy as np
import random
import time

import normLib

load_weight = False
Class_num=20

np.random.seed(1337)

K.set_image_data_format('channels_first')


class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("\nThe running time of this code: %s " % self.elapsed(time.time() - self.start_time) )
        
def build_generator(latent_size):
    cnn = Sequential()
    cnn.add(Dense(1024, input_dim=latent_size, activation='relu',
                  kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    cnn.add(Dense(128 * 2 * 2, activation='relu',kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    cnn.add(Reshape((128, 2, 2)))

    cnn.add(UpSampling2D(size=(2, 2)))

    cnn.add(Conv2D(128,kernel_size=3,strides=1, padding='same',
                           activation='relu', init='glorot_normal'))    
 #   cnn.add(BatchNormalization())
#    cnn.add(LeakyReLU(alpha=0.2))
    
    cnn.add(Conv2D(64,kernel_size=2,strides=1, padding='same',
                          activation='relu', init='glorot_normal'))    
 #   cnn.add(BatchNormalization())
#    cnn.add(LeakyReLU(alpha=0.2))
    
    cnn.add(Conv2D(32,kernel_size=2,strides=1, padding='same',
                           activation='relu', init='glorot_normal')) 
 #   cnn.add(BatchNormalization())
#    cnn.add(LeakyReLU(alpha=0.2))    

    cnn.add(Conv2D(1,kernel_size=2,strides=1, padding='same',
                           activation='relu', init='glorot_normal'))    
#    cnn.add(BatchNormalization())
#    cnn.add(LeakyReLU(alpha=0.2))
# =============================================================================
#     cnn.add(Conv2DTranspose(64, kernel_size=2, strides=1, padding='same', activation='relu',
#                             kernel_initializer='glorot_normal', bias_initializer='Zeros'))
#     cnn.add(BatchNormalization())
# 
#     cnn.add(Conv2DTranspose(32, kernel_size=2, strides=2, padding='same', activation='relu',
#                             kernel_initializer='glorot_normal', bias_initializer='Zeros'))
#     cnn.add(BatchNormalization())
# 
#     cnn.add(Conv2DTranspose(1, kernel_size=2, strides=1, padding='same', activation='relu',
#                             kernel_initializer='glorot_normal', bias_initializer='Zeros'))
# 
# =============================================================================
    # this is the z space commonly refered to in GAN papers
    latent = Input(shape=(latent_size,))

    image_class = Input(shape=(1,), dtype='int32')

    cls = Flatten()(Embedding(Class_num, latent_size,
                              embeddings_initializer='glorot_normal')(image_class))

    # hadamard product between z-space and a class conditional embedding
    h = layers.multiply([latent, cls])

    fake_image = cnn(h)

    return Model([latent, image_class], fake_image)


def build_discriminator():
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper
    cnn = Sequential()

    cnn.add(GaussianNoise(0.05, input_shape=(1, 4, 4)))  # Add this layer to prevent D from overfitting!

    cnn.add(Conv2D(16, kernel_size=2, strides=1, padding='same',
                   kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    cnn.add(LeakyReLU(alpha=0.2))
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(32, kernel_size=2, strides=2, padding='same',
                   kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    cnn.add(BatchNormalization())
    cnn.add(LeakyReLU(alpha=0.2))
    cnn.add(Dropout(0.3))

    cnn.add(Flatten())
    
#    cnn.add(Dense(64,kernel_initializer='glorot_normal',bias_initializer='Zeros'))
#    cnn.add(LeakyReLU(alpha=0.2))
#    cnn.add(Dropout(0.4))
    
#    cnn.add(Dense(32,kernel_initializer='glorot_normal',bias_initializer='Zeros'))
#    cnn.add(LeakyReLU(alpha=0.2))
#    cnn.add(Dropout(0.4))

   # cnn.add(MinibatchDiscrimination(50, 30))

    image = Input(shape=(1, 4, 4))

    features = cnn(image)

    # first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.

    fake = Dense(1, activation='sigmoid', name='generation',
                 kernel_initializer='glorot_normal', bias_initializer='Zeros')(features)
    aux = Dense(Class_num, activation='softmax', name='auxiliary',
                kernel_initializer='glorot_normal', bias_initializer='Zeros')(features)

    return Model(image, [fake, aux])

def load_data(X,Y,percent):
    index_test=sorted(random.sample(range(0,len(X)),int(percent*len(X))))
    x_test=[]
    y_test=[]
    x_train=[]
    y_train=[]
    for i in range(0,len(X)):
        if i in index_test:
            x_test.append(X[i])
            y_test.append(Y[i])
        else:
            x_train.append(X[i])
            y_train.append(Y[i])
    x_test=(np.array(x_test,dtype=float)).reshape(-1,1,4,4)
    y_test=np.array(y_test,dtype='uint8')
    x_train=(np.array(x_train,dtype=float)).reshape(-1,1,4,4)
    y_train=np.array(y_train,dtype='uint8')
    return x_train,y_train,x_test,y_test


def main(Base_dir,X0,y0,nb_epochs=100,batch_size=10):
    # batch and latent size taken from the paper
    latent_size = 100
    adam_lr = 0.0002
    adam_beta_1 = 0.5
    dis_adam=optimizers.Adam(lr=adam_lr, beta_1=adam_beta_1)
    gen_adam=optimizers.Adam(lr=(adam_lr*5), beta_1=adam_beta_1)    
    # build the discriminator
    discriminator = build_discriminator()
    discriminator.compile(optimizer=dis_adam,loss=['binary_crossentropy', 'categorical_crossentropy'])
    discriminator.summary()
    # build the generator
    generator = build_generator(latent_size)
    generator.compile(optimizer=dis_adam,loss='binary_crossentropy')
    generator.summary()
    
    latent = Input(shape=(latent_size, ))
    image_class = Input(shape=(1,), dtype='int32')

    # get a fake image
    fake = generator([latent, image_class])
    # we only want to be able to train generation for the Adversarial model
    discriminator.trainable = False
    fake, aux = discriminator(fake)
    Adversarial_Net = Model(input=[latent, image_class], output=[fake, aux])
    Adversarial_Net.compile(optimizer=gen_adam,loss=['binary_crossentropy', 'categorical_crossentropy'])
#    Adversarial_Net.compile(optimizer=gen_adam,loss=['binary_crossentropy', 'categorical_crossentropy'])    
    Adversarial_Net.summary()

    timer = ElapsedTimer()      
###########################################################################################################   
    [X_train, y_train, X_test, y_test] =load_data(X0,y0,0.1) 
###########################################################################################################   
    y_train=y_train-6
    y_test=y_test-6
    y_train=np_utils.to_categorical(y_train,Class_num)
    y_test=np_utils.to_categorical(y_test,Class_num)
###########################################################################################################       
    X_train = (X_train.astype(np.float32)) 
    X_test = (X_test.astype(np.float32))
    nb_test =X_test.shape[0]

    train_history = defaultdict(list)
    test_history = defaultdict(list)

    if load_weight:
         generator.load_weights('params_generator_epoch_{0:03d}.hdf5'.format(load_epoch))
         discriminator.load_weights('params_discriminator_epoch_{0:03d}.hdf5'.format(load_epoch))
    else:
         load_epoch = -1

    for epoch in range(nb_epochs):
        #print('Epoch {} of {}'.format(load_epoch + 1, nb_epochs))
        load_epoch += 1
        nb_batches = int(X_train.shape[0] / batch_size)
        progress_bar = Progbar(target=nb_batches)

        epoch_gen_loss = []
        epoch_disc_loss = []

        for index in range(nb_batches):
            progress_bar.update(index)
            # generate a new batch of noise
            noise = np.random.normal(0, 0.5, (batch_size, latent_size))
#            noise = np.random.uniform(-1, 1, (batch_size, latent_size))

            # get a batch of real images
            image_batch = X_train[index * batch_size:(index + 1) * batch_size]
            label_batch = y_train[index * batch_size:(index + 1) * batch_size]
###########################################################################################################              
            sampled_labels_index = np.random.randint(0, Class_num, batch_size)
            sampled_labels=np_utils.to_categorical(sampled_labels_index,Class_num)
###########################################################################################################  
            generated_images = generator.predict(
                [noise, sampled_labels_index.reshape((-1, 1))], verbose=0)


            for train_ix in range(3):
                if index % 30 != 0:
                    X_real = image_batch
                    # Label Soomthing
                    y_real = np.random.uniform(0.7, 1.2, size=(batch_size,))
###########################################################################################################  
#                    aux_y1 = label_batch.reshape(-1, )
                    aux_y1=label_batch
###########################################################################################################  
                    epoch_disc_loss.append(discriminator.train_on_batch(X_real, [y_real, aux_y1]))
                    # Label Soomthing
                    X_fake = generated_images
                    y_fake = np.random.uniform(0.0, 0.3, size=(batch_size,))
                    aux_y2 = sampled_labels

                    # see if the discriminator can figure itself out...
                    epoch_disc_loss.append(discriminator.train_on_batch(X_fake, [y_fake, aux_y2]))
                else:
                    # make the labels the noisy for the discriminator: occasionally flip the labels
                    # when training the discriminator
                    X_real = image_batch
#                    y_real = np.random.uniform(0.0, 0.3, size=(batch_size,))
                    y_real = np.random.uniform(0.0, 0.3, size=(batch_size,))
###########################################################################################################  
#                    aux_y1 = label_batch.reshape(-1, )
                    aux_y1=label_batch
###########################################################################################################  
                    epoch_disc_loss.append(discriminator.train_on_batch(X_real, [y_real, aux_y1]))
                    # Label Soomthing
                    X_fake = generated_images
                    y_fake = np.random.uniform(0.7, 1.2, size=(batch_size,))
                    aux_y2 = sampled_labels

                    # see if the discriminator can figure itself out...
                    epoch_disc_loss.append(discriminator.train_on_batch(X_fake, [y_fake, aux_y2]))
            # make new noise. we generate Guassian Noise rather than Uniform Noise according to GANHACK
            noise = np.random.normal(0, 0.5, (2 * batch_size, latent_size))
#            noise = np.random.uniform(-1, 1, (2* batch_size, latent_size))
###########################################################################################################  
            sampled_labels_index = np.random.randint(0, Class_num, 2 * batch_size)
#            sampled_labels=sampled_labels_index-6
#            sampled_labels=np_utils.to_categorical(sampled_labels,Class_num)
            sampled_labels=np_utils.to_categorical(sampled_labels_index,Class_num)
###########################################################################################################  
            # we want to train the generator to trick the discriminator
            # For the generator, we want all the {fake, not-fake} labels to say
            # not-fake
            trick = np.random.uniform(0.7, 1.2, size=(2 * batch_size,))
#            trick = np.ones(batch_size*2)
            epoch_gen_loss.append(Adversarial_Net.train_on_batch(
                [noise, sampled_labels_index.reshape((-1, 1))], [trick, sampled_labels]))
        # generate a new batch of noise
        noise = np.random.normal(0, 0.5, (nb_test, latent_size))
        
        # sample some labels from p_c and generate images from them
        sampled_labels_index = np.random.randint(0, Class_num, nb_test)
        sampled_labels=np_utils.to_categorical(sampled_labels_index,Class_num)       
        generated_images = generator.predict(
            [noise, sampled_labels_index.reshape((-1, 1))], verbose=False)

        X = np.concatenate((X_test, generated_images))
        y = np.array([1] * nb_test + [0] * nb_test)
###########################################################################################################  
#        aux_y = np.concatenate((y_test.reshape(-1, ), sampled_labels), axis=0)
        aux_y = np.concatenate((y_test, sampled_labels), axis=0)
###########################################################################################################  
        # see if the discriminator can figure itself out...
        discriminator_test_loss = discriminator.evaluate(
            X, [y, aux_y], verbose=False)

        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

        # make new noise
        noise = np.random.normal(0, 0.5, (2 * nb_test, latent_size))
#        noise = np.random.uniform(0.7, 1.2, (2 * nb_test, latent_size))
###########################################################################################################  
        sampled_labels_index = np.random.randint(0, Class_num, 2 * nb_test)
        sampled_labels=np_utils.to_categorical(sampled_labels_index,Class_num)                
#        sampled_labels_index = np.random.randint(6, 6+Class_num, 2 * nb_test)
#        sampled_labels=sampled_labels_index-6
#        sampled_labels=np_utils.to_categorical(sampled_labels,Class_num)        
###########################################################################################################  
        trick = np.ones(2 * nb_test)
        
        generator_test_loss = Adversarial_Net.evaluate(
            [noise, sampled_labels_index.reshape((-1, 1))],
            [trick, sampled_labels], verbose=False)

        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)
        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)
        
        if epoch%5==0:
            print('\nTesting for epoch {}:'.format(load_epoch))
            print('///////////////////////////////////////////////////////////////////////////')
            print(generator_train_loss)
            print('///////////////////////////////////////////////////////////////////////////')
            
            print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
                    'component', *discriminator.metrics_names))
            print('-' * 65)

            ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
            print(ROW_FMT.format('generator (train)',
                             *train_history['generator'][-1]))
            print(ROW_FMT.format('generator (test)',
                             *test_history['generator'][-1]))
            print(ROW_FMT.format('discriminator (train)',
                             *train_history['discriminator'][-1]))
            print(ROW_FMT.format('discriminator (test)',
                             *test_history['discriminator'][-1]))
    
#    normLib.viz_model(test_history,Base_dir,'Test',nb_epochs)
#    normLib.viz_model(train_history,Base_dir,'Train',nb_epochs)
    
    timer.elapsed_time()
    return generator,discriminator,train_history,test_history
###########################################################################################################
