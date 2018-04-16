# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 13:25:35 2018

@author: Herman Wu
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from collections import defaultdict
import pickle
from PIL import Image

from six.moves import range
#from Minibatch import MinibatchDiscrimination
from keras.utils import np_utils   

from keras import layers
import keras.backend as K
from keras.layers import Input, Dense, Activation,Reshape,BatchNormalization,Flatten, Embedding, merge, Dropout,Multiply
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D
from keras.layers import Conv2D,Conv2DTranspose
from keras.models import Sequential, Model
from keras import optimizers
from keras.utils.generic_utils import Progbar
from keras.utils import np_utils   
from keras.layers.noise import GaussianNoise
import numpy as np
import random
import time
from keras.datasets import cifar10

#from keras.datasets import mnist

load_epoch = 512
load_weight = False
Class_num=19

np.random.seed(1337)

#K.set_image_data_format('channels_last')
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
        print("The running time of this code: %s " % self.elapsed(time.time() - self.start_time) )
        
def build_generator(latent_size):
    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 3, 32, 32)
    cnn = Sequential()
    cnn.add(Dense(128 * 2 * 2, input_dim=latent_size, activation='relu',
                  kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    cnn.add(Reshape((128, 2, 2)))

    cnn.add(Conv2DTranspose(64, kernel_size=2, strides=1, padding='same', activation='relu',
                            kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    cnn.add(BatchNormalization())

    cnn.add(Conv2DTranspose(32, kernel_size=2, strides=2, padding='same', activation='relu',
                            kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    cnn.add(BatchNormalization())

    cnn.add(Conv2DTranspose(1, kernel_size=2, strides=1, padding='same', activation='tanh',
                            kernel_initializer='glorot_normal', bias_initializer='Zeros'))

    # this is the z space commonly refered to in GAN papers
    latent = Input(shape=(latent_size,))

    # this will be our label
    image_class = Input(shape=(1,), dtype='int32')

    # 10 classes in CIFAR-10
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

    cnn.add(Conv2D(16, kernel_size=3, strides=2, padding='same',
                   kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    cnn.add(LeakyReLU(alpha=0.2))
    cnn.add(Dropout(0.5))

    cnn.add(Conv2D(32, kernel_size=2, strides=1, padding='same',
                   kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    cnn.add(BatchNormalization())
    cnn.add(LeakyReLU(alpha=0.2))
    cnn.add(Dropout(0.5))

    cnn.add(Conv2D(64, kernel_size=2, strides=1, padding='same',
                   kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    cnn.add(BatchNormalization())
    cnn.add(LeakyReLU(alpha=0.2))
    cnn.add(Dropout(0.5))

    cnn.add(Conv2D(64, kernel_size=2, strides=1, padding='same',
                   kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    cnn.add(BatchNormalization())
    cnn.add(LeakyReLU(alpha=0.2))
    cnn.add(Dropout(0.5))

    cnn.add(Conv2D(128, kernel_size=2, strides=1, padding='same',
                   kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    cnn.add(BatchNormalization())
    cnn.add(LeakyReLU(alpha=0.2))
    cnn.add(Dropout(0.5))

#    cnn.add(Conv2D(512, kernel_size=2, strides=1, padding='same',
#                   kernel_initializer='glorot_normal', bias_initializer='Zeros'))
#    cnn.add(BatchNormalization())
#    cnn.add(LeakyReLU(alpha=0.2))
#    cnn.add(Dropout(0.5))

    cnn.add(Flatten())

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


def main(X0,y0,nb_epochs=100,batch_size=10):
    # batch and latent size taken from the paper
    latent_size = 100
    adam_lr = 0.0008
    adam_beta_1 = 0.9
    optimizers.Adam(lr=adam_lr, beta_1=adam_beta_1)
    # build the discriminator
    discriminator = build_discriminator()
    discriminator.compile(optimizer='Adam',loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])
    discriminator.summary()
    # build the generator
    generator = build_generator(latent_size)
    generator.compile(optimizer='Adam',loss='binary_crossentropy')
    generator.summary()
    
    latent = Input(shape=(latent_size, ))
    image_class = Input(shape=(1,), dtype='float')

    # get a fake image
    fake = generator([latent, image_class])

    # we only want to be able to train generation for the Adversarial model
    discriminator.trainable = False
    fake, aux = discriminator(fake)
    Adversarial_Net = Model(input=[latent, image_class], output=[fake, aux])
    Adversarial_Net.compile(optimizer='Adam',loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])
    Adversarial_Net.summary()

    timer = ElapsedTimer()      
###########################################################################################################
#    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
#    y_train = y_test.astype(np.int32)
    
#    X_train = (X_test.astype(np.float32) - 127.5) / 127.5
#    X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    
    [X_train, y_train, X_test, y_test] =load_data(X0,y0,0.1)

#   # (X_train, y_train), (X_test, y_test) = cifar10.load_data()    
    X_train = (X_train.astype(np.float32)) 
    X_test = (X_test.astype(np.float32))
    
    nb_train, nb_test = X_train.shape[0], X_test.shape[0]

    train_history = defaultdict(list)
    test_history = defaultdict(list)

    if load_weight:
        generator.load_weights('params_generator_epoch_{0:03d}.hdf5'.format(load_epoch))
        discriminator.load_weights('params_discriminator_epoch_{0:03d}.hdf5'.format(load_epoch))
    else:
        load_epoch = 0

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

            # get a batch of real images
            image_batch = X_train[index * batch_size:(index + 1) * batch_size]
            label_batch = y_train[index * batch_size:(index + 1) * batch_size]

            # sample some labels from p_c
            sampled_labels = np.random.randint(0, Class_num, batch_size)

           # generate a batch of fake images, using the generated labels as a
            # conditioner. We reshape the sampled labels to be
            # (batch_size, 1) so that we can feed them into the embedding
            # layer as a length one sequence
            generated_images = generator.predict(
                [noise, sampled_labels.reshape((-1, 1))], verbose=0)

            disc_real_weight = [np.ones(batch_size), 2 * np.ones(batch_size)]
            disc_fake_weight = [np.ones(batch_size), np.zeros(batch_size)]

            # According to GANHACK, We training our ACGAN-CIFAR10 in Real->D, Fake->D,
            # Noise->G, rather than traditional method: [Real, Fake]->D, Noise->G, actully,
            # it really make sense!

            for train_ix in range(3):
                if index % 30 != 0:
                    X_real = image_batch
                    # Label Soomthing
                    y_real = np.random.uniform(0.7, 1.2, size=(batch_size,))
                    aux_y1 = label_batch.reshape(-1, )
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
                    y_real = np.random.uniform(0.0, 0.3, size=(batch_size,))
                    aux_y1 = label_batch.reshape(-1, )

                    epoch_disc_loss.append(discriminator.train_on_batch(X_real, [y_real, aux_y1]))
                    # Label Soomthing
                    X_fake = generated_images
                    y_fake = np.random.uniform(0.7, 1.2, size=(batch_size,))
                    aux_y2 = sampled_labels

                    # see if the discriminator can figure itself out...
                    epoch_disc_loss.append(discriminator.train_on_batch(X_fake, [y_fake, aux_y2]))
            # make new noise. we generate Guassian Noise rather than Uniform Noise according to GANHACK
            noise = np.random.normal(0, 0.5, (2 * batch_size, latent_size))
            sampled_labels = np.random.randint(0, Class_num, 2 * batch_size)

            # we want to train the generator to trick the discriminator
            # For the generator, we want all the {fake, not-fake} labels to say
            # not-fake
            trick = np.random.uniform(0.7, 1.2, size=(2 * batch_size,))

            epoch_gen_loss.append(Adversarial_Net.train_on_batch(
                [noise, sampled_labels.reshape((-1, 1))], [trick, sampled_labels]))


        # evaluate the testing loss here

        # generate a new batch of noise
        noise = np.random.normal(0, 0.5, (nb_test, latent_size))

        # sample some labels from p_c and generate images from them
        sampled_labels = np.random.randint(0, Class_num, nb_test)
        generated_images = generator.predict(
            [noise, sampled_labels.reshape((-1, 1))], verbose=False)

        X = np.concatenate((X_test, generated_images))
        y = np.array([1] * nb_test + [0] * nb_test)
        aux_y = np.concatenate((y_test.reshape(-1, ), sampled_labels), axis=0)

        # see if the discriminator can figure itself out...
        discriminator_test_loss = discriminator.evaluate(
            X, [y, aux_y], verbose=False)

        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

        # make new noise
        noise = np.random.normal(0, 0.5, (2 * nb_test, latent_size))
        sampled_labels = np.random.randint(0, Class_num, 2 * nb_test)
        trick = np.ones(2 * nb_test)
        generator_test_loss = Adversarial_Net.evaluate(
            [noise, sampled_labels.reshape((-1, 1))],
            [trick, sampled_labels], verbose=False)

        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

        # generate an epoch report on performance
        if epoch%5==0:
            print('\nTesting for epoch {}:'.format(load_epoch))
            print('///////////////////////////////////////////////////////////////////////////')
            print(generator_train_loss)
            print('///////////////////////////////////////////////////////////////////////////')
            train_history['generator'].append(generator_train_loss)
        
            train_history['discriminator'].append(discriminator_train_loss)

            test_history['generator'].append(generator_test_loss)
            test_history['discriminator'].append(discriminator_test_loss)

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

        # save weights every epoch
     #   generator.save_weights(
     #       'params_generator_epoch_{0:03d}.hdf5'.format(load_epoch), True)
     #   discriminator.save_weights(
     #       'params_discriminator_epoch_{0:03d}.hdf5'.format(load_epoch), True)

        # generate some pictures to display
        noise = np.random.normal(0, 0.5, (100, latent_size))
        sampled_labels = np.array([
            [i] * 10 for i in range(10)
        ]).reshape(-1, 1)
        timer.elapsed_time()
 #       generated_images = generator.predict([noise, sampled_labels]).transpose(0, 2, 3, 1)
 #       generated_images = np.asarray((generated_images).astype(float))
    return generator,discriminator
###########################################################################################################
'''
def build_generator(latent_size):
    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 1, 28, 28)
    Generator = Sequential()
    Generator.add(Dense(100,input_dim=latent_size,activation='relu',name='Gen_Dense1'))
#    Generator.add(Dense(25*2*2,activation='relu',name='Gen_Dense2'))    
#    Generator.add(Reshape((2,2,25),name='Gen_Reshape'))
    Generator.add(Dense(48*2*2,input_dim=latent_size,activation='relu',name='Gen_Dense2'))    
    Generator.add(Reshape((48,2,2),name='Gen_Reshape'))


    Generator.add(UpSampling2D(size=(2,2),name='Gen_UpSam'))
    Generator.add(Conv2D(64,(2,2),padding='same',name="Conv1"))    
    Generator.add(BatchNormalization(name='Gen_Nor1'))
    Generator.add(Activation('relu'))
    Generator.add(Conv2D(48,(2,2),padding='same',activation='selu',name="Conv2"))
    Generator.add(BatchNormalization(name='Gen_Nor2'))
    Generator.add(Activation('relu'))
    Generator.add(Conv2D(32,(2,2),padding='same',activation='selu',name="Conv3"))
    Generator.add(BatchNormalization(name='Gen_Nor3'))
    Generator.add(Activation('relu'))
    #Generator.add(Conv2D(1,(3,3),padding='same',activation='tanh',name="Out_lay"))
    Generator.add(Conv2D(1,(3,3),padding='same',activation='tanh',name="Out_lay"))
    
    latent=Input(shape=(latent_size, ))
    Gap_class=Input(shape=(1,),dtype='uint8')
        
  
    cls=Flatten()(Embedding(Class_num,latent_size,embeddings_initializer='glorot_normal')(Gap_class))
    
    Gen_Input=Multiply()([latent,cls])
    
    fake_data=Generator(Gen_Input)
    
    return Model([latent,Gap_class],fake_data)

###########################################################################################################
def build_discriminator():
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper   
    Discriminator=Sequential()
#    Discriminator.add((GaussianNoise(0.05, input_shape=(4,4,1))))       # Add this layer to prevent D from overfitting!
#    Discriminator.add(Conv2D(20,(2,2),strides=(1,1),input_shape=(4,4,1),padding='same',name="Dis_Conv1"))
    Discriminator.add((GaussianNoise(0.05, input_shape=(1,4,4))))       # Add this layer to prevent D from overfitting!
    Discriminator.add(Conv2D(40,(2,2),strides=(2,2),padding='same',name="Dis_Conv1"))
    Discriminator.add(LeakyReLU(alpha=0.2)) 
    Discriminator.add(Dropout(0.3,name='Dis_Drop1'))
    Discriminator.add(Conv2D(16,(2,2),strides=(1,1),padding='same',name="Dis_Conv2"))    
    Discriminator.add(BatchNormalization(name='Dis_Nor1'))
    Discriminator.add(LeakyReLU(alpha=0.2))
    #Discriminator.add(Dropout(0.4,name='Dis_Drop1')
    Discriminator.add(Dropout(0.4,name='Dis_Drop2'))
    
    Discriminator.add(Flatten())
    
    
   # Discriminator.add(MinibatchDiscrimination(50, 30))
    
    Discriminator.add(Dense(100,activation='relu',name='Dis_Dense1'))
    
#    data=Input(shape=(4,4,1))
    data=Input(shape=(1,4,4))  
    data_F=Discriminator(data)
    
    Verify=Dense(1,activation='sigmoid',name='Verify_authenticity')(data_F)
    Class=Dense(Class_num,activation='softmax',name='Verify_Auxiliary')(data_F)

    # first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.
    
    return Model(data,[Verify,Class])
###########################################################################################################

'''
###########################################################################################################
#        def vis_square(data, padsize=1, padval=0):

            # force the number of filters to be square
#            n = int(np.ceil(np.sqrt(data.shape[0])))
#            padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
#            data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

            # tile the filters into an image
#            data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
#            data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
#            return data


   #     img = vis_square(generated_images)
    #    if not os.path.exists(path):
   #         os.makedirs(path)
   #     Image.fromarray(img).save(
   #         'images/plot_epoch_{0:03d}_generated.png'.format(load_epoch))

   #     pickle.dump({'train': train_history, 'test': test_history},
   #                     open('acgan-history.pkl', 'wb'))     
     


###########################################################################################################
'''
    for epoch in range(nb_epochs):
        if epoch%100==0:
            print('****************************')
            print('In the '+str(epoch)+' step:')
        noise_dis=np.random.normal(0,0.5,(batch_size,latent_size))
#################################################################################################
#        sampled_labels=np.random.randint(6,26,batch_size)
        sampled_labels=np.random.randint(0,Class_num,batch_size)
#################################################################################################
        sampled_labels_dis=sampled_labels.reshape(-1,1)       
        generated_data=generator.predict([noise_dis,sampled_labels_dis],verbose=0 )

        X_Dis=[]
        Y_Dis=[]
        tempX=X.copy()
        tempY=y.copy()
        index=sorted(random.sample(range(0,len(tempX)),batch_size))
        for i in index:
            X_Dis.append(tempX[i])
            Y_Dis.append(tempY[i])
        X_Dis=np.array(X_Dis,dtype=float)
#################################################################################################
#        X_Dis=X_Dis.reshape(-1,1,4,4)
        X_Dis=X_Dis.reshape(-1,3,32,32)
################################################################################################# 
        Y_Dis=np.array(Y_Dis,dtype='uint8')
        
        X_InputDis=np.concatenate((X_Dis,generated_data)) 
#################################################################################################         
        #truth_Y=np.array([1]*batch_size+[0]*batch_size)
        y_true=np.random.uniform(0.7,1.2,size=(batch_size,))
        y_false=np.random.uniform(0.0,0.3,size=(batch_size,))
        truth_Y=np.concatenate((y_true,y_false),axis=0)
#################################################################################################                 
        Label_Y=np.concatenate((Y_Dis,sampled_labels_dis),axis=0)
        
 #       discriminator.trainable=True
        Loss_Dis=discriminator.train_on_batch(X_InputDis,[truth_Y,Label_Y])
        if epoch%100==0:
            print('The loss value of Discriminator [Total; Authenticity; Auxiliary] :' +str(Loss_Dis[0])+' '+str(Loss_Dis[1])+' '+str(Loss_Dis[2]))
        
        noise_gen=np.random.normal(0,0.5,(2*batch_size,latent_size))

################################################################################################# 
#        sampled_labels=np.random.randint(6,26,2*batch_size)
        sampled_labels=np.random.randint(0,Class_num,2*batch_size)
################################################################################################# 

        sampled_labels_gen=sampled_labels.reshape(-1,1)
################################################################################################# 
#        trick_AdverNet=np.ones(2*batch_size)
        trick_AdverNet=np.random.uniform(0.7,1.2,size=(2*batch_size,))
#################################################################################################         

 #       discriminator.trainable=False        
        Loss_Gen=Adversarial_Net.train_on_batch([noise_gen,sampled_labels_gen],[trick_AdverNet,sampled_labels_gen])
        
        if epoch%100==0:
            print('The loss value of Generator [ Total; Authenticity; Auxiliary ] :' +str(Loss_Gen[0])+' '+str(Loss_Gen[1])+' '+str(Loss_Gen[2]))
    print('/*******************************************************/')
    print('         finished!!  ')
    timer.elapsed_time()
    print('/*******************************************************/\n')  
    return generator,discriminator
'''