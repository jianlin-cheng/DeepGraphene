# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 10:10:06 2018

@author: Herman Wu
"""

import numpy as np  
np.random.seed(1337)  # for reproducibility  
from keras.models import Sequential  
from keras.layers import Dense, Dropout,Flatten  
from keras import optimizers
from keras.layers import LSTM
import time
import keras.backend as K

import lib

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

def ModelBuild(Model,input_shape):   
    Model.add(LSTM(50,(2,2),padding='same',input_shape=input_shape,data_format="channels_last",kernel_initializer='RandomNormal',activation='relu',name="LSTMConv1"))
    Model.add(LSTM(24,(2,2),padding='same',activation='relu',name="ConvLSTM2"))
    Model.add(LSTM(36,(2,2),padding='same',activation='relu',name="ConvLSTM3"))
    Model.add(Flatten(name='FlatLay'))
    #Model.add(MaxPooling2D(pool_size=(2,2),name="Pool1"))
###########################################################################################################   
    Model.add(Dropout(0.4,name='Drop_1'))
    Model.add(Dense(64,activation='relu',name='DenseLay_1'))
    Model.add(Dense(48,activation='relu',name='DenseLay_2'))
    Model.add(Dropout(0.3,name='Drop_2'))
    Model.add(Dense(1,activation='selu',name='OutLay'))  #Don't change
    #Model.add(MaxPooling2D(pool_size=(2,2),name="Pool2"))
########################################################################################################### 
def mean_pred(y_true,y_pred):
    return K.mean(y_pred)

def configure(Model,Loss='mse'):
    optimizers.rmsprop(lr=0.015,decay=5e-7)
    Model.compile(loss=Loss,optimizer='rmsprop',metrics=[mean_pred])
    print('\n################    The Detail of the CNN_Standard     ###################')    
    print(Model.summary())
    time.sleep(2)
    print('\n######################################################################\n')

def main(Docx,DocY,epoch=3000,batch_size=50):
    img_rows=Docx.shape[1] 
    img_cols=Docx.shape[2] 
    in_shape= (img_rows, img_cols, 1)  
########################################################################################################### 
    Network=Sequential()
    ModelBuild(Network,in_shape)
    configure(Network)
########################################################################################################### 
    timer = ElapsedTimer()        
    print('/*******************************************************/\n')
    print(' Now we begin to train this model.\n')
    print('/*******************************************************/\n') 
    History=Network.fit(Docx,DocY,batch_size=batch_size,epochs=epoch,shuffle=True,validation_split=0.10) 
    lib.viz_model(History,'');
########################################################################################################### 
    print('/*******************************************************/')
    print('         finished!!  ')
    timer.elapsed_time()
    print('/*******************************************************/\n')    
    return Network