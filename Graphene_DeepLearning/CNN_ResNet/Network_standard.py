# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 21:04:31 2018

@author: Herman Wu
"""


import numpy as np  
np.random.seed(1337)  # for reproducibility  
from keras.models import Sequential  
from keras.layers import Dense, Dropout,Flatten,BatchNormalization
from keras.layers import GlobalMaxPooling2D,GlobalAveragePooling2D
from keras import optimizers
from keras.layers import Conv2D
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
###########################################################################################################   
    Model.add(Conv2D(50,(3,3),padding='same',input_shape=input_shape,data_format="channels_last",kernel_initializer='RandomNormal',activation='elu',name="Conv1"))
    Model.add(BatchNormalization(axis=3,name='BatchNorm_1'))
    Model.add(Conv2D(48,(2,2),padding='same',activation='elu',name="Conv2"))
    Model.add(BatchNormalization(axis=3,name='BatchNorm_2'))
    Model.add(Conv2D(50,(2,2),padding='same',activation='elu',name="Conv3"))
    Model.add(BatchNormalization(axis=3,name='BatchNorm_3'))
    Model.add(Conv2D(48,(2,2),padding='same',activation='elu',name="Conv4"))
    Model.add(BatchNormalization(axis=3,name='BatchNorm_4'))
    Model.add(Conv2D(50,(2,2),padding='same',activation='elu',name="Conv5"))
    Model.add(BatchNormalization(axis=3,name='BatchNorm_5'))
    Model.add(Conv2D(48,(2,2),padding='same',activation='elu',name="Conv6"))
    Model.add(BatchNormalization(axis=3,name='BatchNorm_6'))
    Model.add(Conv2D(50,(2,2),padding='same',activation='elu',name="Conv7"))
    Model.add(BatchNormalization(axis=3,name='BatchNorm_7'))
    Model.add(Conv2D(48,(2,2),padding='same',activation='elu',name="Conv8"))
    Model.add(BatchNormalization(axis=3,name='BatchNorm_8'))
    Model.add(Conv2D(50,(2,2),padding='same',activation='elu',name="Conv9"))
    Model.add(BatchNormalization(axis=3,name='BatchNorm_9'))
    Model.add(Conv2D(50,(2,2),padding='same',activation='elu',name="Conv10"))
    Model.add(BatchNormalization(axis=3,name='BatchNorm_10'))
    Model.add(Conv2D(50,(2,2),padding='same',activation='elu',name="Conv11"))
    Model.add(BatchNormalization(axis=3,name='BatchNorm_11'))
    Model.add(Conv2D(50,(2,2),padding='same',activation='elu',name="Conv12"))
    Model.add(BatchNormalization(axis=3,name='BatchNorm_12'))
    Model.add(Conv2D(50,(2,2),padding='same',activation='elu',name="Conv13"))
    Model.add(BatchNormalization(axis=3,name='BatchNorm_13'))
    Model.add(Conv2D(50,(2,2),padding='same',activation='elu',name="Conv14"))
    Model.add(BatchNormalization(axis=3,name='BatchNorm_14'))
    Model.add(Conv2D(50,(2,2),padding='same',activation='elu',name="Conv15"))
    Model.add(BatchNormalization(axis=3,name='BatchNorm_15'))
    Model.add(GlobalMaxPooling2D(data_format='channels_last',name='Glob_MaxPool_Lay'))
#    Model.add(GlobalAveragePooling2D(data_format='channels_last',name='Glob_AvaPool_Lay'))
    #Model.add(MaxPooling2D(pool_size=(2,2),name="Pool1"))
###########################################################################################################   
    Model.add(Dropout(0.4,name='Drop_1'))
    Model.add(Dense(64,activation='elu',name='DenseLay_1'))
    Model.add(Dense(64,activation='elu',name='DenseLay_2'))
    Model.add(Dense(48,activation='elu',name='DenseLay_3'))
#    Model.add(Dense(12,activation='relu',name='DenseLay_4'))
    Model.add(Dropout(0.3,name='Drop_2'))
    Model.add(Dense(1,activation='elu',name='OutLay'))  #Don't change
    #Model.add(MaxPooling2D(pool_size=(2,2),name="Pool2"))
########################################################################################################### 
#def mean_pred(y_true,y_pred):
#    return K.mean(y_pred)

def configure(Model,Loss='mse'):
    optimizers.rmsprop(lr=0.05,decay=5e-7)
    #optimizers.Adam(lr=0.05,epsilon=10e-8,beta_1=0.9,beta_2=0.99)
    Model.compile(loss=Loss,optimizer='rmsprop')#,metrics=[mean_pred])
#    optimizers.Adadelta(lr=0.015,decay=5e-7)
#    Model.compile(loss=Loss,optimizer='Adadelta',metrics=[mean_pred])
    print('\n################    The Detail of the CNN_Standard     ###################')    
    print(Model.summary())
    time.sleep(2)
    print('\n######################################################################\n')

def main(Docx,DocY,epoch=3000,batch_size=50):
    in_shape= (None, None, 1) 
    History_4by4=[]
    History_5by5=[]
    Iteration_num=int(len(Docx['4by4_data'])/batch_size);
    Five_InputX=Docx['5by5_data']
    Five_InputY=DocY['5by5_data']
    Four_InputX=Docx['4by4_data']
    Four_InputY=DocY['4by4_data']
########################################################################################################### 
    Network=Sequential()
    ModelBuild(Network,in_shape)
    configure(Network)
########################################################################################################### 
    timer = ElapsedTimer()        
    print('/*******************************************************/\n')
    print(' Now we begin to train this model.\n')
    print('/*******************************************************/\n') 
#    History_4=Network.fit(Four_InputX,Four_InputY,batch_size=batch_size,epochs=epoch,shuffle=True,validation_split=0.10) 
#    lib.viz_model(History_4,'');
#    History_5=Network.fit(Five_InputX,Five_InputY,batch_size=batch_size,epochs=epoch,shuffle=True,validation_split=0.10) 
#    lib.viz_model(History_5,'');    
########################################################################################################### 
    for i in range(epoch):
        for j in range(Iteration_num):
            if(j==Iteration_num-1):
                History_4by4.append(Network.train_on_batch(Four_InputX[(j+1)*batch_size:,:,:,:],Four_InputY[(j+1)*batch_size:,:]))
                History_5by5.append(Network.train_on_batch(Five_InputX[(j+1)*batch_size:,:,:,:],Five_InputY[(j+1)*batch_size:,:]))
            else:
                History_4by4.append(Network.train_on_batch(Four_InputX[j*batch_size:(j+1)*batch_size,:,:,:],Four_InputY[j*batch_size:(j+1)*batch_size,:]))
                History_5by5.append(Network.train_on_batch(Five_InputX[j*batch_size:(j+1)*batch_size,:,:,:],Five_InputY[j*batch_size:(j+1)*batch_size,:]))
        if (i%50==0):
            print('In iteration '+str(i)+', The Training detail is :  4by4: '+ str(History_4by4[i]) )#+' ; 5by5: ' + str(History_5by5[i])+'\n')    
            print('In iteration '+str(i)+', The Training detail is :  5by5: '+ str(History_5by5[i])+'\n')    
########################################################################################################### 
    print('/*******************************************************/')
    print('         finished!!  ')
    timer.elapsed_time()
    print('/*******************************************************/\n')    
    return Network