# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 10:10:06 2018

@author: Herman Wu
"""

import numpy as np  
np.random.seed(1337)  # for reproducibility  
from keras.models import Sequential,Model  
from keras.layers import Dense, Dropout, Activation, Flatten,merge,Input  
from keras.layers import AveragePooling2D,MaxPooling2D,GlobalMaxPooling2D  
from keras import optimizers  
from keras.layers import Conv2D,BatchNormalization
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

def Inception_block(inputs):
    x1=Conv2D(48,(1,1),padding='same',data_format="channels_last")(inputs)
    b1=BatchNormalization(axis=3)(x1)
    r1=Activation('elu')(b1)
    
    x2=Conv2D(48,(1,1),padding='same',data_format="channels_last")(inputs)
    b2=BatchNormalization(axis=3)(x2)
    r21=Activation('relu')(b2)
    x21=Conv2D(64,(2,2),padding='same')(r21)
    b21=BatchNormalization(axis=3)(x21)
    r2=Activation('elu')(b21)
    
    x3=Conv2D(16,(1,1),padding='same',data_format="channels_last")(inputs)
    b3=BatchNormalization(axis=3)(x3)
    r31=Activation('elu')(b3)
    x31=Conv2D(32,(3,3),padding='same')(r31)
    b31=BatchNormalization(axis=3)(x31)
    r3=Activation('elu')(b31)

 #   x4=MaxPooling2D(64,(3,3),padding='same',data_format="channels_last")(inputs)
 #   b4=BatchNormalization(axis=3)(x4)
 #   r41=Activation('relu')(b4)
 #   x41=Conv2D(24,(1,1),padding='same')(r41)
 #   b41=BatchNormalization(axis=3)(x41)
 #   r4=Activation('relu')(b41)

    m1=merge([r1,r2],mode='concat',concat_axis=3) 
    m1=merge([m1,r3],mode='concat',concat_axis=3)         
  #  m1=merge([m1,r4],mode='concat',concat_axis=3) 

    return m1 
    
def ModelBuild(model,input_shape):
    inputs = Input(input_shape)
    c1=Conv2D(32,(3,3),padding='same',input_shape=input_shape,data_format="channels_last",name="Conv1")(inputs)
    r1=Activation('elu')(c1)
#    c2=MaxPooling2D(32,(2,2),padding='same',name="MaxPool_1")(r1)
    b2=BatchNormalization(axis=3)(r1) 
#    r2=Activation('relu')(b2)    
#    c3=Conv2D(24,(1,1),padding='same',name="Conv2")(r2)  
#    r3=Activation('relu')(c3)
    c4=Conv2D(32,(2,2),padding='same',name="Conv3")(b2)
    b4=BatchNormalization(axis=3)(c4)     
    r4=Activation('elu')(b4)
#    c5=MaxPooling2D(32,(2,2),padding='same',name="MaxPool_2")(r4)   
###########################################################################################################        
    for i in range(9):
        if i==0:
            m1=Inception_block(r4)
        else:
            m1=Inception_block(m1)
    f1=GlobalMaxPooling2D(name='Glob_MaxPool_Lay')(m1)
###########################################################################################################   
    drop1=Dropout(0.7,name='Drop_1')(f1)
    d1=Dense(64,activation='elu',name='DenseLay_1')(drop1)
    d2=Dense(48,activation='relu',name='DenseLay_2')(d1)
#    d3=Dense(24,activation='relu',name='DenseLay_3')(d2)
    drop2=Dropout(0.2,name='Drop_2')(d2)
    O1=Dense(1,activation='relu',name='OutLay')(drop2)
    model = Model(input=inputs, output=O1)
    return model
########################################################################################################### 
def mean_pred(y_true,y_pred):
    return K.mean(y_pred)

def configure(model,Loss='mse'):
    optimizers.rmsprop(lr=0.055)
#    optimizers.Adam(lr=0.055,epsilon=10e-8,beta_1=0.9,beta_2=0.99)
    model.compile(loss=Loss,optimizer='rmsprop')#,metrics=[mean_pred])
#    model.compile(loss=Loss,optimizer='Adam',metrics=[mean_pred])
    print('\n################    The Detail of the Inception_Net     ###################')    
    print(model.summary())
    time.sleep(5)
    print('\n######################################################################\n')

def main(Docx,DocY,epoch=3000,batch_size=128):
    History_4by4=[]
    History_5by5=[]
    in_shape= (None, None, 1)
    Iteration_num=int(len(Docx['4by4_data'])/batch_size);
    Five_InputX=Docx['5by5_data']
    Five_InputY=DocY['5by5_data']
    Four_InputX=Docx['4by4_data']
    Four_InputY=DocY['5by5_data']
########################################################################################################### 
    Network=Sequential()
    Network=ModelBuild(Network,in_shape)
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
#                History_4by4.append(Network.train_on_batch(Four_InputX[(j+1)*batch_size:,:,:,:],Four_InputY[(j+1)*batch_size:,:]))
                History_5by5.append(Network.train_on_batch(Five_InputX[(j+1)*batch_size:,:,:,:],Five_InputY[(j+1)*batch_size:,:]))
            else:
#                History_4by4.append(Network.train_on_batch(Four_InputX[j*batch_size:(j+1)*batch_size,:,:,:],Four_InputY[j*batch_size:(j+1)*batch_size,:]))
                History_5by5.append(Network.train_on_batch(Five_InputX[j*batch_size:(j+1)*batch_size,:,:,:],Five_InputY[j*batch_size:(j+1)*batch_size,:]))
        if (i%50==0):
#            print('In iteration '+str(i)+', The Training detail is :  4by4: '+ str(History_4by4[i]) +' ; 5by5: ' + str(History_5by5[i]))
            print('In iteration '+str(i)+', The Training detail is :  5by5: '+ str(History_5by5[i])+'\n')              
########################################################################################################### 
    print('/*******************************************************/')
    print('         finished!!  ')
    timer.elapsed_time()
    print('/*******************************************************/\n')    
    return Network