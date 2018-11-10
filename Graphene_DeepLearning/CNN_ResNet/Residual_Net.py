# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 10:46:38 2018

@author: Herman Wu
"""



import numpy as np  
np.random.seed(1337)  # for reproducibility  
from keras.models import Sequential,Model  
from keras.layers import Add,Dense,Dropout,Activation,Input  
from keras.layers import GlobalMaxPooling2D,GlobalAveragePooling2D,ZeroPadding2D  
from keras import optimizers   
from keras.layers import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.initializers import glorot_uniform
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

def Identity_Block(inputs,filters):
    Filter1,Filter2,Filter3=filters
###########################################################################################################   
    x1=Conv2D(Filter1,(1,1),padding='same',kernel_initializer=glorot_uniform(seed=0))(inputs)
    z1=BatchNormalization(axis=3)(x1)
    a1=Activation('elu')(z1)
    
    x2=Conv2D(Filter2,(3,3),padding='same',kernel_initializer=glorot_uniform(seed=0))(a1)
    z2=BatchNormalization(axis=3)(x2)
    a2=Activation('elu')(z2)
    
    x3=Conv2D(Filter3,(1,1),padding='same',kernel_initializer=glorot_uniform(seed=0))(a2)
    z3=BatchNormalization(axis=3)(x3)
    
    z_sum=Add()([z3,inputs])
    a3=Activation('elu')(z_sum)
###########################################################################################################   
    return a3

def Convolution_Block(inputs,filters):
    Filter1,Filter2,Filter3=filters
###########################################################################################################   
    x1=Conv2D(Filter1,(1,1),padding='same',kernel_initializer=glorot_uniform(seed=0))(inputs)    
    z1=BatchNormalization(axis=3)(x1)
    a1=Activation('elu')(z1)
    
    x2=Conv2D(Filter2,(3,3),padding='same',kernel_initializer=glorot_uniform(seed=0))(a1)
    z2=BatchNormalization(axis=3)(x2)
    a2=Activation('elu')(z2)
    
    x3=Conv2D(Filter3,(1,1),padding='same',kernel_initializer=glorot_uniform(seed=0))(a2)
    z3=BatchNormalization(axis=3)(x3)
    
    x_shortcut=Conv2D(Filter3,(1,1),padding='same',kernel_initializer=glorot_uniform(seed=0))(inputs)
    z_shortcut=BatchNormalization(axis=3)(x_shortcut)
    
    z_sum=Add()([z3,z_shortcut])
    a3=Activation('elu')(z_sum)
###########################################################################################################   
    return a3    
    
def ModelBuild(model,input_shape,Num_Lay):
    inputs = Input(input_shape)
    X_inputs=ZeroPadding2D((1,1))(inputs)
    
    c1=Conv2D(24,(3,3),padding='valid',data_format="channels_last",kernel_initializer=glorot_uniform(seed=0))(X_inputs)
    b1=BatchNormalization(axis=3)(c1)
    r1=Activation('elu')(b1)
###########################################################################################################       
    for i in range(Num_Lay):
        temp=i+1
        r1=Convolution_Block(r1,[24*temp,24*temp,24*(temp+1)])
        r1=Identity_Block(r1,[24*temp,24*temp,24*(temp+1)])
        r1=Identity_Block(r1,[24*temp,24*temp,24*(temp+1)])
    
    f1=GlobalAveragePooling2D(name='Glob_MaxPool_Lay')(r1)
###########################################################################################################   
    drop1=Dropout(0.4,name='Drop_1')(f1)
    d1=Dense(64,activation='relu',name='DenseLay_1')(drop1)
    d2=Dense(48,activation='relu',name='DenseLay_2')(d1)
    d3=Dense(24,activation='elu',name='DenseLay_3')(d2)
    drop2=Dropout(0.2,name='Drop_2')(d3)
    O1=Dense(1,activation='elu',name='OutLay')(d3)
    model = Model(input=inputs, output=O1)
    return model
########################################################################################################### 
def mean_pred(y_true,y_pred):
    return K.mean(y_pred)

def configure(model,Loss='mse',Learning_rate=0.055):
    optimizers.rmsprop(lr=Learning_rate)
#    optimizers.Adam(lr=0.055,epsilon=10e-8,beta_1=0.9,beta_2=0.99)
    model.compile(loss=Loss,optimizer='rmsprop')#,metrics=[mean_pred])
#    model.compile(loss=Loss,optimizer='Adam')#,metrics=[mean_pred])
    print('\n################    The Detail of the ResNet     ###################')    
    print(model.summary())
    time.sleep(5)
    print('\n######################################################################\n')

def main(Docx,DocY,epoch=3000,Learning_rate,Num_Lay=Number,batch_size=128):
#    History_4by4=[]
#    History_5by5=[]
    in_shape= (None, None, 1)
    Iteration_num=int(len(Docx['4by4_data'])/batch_size);
    Five_InputX=Docx['5by5_data']
    Five_InputY=DocY['5by5_data']
    Four_InputX=Docx['4by4_data']
    Four_InputY=DocY['4by4_data']
########################################################################################################### 
    Network=Sequential()
    Network=ModelBuild(Network,in_shape,Number)
    configure(Network,Learning_rate=Learning_rate)
########################################################################################################### 
    timer = ElapsedTimer()        
    print('/*******************************************************/\n')
    print(' Now we begin to train this model.\n')
    print('/*******************************************************/\n') 
#    History_4=Network.fit(Four_InputX,Four_InputY,batch_size=batch_size,epochs=epoch,shuffle=True,validation_split=0.10) 
#    lib.viz_model(History_4,'');
    History_5=Network.fit(Five_InputX,Five_InputY,batch_size=batch_size,epochs=epoch,shuffle=True,validation_split=0.10) 
    lib.viz_model(History_5,'');    
########################################################################################################### 
#    for i in range(epoch):
#        for j in range(Iteration_num):
#            if(j==Iteration_num-1):
#                History_4by4.append(Network.train_on_batch(Four_InputX[(j+1)*batch_size:,:,:,:],Four_InputY[(j+1)*batch_size:,:]))
#                History_5by5.append(Network.train_on_batch(Five_InputX[(j+1)*batch_size:,:,:,:],Five_InputY[(j+1)*batch_size:,:]))
#            else:
#                History_4by4.append(Network.train_on_batch(Four_InputX[j*batch_size:(j+1)*batch_size,:,:,:],Four_InputY[j*batch_size:(j+1)*batch_size,:]))
#                History_5by5.append(Network.train_on_batch(Five_InputX[j*batch_size:(j+1)*batch_size,:,:,:],Five_InputY[j*batch_size:(j+1)*batch_size,:]))
#        if (i%50==0):
#            print('In iteration '+str(i)+', The Training detail is :  4by4: '+ str(History_4by4[i]))# +' ; 5by5: ' + str(History_5by5[i]))
#            print('In iteration '+str(i)+', The Training detail is :  5by5: '+ str(History_5by5[i])+'\n')              
########################################################################################################### 
    print('/*******************************************************/')
    print('         finished!!  ')
    timer.elapsed_time()
    print('/*******************************************************/\n')    
    return Network