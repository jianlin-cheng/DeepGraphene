# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 23:34:42 2018

@author: Herman Wu
"""

import numpy as np  
np.random.seed(1337)  # for reproducibility  
from keras.models import Sequential,Model  
from keras.layers import Dense, Dropout, Activation, Flatten,merge,Input  
from keras.layers import  MaxPooling2D,UpSampling2D,GlobalMaxPooling2D  
from keras import optimizers
from keras.utils import np_utils   
from keras.layers import Conv2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
import pandas as pd
import time
from keras.models import load_model
import keras.backend as K 
from keras.layers import concatenate


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

def Concat_block(inputs):
    x1=Conv2D(60,(3,3),padding='same',data_format="channels_last")(inputs)
    b1=BatchNormalization(axis=3)(x1)
    r1=Activation('elu')(b1) 
    x2=Conv2D(32,(1,1),padding='same')(r1)
    r2=Activation('elu')(x2)  
    x3=Conv2D(60,(3,3),padding='same')(r2)   
    b3=BatchNormalization(axis=3)(x3)
#    m1=merge([b3,inputs],mode='concat',concat_axis=3) 
    m1 = concatenate([b3,inputs],axis=3)
    r3=Activation('elu')(m1)
    return r3 
    
def ModelBuild(model,input_shape):
    inputs = Input(input_shape)
    c1=Conv2D(60,(3,3),padding='same',input_shape=input_shape,data_format="channels_last",name="Conv1")(inputs)
    b1=BatchNormalization(axis=3)(c1)
    r1=Activation('elu')(b1)
    c2=Conv2D(60,(3,3),padding='same',name="Conv2")(r1)
    b2=BatchNormalization(axis=3)(c2)      
    r2=Activation('elu')(b2)
    #m1=merge([r2,inputs],mode='concat',concat_axis=3)
    m1 = concatenate([r2,inputs],axis=3)
###########################################################################################################        
    for i in range(15):
        m1=Concat_block(m1)
    f1=GlobalMaxPooling2D(name='Glob_MaxPool_Lay')(m1)
    drop1=Dropout(0.4,name='Drop_1')(f1)
    d1=Dense(64,activation='elu',name='DenseLay_1')(drop1)
    d2=Dense(48,activation='elu',name='DenseLay_2')(d1)
    d3=Dense(24,activation='elu',name='DenseLay_3')(d2)
    drop2=Dropout(0.2,name='Drop_2')(d3)
    O1=Dense(1,activation='elu',name='OutLay')(drop2)
    model = Model(input=inputs, output=O1)
    return model
########################################################################################################### 
def configure(model,Loss='mse'):
    optimizers.rmsprop(lr=0.055)
#    optimizers.Adam(lr=0.055,epsilon=10e-8,beta_1=0.9,beta_2=0.99)
    model.compile(loss=Loss,optimizer='rmsprop')
#    model.compile(loss=Loss,optimizer='Adam')
    print('\n################    The Detail of the ConcatNet     ###################')    
    print(model.summary())
    time.sleep(5)
    print('\n######################################################################\n')

###########################################################################################################   
def GAN_main(Base_dir,Docx,DocY,epoch=3000,batch_size=50,TF=False,mode=None):
    in_shape= (None, None, 1) 
    Four_InputX=Docx['4by4_data']
    Four_InputX=Docx['4by4_data']
    Four_InputY=DocY['4by4_data']
    Five_InputX=Docx['5by5_data']
    Five_InputY=DocY['5by5_data']
    Six_InputX=Docx['6by6_data']
    Six_InputY=DocY['6by6_data']
########################################################################################################### 
    if TF==False:
        Network=Sequential()
        Network=ModelBuild(Network,in_shape)
        configure(Network)
    else :
        H5_file=Base_dir+'/predict_h5file/5by5_ConcatNet.h5'
        Network=load_model(H5_file)
########################################################################################################### 
    timer = ElapsedTimer()        
    print('/*******************************************************/\n')
    print(' Now we begin to train this model.\n')
    print('/*******************************************************/\n') 
    if mode=='4by4':
        Network.fit(Four_InputX,Four_InputY,epochs=epoch,batch_size=batch_size,validation_split=0.1,shuffle=True)
    elif mode=='5by5':
        Network.fit(Five_InputX,Five_InputY,epochs=epoch,batch_size=batch_size,validation_split=0.1,shuffle=True)
    elif mode=='6by6':
        Network.fit(Six_InputX,Six_InputY,epochs=epoch,batch_size=batch_size,validation_split=0.1,shuffle=True) 
# =============================================================================
    print('/*******************************************************/')
    print('         finished!!  ')
    timer.elapsed_time()
########################################################################################################### 
    #if TF==True:
    #    h5_dir=Base_dir+'/predict_h5file/total_TF6by6_ConcatNet.h5'
    #elif TF==False:
    #    h5_dir=Base_dir+'/predict_h5file/total_Non-TF6by6_ConcatNet.h5'
    #Network.save(h5_dir)
    print('/*******************************************************/\n')    
    return Network
########################################################################################################### 
    
def main(Base_dir,Docx,DocY,epoch=3000,batch_size=50,TF=False):
    in_shape= (None, None, 1) 
    History_4by4=[]
    History_5by5=[]
    History_6by6=[]
#    Iteration_num=int(len(Docx['6by6_data'])/batch_size);
    temp_len=max(len(Docx['4by4_data']),len(Docx['5by5_data']),len(Docx['6by6_data']))
    Iteration_num=int(temp_len//batch_size);
    Four_num=int(len(Docx['4by4_data'])//batch_size);
    Five_num=int(len(Docx['5by5_data'])//batch_size);
    Six_num=int(len(Docx['6by6_data'])//batch_size);                 
    Four_InputX=Docx['4by4_data']
    Four_InputX=Docx['4by4_data']
    Four_InputY=DocY['4by4_data']
    Five_InputX=Docx['5by5_data']
    Five_InputY=DocY['5by5_data']
    Six_InputX=Docx['6by6_data']
    Six_InputY=DocY['6by6_data']
########################################################################################################### 
    if TF==False:
        Network=Sequential()
        Network=ModelBuild(Network,in_shape)
        configure(Network)
    else :
        H5_file=Base_dir+'/predict_h5file/5by5_ConcatNet.h5'
        Network=load_model(H5_file)
########################################################################################################### 
    timer = ElapsedTimer()        
    print('/*******************************************************/\n')
    print(' Now we begin to train this model.\n')
    print('/*******************************************************/\n') 
########################################################################################################### 
    for i in range(epoch):
        for j in range(Iteration_num):
            if (j+1)*batch_size>len(Docx['4by4_data']):
                j0=j%Four_num
                History_4by4.append(Network.train_on_batch(Four_InputX[j0*batch_size:(j0+1)*batch_size,:,:,:],Four_InputY[j0*batch_size:(j0+1)*batch_size,:]))
            else:
                History_4by4.append(Network.train_on_batch(Four_InputX[j*batch_size:(j+1)*batch_size,:,:,:],Four_InputY[j*batch_size:(j+1)*batch_size,:]))
            if (j+1)*batch_size>len(Docx['5by5_data']):
                j0=j%Five_num
                History_5by5.append(Network.train_on_batch(Five_InputX[j0*batch_size:(j0+1)*batch_size,:,:,:],Five_InputY[j0*batch_size:(j0+1)*batch_size,:]))
            else:
                History_5by5.append(Network.train_on_batch(Five_InputX[j*batch_size:(j+1)*batch_size,:,:,:],Five_InputY[j*batch_size:(j+1)*batch_size,:]))                
            if (j+1)*batch_size>len(Docx['6by6_data']):
                j0=j%Six_num
                History_6by6.append(Network.train_on_batch(Six_InputX[j0*batch_size:(j0+1)*batch_size,:,:,:],Six_InputY[j0*batch_size:(j0+1)*batch_size,:]))
            else:
                History_6by6.append(Network.train_on_batch(Six_InputX[j*batch_size:(j+1)*batch_size,:,:,:],Six_InputY[j*batch_size:(j+1)*batch_size,:]))                
        if (i%50==0):
            print('In iteration '+str(i)+', The Training detail is :  4by4: '+ str(History_4by4[i]))    
            print('In iteration '+str(i)+', The Training detail is :  5by5: '+ str(History_5by5[i])+'\n')    
            print('In iteration '+str(i)+', The Training detail is :  6by6: '+ str(History_6by6[i])+'\n') 

# =============================================================================
#     for i in range(epoch):
#         for j in range(Iteration_num):
#             if(j==Iteration_num-1):
#                 History_4by4.append(Network.train_on_batch(Four_InputX[(j+1)*batch_size:,:,:,:],Four_InputY[(j+1)*batch_size:,:]))
#                 History_5by5.append(Network.train_on_batch(Five_InputX[(j+1)*batch_size:,:,:,:],Five_InputY[(j+1)*batch_size:,:]))
# #                History_6by6.append(Network.train_on_batch(Six_InputX[(j+1)*batch_size:,:,:,:],Six_InputY[(j+1)*batch_size:,:]))
#             else:
#                 History_4by4.append(Network.train_on_batch(Four_InputX[j*batch_size:(j+1)*batch_size,:,:,:],Four_InputY[j*batch_size:(j+1)*batch_size,:]))
#                 History_5by5.append(Network.train_on_batch(Five_InputX[j*batch_size:(j+1)*batch_size,:,:,:],Five_InputY[j*batch_size:(j+1)*batch_size,:]))
# #                History_6by6.append(Network.train_on_batch(Six_InputX[j*batch_size:(j+1)*batch_size,:,:,:],Six_InputY[j*batch_size:(j+1)*batch_size,:]))
#         if (i%10==0):
#             print('In iteration '+str(i)+', The Training detail is :  4by4: '+ str(History_4by4[i]))    
#             print('In iteration '+str(i)+', The Training detail is :  5by5: '+ str(History_5by5[i])+'\n')    
# #            print('In iteration '+str(i)+', The Training detail is :  6by6: '+ str(History_6by6[i])+'\n')    
# ########################################################################################################### 
# =============================================================================
    print('/*******************************************************/')
    print('         finished!!  ')
    timer.elapsed_time()
########################################################################################################### 
    #if TF==True:
    #    h5_dir=Base_dir+'/predict_h5file/total_TF6by6_ConcatNet.h5'
    #elif TF==False:
    #    h5_dir=Base_dir+'/predict_h5file/total_Non-TF6by6_ConcatNet.h5'
    Network.save(h5_dir)
    print('/*******************************************************/\n')    
    return Network