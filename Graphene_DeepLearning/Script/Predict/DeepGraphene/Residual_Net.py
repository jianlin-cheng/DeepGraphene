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
from keras.models import load_model
from keras.callbacks import Callback

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

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
 
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        
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
    
def ModelBuild(model,input_shape):
    inputs = Input(input_shape)
    X_inputs=ZeroPadding2D((1,1))(inputs)
    
    c1=Conv2D(24,(3,3),padding='valid',data_format="channels_last",kernel_initializer=glorot_uniform(seed=0))(X_inputs)
    b1=BatchNormalization(axis=3)(c1)
    r1=Activation('elu')(b1)
###########################################################################################################       
    Con1=Convolution_Block(r1,[24,24,48])
    Iden1=Identity_Block(Con1,[24,24,48])
    Iden2=Identity_Block(Iden1,[24,24,48])
    
    Con2=Convolution_Block(Iden2,[48,48,64])
    Iden3=Identity_Block(Con2,[48,48,64])
    Iden4=Identity_Block(Iden3,[48,48,64])

    Con3=Convolution_Block(Iden4,[64,64,72])
    Iden5=Identity_Block(Con3,[64,64,72])
    Iden6=Identity_Block(Iden5,[64,64,72])

    Con4=Convolution_Block(Iden6,[72,72,84])
    Iden6=Identity_Block(Con4,[72,72,84])
    Iden7=Identity_Block(Iden6,[72,72,84])
    
    f1=GlobalAveragePooling2D(name='Glob_MaxPool_Lay')(Iden7)
###########################################################################################################   
    drop1=Dropout(0.4,name='Drop_1')(f1)
 #   d1=Dense(64,activation='relu',name='DenseLay_1')(drop1)
    d2=Dense(48,activation='relu',name='DenseLay_2')(drop1)
    d3=Dense(24,activation='elu',name='DenseLay_3')(d2)
    O1=Dense(1,activation='elu',name='OutLay')(d3)
    model = Model(input=inputs, output=O1)
    return model
########################################################################################################### 
def mean_pred(y_true,y_pred):
    return K.mean(y_pred)

def configure(model,Loss='mse'):
    optimizers.rmsprop(lr=0.045)
    model.compile(loss=Loss,optimizer='rmsprop')#,metrics=[mean_pred])
    print('\n################    The Detail of the ResNet     ###################')    
    print(model.summary())
    time.sleep(5)
    print('\n######################################################################\n')

def Single_main(Base_dir,Docx,DocY,epoch=3000,batch_size=50,TF=False,mode=None):
    in_shape= (None, None, 1) 
#    Four_InputX=Docx['4by4_data']
#    Four_InputY=DocY['4by4_data']
    Five_InputX=Docx['5by5_data']
    Five_InputY=DocY['5by5_data']
#    Six_InputX=Docx['6by6_data']
#    Six_InputY=DocY['6by6_data']
 
    if TF==False:
        Network=Sequential()
        Network=ModelBuild(Network,in_shape)
        configure(Network)
    else :
        H5_file=Base_dir+'/predict_h5file/5by5_ConcatNet.h5'
        Network=load_model(H5_file)
 
    timer = ElapsedTimer()        

    history = LossHistory()
    print('/*******************************************************/\n')
    print(' Now we begin to train this model.\n')
    print('/*******************************************************/\n') 
    if mode=='4by4':
        Network.fit(Four_InputX,Four_InputY,epochs=epoch,batch_size=batch_size,validation_split=0.1,shuffle=True)
    elif mode=='5by5':
        Network.fit(Five_InputX,Five_InputY,epochs=epoch,batch_size=batch_size,validation_split=0.1,shuffle=True,callbacks=[history])
    elif mode=='6by6':
        Network.fit(Six_InputX,Six_InputY,epochs=epoch,batch_size=batch_size,validation_split=0.1,shuffle=True) 
# =============================================================================
    print('/*******************************************************/')
    print('         finished!!  ')
    timer.elapsed_time()

    print('/*******************************************************/\n')    
    return Network,history

def main(Base_dir,Docx,DocY,epoch=3000,batch_size=50,TF=False):
    in_shape= (None, None, 1) 
    History_4by4=[]
    History_5by5=[]
#    History_6by6=[]
    Iteration_num=int(len(Docx['5by5_data'])/batch_size);
    Four_InputX=Docx['4by4_data']
    Four_InputY=DocY['4by4_data']
    Five_InputX=Docx['5by5_data']
    Five_InputY=DocY['5by5_data']
#    Six_InputX=Docx['6by6_data']
#    Six_InputY=DocY['6by6_data']

########################################################################################################### 
    if TF==False:
        Network=Sequential()
        Network=ModelBuild(Network,in_shape)
        configure(Network)
    else :
        H5_file=Base_dir+'/predict_h5file/5by5_ResNet.h5'
        Network=load_model(H5_file)
########################################################################################################### 
    timer = ElapsedTimer()        
    print('/*******************************************************/\n')
    print(' Now we begin to train this model.\n')
    print('/*******************************************************/\n') 
########################################################################################################### 
    for i in range(epoch):
        for j in range(Iteration_num):
            if(j==Iteration_num-1):
                History_4by4.append(Network.train_on_batch(Four_InputX[(j+1)*batch_size:,:,:,:],Four_InputY[(j+1)*batch_size:,:]))
                History_5by5.append(Network.train_on_batch(Five_InputX[(j+1)*batch_size:,:,:,:],Five_InputY[(j+1)*batch_size:,:]))
#                History_6by6.append(Network.train_on_batch(Six_InputX[(j+1)*batch_size:,:,:,:],Six_InputY[(j+1)*batch_size:,:]))
            else:
                History_4by4.append(Network.train_on_batch(Four_InputX[j*batch_size:(j+1)*batch_size,:,:,:],Four_InputY[j*batch_size:(j+1)*batch_size,:]))
                History_5by5.append(Network.train_on_batch(Five_InputX[j*batch_size:(j+1)*batch_size,:,:,:],Five_InputY[j*batch_size:(j+1)*batch_size,:]))
#                History_6by6.append(Network.train_on_batch(Six_InputX[j*batch_size:(j+1)*batch_size,:,:,:],Six_InputY[j*batch_size:(j+1)*batch_size,:]))
        if (i%50==0):
            print('In iteration '+str(i)+', The Training detail is :  4by4: '+ str(History_4by4[i]))    
            print('In iteration '+str(i)+', The Training detail is :  5by5: '+ str(History_5by5[i])+'\n')    
#            print('In iteration '+str(i)+', The Training detail is :  6by6: '+ str(History_6by6[i])+'\n')    
########################################################################################################### 
    print('/*******************************************************/')
    print('         finished!!  ')
########################################################################################################### 
    if TF==True:
        h5_dir=Base_dir+'/predict_h5file/total_TF6by6_ResNet.h5'
    elif TF==False:
        h5_dir=Base_dir+'/predict_h5file/total_Non-TF6by6_ResNet.h5'
    Network.save(h5_dir)
    timer.elapsed_time()
    print('/*******************************************************/\n')    
    return Network
