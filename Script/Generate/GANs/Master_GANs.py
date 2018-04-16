# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 13:21:28 2018

@author: Herman Wu
"""

import os
import math

import Freeze_Result
import Data_processing_GANs
import ACGANs
import numpy as np
import time
from keras.models import load_model

#if __name__==' __main__':
Freeze_Result.main()
Local_dir = os.path.dirname(__file__)
Base_dir=(Local_dir.split('Script'))[0]
Epc=100
latent_size=100
#batch_size=1
###########################################################################################################
Input_X,Input_Y=Data_processing_GANs.Data_process(Base_dir,13000)
print(' \n           Finished Loading the Data!!  \n ')
print('/*******************************************************/')
time.sleep(2)
#################################################################G##########################################

[Gen,Dis]=ACGANs.main(Input_X,Input_Y,nb_epochs=100,batch_size=256)
Data_processing_GANs.save(Gen,Base_dir,100,Type='Generator')
Data_processing_GANs.save(Dis,Base_dir,100,Type='Discriminator')
#Model=ResNet.main(Input_X,Input_Y,epoch=Epc,batch_size=40)
print('            Finished Training the Network!!  \n ')
print('/*******************************************************/')


#noise_dis=np.random.normal(0,0.5,(batch_size,latent_size))
#sampled_labels=np.random.randint(6,26,batch_size)
#sampled_labels_dis=np.array(sampled_labels.reshape(-1,1),dtype='uint8')       
#generated_data=Gen.predict([noise_dis,sampled_labels_dis],verbose=0 )

'''
Test=Input_X[:,:,:,:]
Prediction=(Model.predict(Test))*4.6
Real=Input_Y*4.6

ava_error=0
abs_error=0
acc_error=0
for index,i in enumerate(Real):
    temp=abs(Prediction[index]-i)/Prediction[index]
    #print(temp)
    ava_error+=temp
ava_error=ava_error/116

for index,i in enumerate(Real):
    temp=abs(Prediction[index]-i)
    if i!=0:
        temp_acc=temp/i
    acc_error+=temp_acc
    abs_error+=temp
    
abs_error=abs_error/116
acc_error=acc_error/115
print('/*******************************************************/')
print(' \n     The  relative error is:'+str(ava_error)+'   \n')
print(' \n     The  Absolute error is:'+str(abs_error)+'   \n')
print(' \n     The Accuracy error is : '+str(acc_error)+'   \n')
print('/*******************************************************/')
'''