# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 22:05:17 2018

@author: Herman Wu
"""

import os
 
import numpy as np
import Data_processing
import Network_standard
import Concat_Net
import Residual_Net
import Inception_Net
import Freeze_Result

import time 

Local_dir = os.path.dirname(__file__)
Base_dir=(Local_dir.split('Script'))[0]
Freeze_Result.main()
###########################################################################################################
Epc=200
#Type='RCN'
Type='CCN'
 #Type='VCN'
TransferLearning=False
###########################################################################################################
Input_X=[]  
Input_Y=[]
Test_X=[]
Test_Y=[]
#Input_X,Input_Y=Data_processing.Data_process(Base_dir,Four_num=13018,Five_num=79647,Six_num=6382)
#Test_X,Test_Y=Data_processing.Test_Data_process(Base_dir,1000)
Input_X,Input_Y=Data_processing.Data_loading(Base_dir)
Test_X,Test_Y=Data_processing.Test_Data_loading(Base_dir)
#new_X=Data_processing.new_Data_process(Base_dir,1000)
print('/*******************************************************/')
print(' \n           Finished Loading the Data!!  \n ')
print('  Choosing '+Type+ ' type of Network to train the dataset \n')
print('/*******************************************************/')
time.sleep(2)
###########################################################################################################
if Type=='VCN':
#    Model1,History1=Network_standard.Single_main(Base_dir,Input_X,Input_Y,epoch=Epc,batch_size=512,TF=TransferLearning,mode='5by5')   
    Model=Network_standard.main(Base_dir,Input_X,Input_Y,epoch=Epc,batch_size=512,TF=TransferLearning)
elif Type=='CCN':
#    Model2,History2=Concat_Net.Single_main(Base_dir,Input_X,Input_Y,epoch=Epc,batch_size=256,TF=TransferLearning,mode='5by5')
    Model=Concat_Net.main(Base_dir,Input_X,Input_Y,epoch=Epc,batch_size=128,TF=TransferLearning)
elif Type=='RCN':
#    Model3,History3=Residual_Net.Single_main(Base_dir,Input_X,Input_Y,epoch=Epc,batch_size=512,TF=TransferLearning,mode='5by5')          
    Model=Residual_Net.main(Base_dir,Input_X,Input_Y,epoch=Epc,batch_size=256,TF=TransferLearning)    
print('            Finished Training the Network!!  \n ')
print('/*******************************************************/')
###########################################################################################################
Test_4by4=Test_X['4by4_data']
Test_5by5=Test_X['5by5_data']
Test_6by6=Test_X['6by6_data']
tempx_1=(Model.predict(Test_4by4))*4.6    
tempx_2=(Model.predict(Test_5by5))*4.6
tempx_3=(Model.predict(Test_6by6))*4.6
temp1=Test_Y['4by4_data']*4.6
temp2=Test_Y['5by5_data']*4.6
temp3=Test_Y['6by6_data']*4.6
Prediction=np.concatenate((tempx_1,tempx_2,tempx_3),axis=0)
Real=np.concatenate((temp1,temp2,temp3),axis=0)
###########################################################################################################
rela_error=0
abs_error=0
per_error=0
temp_ava=len(Real)
for index,i in enumerate(Real):
    if Prediction[index]==0:
        temp_ava+=-1
    else:
        temp=abs(Prediction[index]-i)/Prediction[index]
        rela_error+=temp
rela_error=rela_error/temp_ava

for index,i in enumerate(Real):
    if i!=0:
        temp=abs(Prediction[index]-i)
        abs_error+=temp
    
abs_error=abs_error/3000
per_error=rela_error*100

print('/*******************************************************/')
print(' \n     The  relative error is:'+str(rela_error)+'   \n')
print(' \n     The  Absolute error is:'+str(abs_error)+'   \n')
print(' \n     The  Percent error is : '+str(per_error)+' %   \n')
print('/*******************************************************/')

