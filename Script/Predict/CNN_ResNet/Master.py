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
from keras.models import load_model

#if __name__==' __main__':
Local_dir = os.path.dirname(__file__)
Base_dir=(Local_dir.split('Script'))[0]
Freeze_Result.main()
###########################################################################################################
Epc=200
Type='ResNet'
#Type='ConcatNet'
#Type='CNN'
#Type='Inception'
###########################################################################################################
Input_X=[]  
Input_Y=[]
Test_X=[]
Test_Y=[]
Input_X,Input_Y=Data_processing.Data_process(Base_dir,13018)
Test_X,Test_Y=Data_processing.Test_Data_process(Base_dir,300)

print('/*******************************************************/')
print(' \n           Finished Loading the Data!!  \n ')
print('  Choosing '+Type+ ' type of Network to train the dataset \n')
print('/*******************************************************/')
time.sleep(2)
###########################################################################################################
if Type=='CNN':
    Model=Network_standard.main(Input_X,Input_Y,epoch=Epc,batch_size=512)
elif Type=='ConcatNet':
    Model=Concat_Net.main(Input_X,Input_Y,epoch=Epc,batch_size=512)
elif Type=='Inception':
    Model=Inception_Net.main(Input_X,Input_Y,epoch=Epc,batch_size=512)
elif Type=='ResNet':
    Model=Residual_Net.main(Input_X,Input_Y,epoch=Epc,batch_size=512)
print('            Finished Training the Network!!  \n ')
print('/*******************************************************/')
###########################################################################################################
#Test_X,Test_Y=Data_processing.Test_Data_process(Base_dir,266)
#Test_4by4=Test_X['4by4_data']
Test_5by5=Test_X['5by5_data']
#temp1=(Model.predict(Test_4by4))*4.6
temp2=(Model.predict(Test_5by5))*4.6
#temp3=Test_Y['4by4_data']*4.6
temp4=Test_Y['5by5_data']*4.6
#Prediction=np.concatenate((temp1,temp2),axis=0)
Prediction=temp2
#Real=np.concatenate((temp3,temp4),axis=0)
Real=temp4
###########################################################################################################
ava_error=0
abs_error=0
acc_error=0
temp_ava=300
for index,i in enumerate(Real):
    temp=abs(Prediction[index]-i)/Prediction[index]
    if Prediction[index]==0:
        temp_ava+=-1
    else:
        ava_error+=temp
ava_error=ava_error/temp_ava

for index,i in enumerate(Real):
    temp=abs(Prediction[index]-i)
    if i!=0:
        temp_acc=temp/i
        acc_error+=temp_acc
    abs_error+=temp
    
abs_error=abs_error/300
acc_error=acc_error/300*100
print('/*******************************************************/')
print(' \n     The  relative error is:'+str(ava_error)+'   \n')
print(' \n     The  Absolute error is:'+str(abs_error)+'   \n')
print(' \n     The Accuracy error is : '+str(acc_error)+' %   \n')
print('/*******************************************************/')
#Data_processing.restore_save(Real,Prediction,Random_index,Type,'D:/Working_Application/DropBox_File/Dropbox/Graphene_DeepLearning/Test_dataset/Original_Data')

