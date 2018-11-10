# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 22:05:17 2018

@author: Herman Wu
"""

import os
import sys

import numpy as np
import Data_processing
import Network_standard
import Concat_Net
import Residual_Net
import Inception_Net
import Freeze_Result

import time 

#if __name__==' __main__':

Local_dir = os.path.dirname(__file__)
Base_dir=(Local_dir.split('Script'))[0]
Freeze_Result.main()
###########################################################################################################
Type=sys.argv[1]
Learning_rate=int(sys.argv[2])
Number=int(sys.argv[3])
Epc=150
#Type='ResNet'
#Type='ConcatNet'
#Type='CNN'
#Type='Inception'
###########################################################################################################
Input_X=[]  
Input_Y=[]
Test_X=[]
Test_Y=[]
Input_X,Input_Y=Data_processing.Data_process(Base_dir,Four_num=13018,Five_num=18027)
Test_X,Test_Y=Data_processing.Test_Data_process(Base_dir,1000)

print('/*******************************************************/',flush=True)
print(' \n           Finished Loading the Data!!  \n ',flush=True)
print('  Choosing '+Type+ ' type of Network to train the dataset \n',flush=True)
print('/*******************************************************/',flush=True)
time.sleep(2)
###########################################################################################################
if Type=='CNN':
    Model=Network_standard.main(Input_X,Input_Y,epoch=Epc,Learning_rate,Num_Lay=Number,batch_size=512)
elif Type=='ConcatNet':
    Model=Concat_Net.main(Input_X,Input_Y,epoch=Epc,Learning_rate,Num_Lay=Number,batch_size=512)
elif Type=='Inception':
    Model=Inception_Net.main(Input_X,Input_Y,epoch=Epc,Learning_rate,Num_Lay=Number,batch_size=256)
elif Type=='ResNet':
    Model=Residual_Net.main(Input_X,Input_Y,epoch=Epc,Learning_rate,Num_Lay=Number,batch_size=512)      
print('            Finished Training the Network!!  \n ',flush=True)
print('/*******************************************************/',flush=True)
###########################################################################################################
#Test_X,Test_Y=Data_processing.Test_Data_process(Base_dir,266)
Test_4by4=Test_X['4by4_data']
Test_5by5=Test_X['5by5_data']
temp1=(Model.predict(Test_4by4))*4.6    
temp2=(Model.predict(Test_5by5))*4.6
temp3=Test_Y['4by4_data']*4.6
temp4=Test_Y['5by5_data']*4.6
Prediction=np.concatenate((temp1,temp2),axis=0)
#Prediction=temp1
#Prediction=temp2
#Real=temp3
#Real=temp4
Real=np.concatenate((temp3,temp4),axis=0)
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
    
abs_error=abs_error/2000
per_error=rela_error*100
print('/*******************************************************/')
print(' \n     The  relative error is:'+str(rela_error)+'   \n',flush=True)
print(' \n     The  Absolute error is:'+str(abs_error)+'   \n',flush=True)
print(' \n     The  Percent error is : '+str(per_error)+' %   \n',flush=True)
print('/*******************************************************/')

