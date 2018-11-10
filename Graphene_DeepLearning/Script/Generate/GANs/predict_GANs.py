# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 23:31:02 2018

@author: Herman Wu
"""
from keras.models import load_model
import os
import numpy as np

if __name__=='__main__':
    Base_dir = os.path.dirname(__file__)
    Base_dir=(Base_dir.split('Script'))[0]
    Label=1.0
    latent_size=100
    Data_num=1
###########################################################################################################    
    label=Label*10-6
    h5_dir=Base_dir+'GANs_result/H5_file/Gen_100.h5'
    Gen=load_model(h5_dir)
    noise_Gen=np.random.randint(-1,1,(Data_num,latent_size))    
    labels=np.ones(Data_num)*label
    labels=labels.reshape(Data_num,1)
    Generate_data=Gen.predict([noise_Gen,labels])
    Generate_data=Generate_data.reshape(Data_num,4,4)
    for c in range(len(Generate_data)):
        for i in range(4):
            for j in range(4):
                if Generate_data[c,i,j]>0.5:
                    Generate_data[c,i,j]=1
                else:
                    Generate_data[c,i,j]=0
        temp=Generate_data[c,:,:]
        csv_file=Base_dir+'GANs_result/Predict_result/Label_'+str(Label)+'/'+str(c+1)+'.csv'
        np.savetxt(csv_file,temp,fmt='%10.5f',delimiter=',')    
###########################################################################################################    
    