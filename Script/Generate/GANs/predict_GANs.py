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
    label=17
    epoch=10000
    latent_size=50
###########################################################################################################    
    h5_dir=Base_dir+'/H5_file/Generator/Gen_'+str(epoch)+'.h5'
    Gen=load_model(h5_dir)
    noise_Gen=np.random.randint(-1,1,(1,latent_size))    
    labels=np.array(label,dtype=float)
    labels=labels.reshape(1,1)
    Generate_data=Gen.predict([noise_Gen,labels])
    Generate_data=Generate_data.reshape(4,4)
    csv_file=Base_dir+'/predict/Label_'+str(labels[0])+'_Epoch_'+str(epoch)+'_.csv'
    np.savetxt(csv_file,Generate_data,fmt='%10.5f',delimiter=',')    
###########################################################################################################    
    