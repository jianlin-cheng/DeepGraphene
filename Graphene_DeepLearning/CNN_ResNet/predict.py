# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 22:28:53 2018

@author: Herman Wu
"""

import csv
import os
import sys
import numpy as np
import keras.backend as K
from keras.models import load_model

Type=sys.argv[1]
Out_Dir=sys.argv[2]
Data_Num=int(sys.argv[3])

def mean_pred(y_true,y_pred):
    return K.mean(y_pred)

if __name__==' __main__':
    Local_dir = os.path.dirname(__file__)
    Base_dir=(Local_dir.split('Script'))[0]
    Data_dir=Base_dir+'/test_data/'
#------------------------------------------------------------------------------
    if Type=='CNN':
        h5_dir=Base_dir+'/h5_file/CNN_Standard.h5'
    elif Type=='ResNet':
        h5_dir=Base_dir+'/h5_file/ResNet.h5'
    elif Type=='LSTM':
        h5_dir=Base_dir+'/h5_file/CNN+LSTM.h5'
    print('/*****************************************/')
#------------------------------------------------------------------------------
    Test_Data=[]
    for i in range(1,(Data_Num+1)):
        File_dir=Data_dir+'Test'+'_'+str(i)+'.csv'
        csv_file=csv.reader(open(File_dir,encoding='utf-8'))
        temp=[]
        for i0 in csv_file:
            temp.append(i0)
        temp=np.array(temp,dtype=float)  
        temp=temp.reshape(1,-1,-1,1)
        if i==1:
            Test_Data=temp
        else:
            Test_Data=np.concatenate((Test_Data,temp),axis=0)
#------------------------------------------------------------------------------    
    Model=load_model(h5_dir,custom_objects={'mean_pred':mean_pred})
    Result=Model.predict(Test_Data)
    np.savetxt(Out_Dir,Result,fmt='%10.5f',delimiter=',')
#------------------------------------------------------------------------------