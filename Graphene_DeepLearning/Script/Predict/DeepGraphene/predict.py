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
import re
import pandas as pd
#Type=sys.argv[1]
#Out_Dir=sys.argv[2]
#Data_Num=int(sys.argv[3])

Type='ResNet'
Data_Num=114
num_type='high'

def mean_pred(y_true,y_pred):
    return K.mean(y_pred)

def ConvertToNumber(Dataset,size):
    number=[]
    for data in Dataset:
        temp=[]
        for i in range(size):
            for j in range(size):
                temp.append(str(int(data[i,j,:])))
        temp=(''.join(temp))
        temp_10=int(temp,2)
        number.append(temp_10)
    return number

def ConvertStructure(number):
    structure=np.zeros((1,5,5,1))
    for i in range(5):
        for j in range(5):
            structure[0,i,j,0]=int(number[i*5+j])    
    
    return structure

def countConcentration(original_number,number,low,middle,high,low_s,middle_s,high_s):
    count=0
    for i in number:
        if i=='1':
            count+=1
    if 3<=count<=9:
        low.append(original_number)
        structure=ConvertStructure(number)
        low_s.append(structure)
    elif 10<=count<=15:
        middle.append(original_number)
        structure=ConvertStructure(number)
        middle_s.append(structure)
    elif 16<=count<=22:
        high.append(original_number)
        structure=ConvertStructure(number)
        high_s.append(structure)        

def createnumber(size,Type):
    if Type=='4by4':
        number=np.random.randint(1,pow(2,16),size)
    elif Type=='5by5':
        number=np.random.randint(1,pow(2,25),size)
    elif Type=='6by6':
        number=np.random.randint(1,pow(2,36),size)
    number=set(number)
    number=list(number)
    number=np.array(number)
    high=[];    middle=[];  low=[];
    high_s=[];  middle_s=[];    low_s=[];
    kk=re.compile(r'\d+')
    for i in number:
        kk1=re.findall(kk,i)
        temp=kk1[1]
#        if len(temp)!=25:
            
#        countConcentration(i,temp,low,middle,high,low_s,middle_s,high_s)        
#    return_number={'high':high,'high_s':high_s,'middle':middle,'middle_s':middle_s,'low':low,'low_s'：low_s}
    return return_number
    
#if __name__==' __main__':
Local_dir = os.path.dirname(__file__)
Base_dir=(Local_dir.split('Script'))[0]
Data_dir=Base_dir+'/dataset/Data/Create_Data_5by5/'+num_type+'_Data/'
result_dir=(Data_dir.split(num_type+'_Data'))[0]+'/Result/'+num_type+'_'+Type+'_result.csv'
number_dir=(Data_dir.split(num_type+'_Data'))[0]+'Test_5by5_'+num_type+'_Index.csv'
number=pd.read_csv(number_dir,header=None,dtype=np.int32,float_precision = '%10.8f')
number.rename(columns={0:'Data_number'},inplace=True)
#------------------------------------------------------------------------------
if Type=='CNN':
    h5_dir=Base_dir+'predict_h5file/total_TF6by6_CNN.h5'
elif Type=='ResNet':
    h5_dir=Base_dir+'predict_h5file/total_TF6by6_ResNet.h5'
elif Type=='Concat':
    h5_dir=Base_dir+'predict_h5file/total_TF6by6_ConcatNet.h5'
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
    temp=temp.reshape(1,5,5,1)
    if i==1:
        Test_Data=temp
    else:
        Test_Data=np.concatenate((Test_Data,temp),axis=0)
#------------------------------------------------------------------------------    
Model=load_model(h5_dir)#,custom_objects={'mean_pred':mean_pred})
Result=Model.predict(Test_Data)*4.6
number.insert(1,'Bandgap_value',Result)
number.to_csv(result_dir)
 #   np.savetxt(Out_Dir,Result,fmt='%10.5f',delimiter=',')
#------------------------------------------------------------------------------