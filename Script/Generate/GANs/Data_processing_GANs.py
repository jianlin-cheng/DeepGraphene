# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 13:20:23 2018

@author: Herman Wu
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 22:00:57 2018

@author: Herman Wu
"""
from keras.models import load_model
import numpy as np
import csv
import time

def X_loading(data_dir,num_file,counting):
    print('\n**********************************************')
    print('   Now Loading the data of Input   ')
    print('**********************************************')
    time.sleep(2)
    DataBase_dir=data_dir+'dataset/Train_Data/InputX/'
    Data=[]
#-------------------------------------------------------------
#   the number of file in this folder
#-------------------------------------------------------------    
    for i in counting:        
        conditions=((i/(num_file+1))*100)
        count=int(conditions)
        condition=str(count)+'%'      
        if (conditions%5<=0.58):  
            print('The loading condition of X :' +condition )  
        temp=[]
        File_dir=DataBase_dir+'InputX_'+str(i)+'.csv'
        csv_file=csv.reader(open(File_dir,encoding='utf-8'))
        for i0 in csv_file:
            temp.append(i0)
        temp=np.array(temp,dtype=float)
        temp=temp.reshape(1,4,4,1)
        if len(Data)==0:
            Data=temp
        else:
            Data=np.concatenate((Data,temp),axis=0)
    print('\n*************  Done!!  *******************\n')
    time.sleep(1)
    return Data


def Y_loading(data_dir,num_file):
    print('\n**********************************************')
    print('   Now Loading the data of Output   ')
    print('**********************************************')
    time.sleep(1)
    DataBase_dir=data_dir+'/dataset/Train_Data/InputY/'
    Data=[]
    counting=[]
#-------------------------------------------------------------
#   the number of file in this folder
#-------------------------------------------------------------    
    for i in range(1,(num_file+1)):        
        conditions=((i/(num_file+1))*100)
        count=int(conditions)
        condition=str(count)+'%'      
        if (conditions%5<=0.18):  
            print('The loading condition of Y :' +condition )  
        File_dir=DataBase_dir+'InputY_'+str(i)+'.csv'
        csv_file=csv.reader(open(File_dir,encoding='utf-8'))
        for i0 in csv_file:
            i0=round(float(i0[0])*4.6,1)*10
        if i0<=25 and i0>=6:
            counting.append(i)            
            i0=(np.array(i0,dtype=float)).reshape(1,1)
            if len(Data)!=0 :
                Data=np.concatenate((Data,i0),axis=0)
            else:
                Data=i0
    print('\n*************  Done!!  *******************\n')
    time.sleep(1)
    return Data,counting

def loading(data_dir,num_file,Type):
    print('\n**********************************************')
    print('   Now Loading the data of '+ Type+' ')
    print('**********************************************')
    time.sleep(1)
    DataBase_dir=data_dir+'/dataset/InputX/'
    Data=[]
#-------------------------------------------------------------
#   the number of file in this folder
#-------------------------------------------------------------    
    for i in range(1,(num_file+1)):        
        conditions=((i/(num_file+1))*100)
        count=int(conditions)
        condition=str(count)+'%'      
        if (conditions%5<=0.18):  
            print('The loading condition of X :' +condition )  
        temp=[]
        Temp=[]
        File_dir=DataBase_dir+'InputX'+'_'+str(i)+'.csv'
        csv_file=csv.reader(open(File_dir,encoding='utf-8'))
        for i0 in csv_file:
            temp.append(i0)
        temp=np.array(temp,dtype=float)
        temp=temp.reshape(1,4,4,1)
        if i==1:
            Data=Temp
        else:
            Data=np.concatenate((Data,Temp),axis=0)
    print('\n*************  Done!!  *******************\n')
    time.sleep(2)
    return Data


def restore(data):
    Result=[]
    for i in data:
        temp=[]
        for i0 in range(0,3):
            img=i[:,:,i0]
            if i0:
                temp=np.concatenate((temp,img),axis=0)
            else:
                temp=img
        temp=(temp.reshape(1,120,800))*255
        if len(Result):
            Result=np.concatenate((Result,temp),axis=0)
        else:
            Result=temp
    return Result
    
def save(Model,Base_dir,epochs,Type='Generator'):
    if Type=='Generator':
        save_dir=Base_dir+'/Script/GANs/H5_File/Generator/Gen_'+str(epochs)+'.h5'
    elif Type=='Discriminator':
        save_dir=Base_dir+'/Script/GANs/H5_File/Discriminator/Dis_'+str(epochs)+'.h5'
    Model.save(save_dir)
    
def Data_process(data_dir,num_file):
    [InputY,count]=Y_loading(data_dir,num_file)
    InputX=X_loading(data_dir,num_file,count)
    return InputX,InputY