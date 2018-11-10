# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 22:00:57 2018

@author: Herman Wu
"""

import numpy as np
import csv
import time

def X_loading(data_dir,Type,num_file):
    print('\n**********************************************')
    print('   Now Loading the data of Input ('+ Type+') :' )
    print('**********************************************')
    time.sleep(2)
    DataBase_dir=data_dir+'/dataset/'+Type+'/InputX/'
    Data=[]
#-------------------------------------------------------------
#   the number of file in this folder
#-------------------------------------------------------------    
    for i in range(1,(num_file+1)):        
        conditions=((i/(num_file+1))*100)
        count=int(conditions)
        condition=str(count)+'%'      
        if (conditions%5<=0.01):  
            print('The loading condition of X :' +condition )  
        temp=[]
        File_dir=DataBase_dir+'InputX'+'_'+str(i)+'.csv'
        csv_file=csv.reader(open(File_dir,encoding='utf-8'))
        for i0 in csv_file:
            temp.append(i0)
        temp=np.array(temp,dtype=float)
        temp=temp.reshape(1,4,4,1)
        if i==1:
            Data=temp
        else:
            Data=np.concatenate((Data,temp),axis=0)
    print('\n*************  Done!!  *******************\n')
    time.sleep(1)
    return Data

def X_loading_5by5(data_dir,Type,num_file):
    print('\n**********************************************')
    print('   Now Loading the data of Input ('+ Type+'） :' )
    print('**********************************************')
    time.sleep(2)
    DataBase_dir=data_dir+'/dataset/'+Type+'/InputX/'
    Data=[]
#-------------------------------------------------------------
#   the number of file in this folder
#-------------------------------------------------------------    
    for i in range(1,(num_file+1)):        
        conditions=((i/(num_file+1))*100)
        count=int(conditions)
        condition=str(count)+'%'      
        if (conditions%5<=0.01):  
            print('The loading condition of X :' +condition )  
        temp=[]
        File_dir=DataBase_dir+'InputX'+'_'+str(i)+'.csv'
        csv_file=csv.reader(open(File_dir,encoding='utf-8'))
        for i0 in csv_file:
            temp.append(i0)
        temp=np.array(temp,dtype=float)
        temp=temp.reshape(1,5,5,1)
        if i==1:
            Data=temp
        else:
            Data=np.concatenate((Data,temp),axis=0)
    print('\n*************  Done!!  *******************\n')
    time.sleep(1)
    return Data

def Y_loading(data_dir,Type,num_file):
    print('\n**********************************************')
    print('   Now Loading the data of Output ('+Type+') :')
    print('**********************************************')
    time.sleep(1)
    DataBase_dir=data_dir+'/dataset/'+Type+'/InputY/'
    Data=[]
#-------------------------------------------------------------
#   the number of file in this folder
#-------------------------------------------------------------    
    for i in range(1,(num_file+1)):        
        conditions=((i/(num_file+1))*100)
        count=int(conditions)
        condition=str(count)+'%'      
        if (conditions%5<=0.01):  
            print('The loading condition of Y :' +condition )  
        temp=[]
        File_dir=DataBase_dir+'InputY'+'_'+str(i)+'.csv'
        csv_file=csv.reader(open(File_dir,encoding='utf-8'))
        for i0 in csv_file:
            temp.append(i0)
        temp=np.array(temp,dtype=float)
        temp=temp.reshape(1,1)
        if i==1:
            Data=temp
        else:
            Data=np.concatenate((Data,temp),axis=0)
    print('\n*************  Done!!  *******************\n')
    time.sleep(1)
    return Data

def restore_save(Real,predict,random_type,Type,original_dir):
        random_type+=-np.ones(len(random_type),dtype=int)
        original_dir=original_dir+'/data_0311.csv'
        csv_file=csv.reader(open(original_dir,encoding='utf-8'))
        temp=[]
        temp0=[]
        for i0 in csv_file:
            temp.append(i0)
        temp=np.array(temp,dtype=float)        
        temp=temp.reshape(-1,2)
        index=(temp[:,0]).reshape(-1,1)
        for i in random_type:
            temp0.append(index[i])
        temp0=np.array(temp0,dtype=int)
        predict_file=np.concatenate((temp0,Real),axis=1)
        predict_file=np.concatenate((predict_file,predict),axis=1)
        csv_dir='D:/Working_Application/DropBox_File/Dropbox/Graphene_DeepLearning/predict/'+Type+'_predict.csv'    
        np.savetxt(csv_dir,predict_file,fmt='%10.5f',delimiter=',')

def merge_Hash(data_4by4,data_5by5):
    Hash={'4by4_data':data_4by4,'5by5_data':data_5by5}
    return Hash

def Data_process(data_dir,Four_num=10000,Five_num=10000):
    InputX_4by4=X_loading(data_dir,'Train_Data',Four_num)
    InputX_5by5=X_loading_5by5(data_dir,'Train_Data_5by5',Five_num)
    InputY_4by4=Y_loading(data_dir,'Train_Data',Four_num)
    InputY_5by5=Y_loading(data_dir,'Train_Data_5by5',Five_num)
    InputX=merge_Hash(InputX_4by4,InputX_5by5)
    InputY=merge_Hash(InputY_4by4,InputY_5by5)
    return InputX,InputY

def Test_Data_process(data_dir,num_file):
    InputX_4by4=X_loading(data_dir,'Test_Data',num_file)
    InputX_5by5=X_loading_5by5(data_dir,'Test_Data_5by5',num_file)
    InputY_4by4=Y_loading(data_dir,'Test_Data',num_file)
    InputY_5by5=Y_loading(data_dir,'Test_Data_5by5',num_file)
    InputX=merge_Hash(InputX_4by4,InputX_5by5)
    InputY=merge_Hash(InputY_4by4,InputY_5by5)
    return InputX,InputY


'''
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
 '''