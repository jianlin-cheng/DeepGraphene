# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 10:20:51 2018

@author: Herman Wu
"""
import csv
import numpy as np

def write_data(local_dir,Type,num):
    file_dir=local_dir+'/Data/dataset/Train_Data_'+Type    
    save_dir_x=local_dir+'/dataset/InputX_'+Type
    save_dir_y=local_dir+'/dataset/InputY_'+Type
    Data_x=''
    Data_y=''
    for i in range(num):
        temp_x=[]
        temp_y=[]
        address_x=file_dir+'/InputX/InputX_'+str(i+1)+'.csv'
        address_y=file_dir+'/InputY/InputY_'+str(i+1)+'.csv'
        csv_file_x=csv.reader(open(address_x,encoding='utf-8'))
        csv_file_y=csv.reader(open(address_y,encoding='utf-8'))
        for i0 in csv_file_x:
            temp_x.append(i0)
        for i0 in csv_file_y:
            temp_y.append(i0)
        for i1 in temp_x[0]:
            if Data_x=='':
                Data_x='\t'+i1
            else:
                Data_x=Data_x+'\t'+i1
        Data_x=Data_x+'\n'
        for i1 in temp_y[0]:
            if Data_y=='':
                Data_y=i1
            else:
                Data_y=Data_y+'\n'+i1
    file=open(save_dir_x,'w')
    file.write(Data_x)
    file=open(save_dir_y,'w')
    file.write(Data_y)

def write_test_data(local_dir,Type,num=1000):
    file_dir=local_dir+'/Data/dataset/Test_Data_'+Type    
    save_dir_x=local_dir+'/dataset/InputX_'+Type+'_test'
    save_dir_y=local_dir+'/dataset/InputY_'+Type+'_test'
    Data_x=''
    Data_y=''
    for i in range(num):
        temp_x=[]
        temp_y=[]
        address_x=file_dir+'/InputX/InputX_'+str(i+1)+'.csv'
        address_y=file_dir+'/InputY/InputY_'+str(i+1)+'.csv'
        csv_file_x=csv.reader(open(address_x,encoding='utf-8'))
        csv_file_y=csv.reader(open(address_y,encoding='utf-8'))
        for i0 in csv_file_x:
            temp_x.append(i0)
        for i0 in csv_file_y:
            temp_y.append(i0)
        for i1 in temp_x[0]:
            if Data_x=='':
                Data_x='\t'+i1
            else:
                Data_x=Data_x+'\t'+i1
        Data_x=Data_x+'\n'
        for i1 in temp_y[0]:
            if Data_y=='':
                Data_y=i1
            else:
                Data_y=Data_y+'\n'+i1
    file=open(save_dir_x,'w')
    file.write(Data_x)
    file=open(save_dir_y,'w')
    file.write(Data_y)
    
#save_dir=local_dir+'/dataset/InputX_4by4'
#save_dir_y=local_dir+'/dataset/InputY_4by4'
            
# =============================================================================
# DataBase_dir='D:/Working_Application/DropBox_File/Dropbox/Graphene_DeepLearning/test_dataset'
# File_dir=DataBase_dir+'/InputX_1.csv'
# 
# csv_file=csv.reader(open(File_dir,encoding='utf-8'))
# Data=[]
# temp=[]
# for i0 in csv_file:
#     temp.append(i0)
# 
# text=''
# for i0 in temp[0]:
#     if text=='':
#         text=str(i0)
#     else:
#         text=text+'\t'+str(i0)
# text=text+'\n'
# 
# #temp=np.array(temp,dtype=float)
# Data=temp
# Data=np.concatenate((Data,temp),axis=0)
# 
# text=''
# for i0 in temp[0]:
#     if text=='':
#         text=str(i0)
#     else:
#         text=text+'\t'+str(i0)
# text=text+'\n'
# =============================================================================
