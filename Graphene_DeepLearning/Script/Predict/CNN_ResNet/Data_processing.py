# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 22:00:57 2018

@author: Herman Wu
"""

import numpy as np
import csv
import time


def load_data(local_dir,Type):
    print('\n**********************************************')
    print('   Now Loading the data of Input ('+ Type+') :' )
    print('**********************************************')
    dir_x=local_dir+'/dataset/Processed_Dataset/InputX_'+Type
    dir_y=local_dir+'/dataset/Processed_Dataset/InputY_'+Type
    data_x=''
    data_y=''
    fp_text_x = open(dir_x, "r")
    fp_text_y = open(dir_y, "r")
    for line in fp_text_x:
       col=line.split("\t")
       if '' in col:
           col.remove('')
       temp=np.array(col,dtype=float)
       if Type=='4by4' or Type=='4by4_test':
           temp=temp.reshape(1,4,4,1)
       if Type=='5by5'or Type=='5by5_test':
           temp=temp.reshape(1,5,5,1)
       if Type=='6by6'or Type=='6by6_test':
           temp=temp.reshape(1,6,6,1)
       if data_x=='':
           data_x=temp
       else:
           data_x=np.concatenate((data_x,temp),axis=0)
    for line in fp_text_y:
       col=line.split("\n")
       if '' in col:
           col.remove('')
       temp=np.array(col,dtype=float)
       temp=temp.reshape(1,1)
       if data_y=='':
           data_y=temp
       else:
           data_y=np.concatenate((data_y,temp),axis=0)
    print('\n*************  Done!!  *******************\n')
    return data_x,data_y

def merge_Hash(data_4by4,data_5by5,data_6by6):
    Hash={'4by4_data':data_4by4,'5by5_data':data_5by5,'6by6_data':data_6by6}
    return Hash

def Data_loading(data_dir):
    InputX_4by4,InputY_4by4=load_data(data_dir,'4by4')
    InputX_5by5,InputY_5by5=load_data(data_dir,'5by5')
    InputX_6by6,InputY_6by6=load_data(data_dir,'6by6')
    InputX=merge_Hash(InputX_4by4,InputX_5by5,InputX_6by6)
    InputY=merge_Hash(InputY_4by4,InputY_5by5,InputY_6by6)
    return InputX,InputY

def Test_Data_loading(data_dir):
    InputX_4by4,InputY_4by4=load_data(data_dir,'4by4_test')
    InputX_5by5,InputY_5by5=load_data(data_dir,'5by5_test')
    InputX_6by6,InputY_6by6=load_data(data_dir,'6by6_test')
    InputX=merge_Hash(InputX_4by4,InputX_5by5,InputX_6by6)
    InputY=merge_Hash(InputY_4by4,InputY_5by5,InputY_6by6)
    return InputX,InputY

# =============================================================================
# def X_loading(data_dir,Type,num_file):
#     print('\n**********************************************')
#     print('   Now Loading the data of Input ('+ Type+') :' )
#     print('**********************************************')
#     time.sleep(2)
#     DataBase_dir=data_dir+'/dataset/'+Type+'/InputX/'
# #    DataBase_dir=data_dir+'/dataset/'+Type+'/Data/'
#     Data=[]
# #-------------------------------------------------------------
# #   the number of file in this folder
# #-------------------------------------------------------------    
#     for i in range(1,(num_file+1)):        
#         conditions=((i/(num_file+1))*100)
#         count=int(conditions)
#         condition=str(count)+'%'      
#         if (conditions%5<=0.01):  
#             print('The loading condition of X :' +condition)  
#         temp=[]
#         File_dir=DataBase_dir+'InputX'+'_'+str(i)+'.csv'
# #        File_dir=DataBase_dir+'Test'+'_'+str(i)+'.csv'
#         csv_file=csv.reader(open(File_dir,encoding='utf-8'))
#         for i0 in csv_file:
#             temp.append(i0)
#         temp=np.array(temp,dtype=float)
#         if (Type=='Train_Data') or (Type=='Test_Data'):
#             temp=temp.reshape(1,4,4,1)
#         elif (Type=='Train_Data_5by5') or (Type=='Test_Data_5by5'):
#             temp=temp.reshape(1,5,5,1)
#         elif (Type=='Train_Data_6by6') or (Type=='Test_Data_6by6')or( (Type=='Create_Data_6by6')):
#             temp=temp.reshape(1,6,6,1)
#         if i==1:
#             Data=temp
#         else:
#             Data=np.concatenate((Data,temp),axis=0)
#     print('\n*************  Done!!  *******************\n')
#     time.sleep(1)
#     return Data
# 
# def Y_loading(data_dir,Type,num_file):
#     print('\n**********************************************')
#     print('   Now Loading the data of Output ('+Type+') :')
#     print('**********************************************')
#     time.sleep(1)
#     DataBase_dir=data_dir+'/dataset/'+Type+'/InputY/'
#     Data=[]
# #-------------------------------------------------------------
# #   the number of file in this folder
# #-------------------------------------------------------------    
#     for i in range(1,(num_file+1)):        
#         conditions=((i/(num_file+1))*100)
#         count=int(conditions)
#         condition=str(count)+'%'      
#         if (conditions%5<=0.01):  
#             print('The loading condition of Y :' +condition )  
#         temp=[]
#         File_dir=DataBase_dir+'InputY'+'_'+str(i)+'.csv'
#         csv_file=csv.reader(open(File_dir,encoding='utf-8'))
#         for i0 in csv_file:
#             temp.append(i0)
#         temp=np.array(temp,dtype=float)
#         temp=temp.reshape(1,1)
#         if i==1:
#             Data=temp
#         else:
#             Data=np.concatenate((Data,temp),axis=0)
#     print('\n*************  Done!!  *******************\n')
#     time.sleep(1)
#     return Data
# 
# def restore_save(Real,predict,random_type,Type,original_dir):
#         random_type+=-np.ones(len(random_type),dtype=int)
#         original_dir=original_dir+'/data_0311.csv'
#         csv_file=csv.reader(open(original_dir,encoding='utf-8'))
#         temp=[]
#         temp0=[]
#         for i0 in csv_file:
#             temp.append(i0)
#         temp=np.array(temp,dtype=float)        
#         temp=temp.reshape(-1,2)
#         index=(temp[:,0]).reshape(-1,1)
#         for i in random_type:
#             temp0.append(index[i])
#         temp0=np.array(temp0,dtype=int)
#         predict_file=np.concatenate((temp0,Real),axis=1)
#         predict_file=np.concatenate((predict_file,predict),axis=1)
#         csv_dir='D:/Working_Application/DropBox_File/Dropbox/Graphene_DeepLearning/predict/'+Type+'_predict.csv'    
#         np.savetxt(csv_dir,predict_file,fmt='%10.5f',delimiter=',')
# 
# def newX_loading(data_dir,Type,num_file):
#     print('\n**********************************************')
#     print('   Now Loading the data of Input ('+ Type+') :' )
#     print('**********************************************')
#     time.sleep(2)
#     DataBase_dir=data_dir+'/dataset/'+Type+'/InputX/'
#     Data=[]
# #-------------------------------------------------------------
# #   the number of file in this folder
# #-------------------------------------------------------------    
#     for i in range(1,(num_file+1)):        
#         conditions=((i/(num_file+1))*100)
#         count=int(conditions)
#         condition=str(count)+'%'      
#         if (conditions%5<=0.01):  
#             print('The loading condition of X :' +condition )  
#         temp=[]
#         File_dir=DataBase_dir+'Test'+'_'+str(i)+'.csv'
#         csv_file=csv.reader(open(File_dir,encoding='utf-8'))
#         for i0 in csv_file:
#             temp.append(i0)
#         temp=np.array(temp,dtype=float)
#         temp=temp.reshape(1,4,4,1)
#         if Type=='Train_Data':
#             temp=temp.reshape(1,4,4,1)
#         elif Type=='Train_Data_5by5':
#             temp=temp.reshape(1,5,5,1)
#         elif Type=='Train_Data_6by6':
#             temp=temp.reshape(1,6,6,1)
#         if i==1:
#             Data=temp
#         else:
#             Data=np.concatenate((Data,temp),axis=0)
#     print('\n*************  Done!!  *******************\n')
#     time.sleep(1)
#     return Data
# =============================================================================

# =============================================================================
# def Data_process(data_dir,Four_num=10000,Five_num=10000,Six_num=10000):
#     InputX_4by4=X_loading(data_dir,'Train_Data',Four_num)
#     InputX_5by5=X_loading(data_dir,'Train_Data_5by5',Five_num)
#     InputX_6by6=X_loading(data_dir,'Train_Data_6by6',Six_num)
#     InputY_4by4=Y_loading(data_dir,'Train_Data',Four_num)
#     InputY_5by5=Y_loading(data_dir,'Train_Data_5by5',Five_num)
#     InputY_6by6=Y_loading(data_dir,'Train_Data_6by6',Six_num)
#     InputX=merge_Hash(InputX_4by4,InputX_5by5,InputX_6by6)
#     InputY=merge_Hash(InputY_4by4,InputY_5by5,InputY_6by6)
#     return InputX,InputY
# 
# def Test_Data_process(data_dir,num_file):
#     InputX_4by4=X_loading(data_dir,'Test_Data',num_file)
#     InputX_5by5=X_loading(data_dir,'Test_Data_5by5',num_file)
#     InputX_6by6=X_loading(data_dir,'Test_Data_6by6',num_file)
#     InputY_4by4=Y_loading(data_dir,'Test_Data',num_file)
#     InputY_5by5=Y_loading(data_dir,'Test_Data_5by5',num_file)
#     InputY_6by6=Y_loading(data_dir,'Test_Data_6by6',num_file)
#     InputX=merge_Hash(InputX_4by4,InputX_5by5,InputX_6by6)
#     InputY=merge_Hash(InputY_4by4,InputY_5by5,InputY_6by6)
#     return InputX,InputY
# 
# def new_Data_process(data_dir,num_file):
#     InputX_4by4=newX_loading(data_dir,'Create_Data_4by4',num_file)
#     InputX_5by5=newX_loading(data_dir,'Create_Data_5by5',num_file)
#     InputX_6by6=newX_loading(data_dir,'Create_Data_6by6',num_file)
#     InputX=merge_Hash(InputX_4by4,InputX_5by5,InputX_6by6)
#     return InputX
# =============================================================================

