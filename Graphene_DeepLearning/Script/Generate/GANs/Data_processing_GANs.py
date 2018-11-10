# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 13:20:23 2018

@author: Herman Wu
"""

from keras.models import load_model
import numpy as np
import csv
import time

def load_data(local_dir,Type):
    print('\n**********************************************')
    print('   Now Loading the data of Input ('+ Type+') :' )
    print('**********************************************')
    dir_x=local_dir+'/dataset/Processed_Dataset/InputX_'+Type
    dir_y=local_dir+'/dataset/Processed_Dataset/InputY_'+Type
    Input_X=[]
    Input_Y=[]
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
       temp=round(float(temp[0])*4.6,1)*10
       temp=np.array(temp,dtype=int)
       temp=temp.reshape(1,1)
       if data_y=='':
           data_y=temp
       else:
           data_y=np.concatenate((data_y,temp),axis=0)
    count=0
    for index,i in enumerate(data_y):
         if i<=25 and i>=6:
             if count==0:
                 Input_X=data_x[index,:,:,:]
                 if Type=='4by4' or Type=='4by4_test':
                     Input_X=Input_X.reshape(1,4,4,1)
                 if Type=='5by5'or Type=='5by5_test':
                     Input_X=Input_X.reshape(1,5,5,1)
                 if Type=='6by6'or Type=='6by6_test':
                     Input_X=Input_X.reshape(1,6,6,1)
                 Input_Y=i
                 Input_Y=Input_Y.reshape(1,1)
                 count+=1
             else:
                 temp=data_x[index,:,:,:]
                 if Type=='4by4' or Type=='4by4_test':
                     temp=temp.reshape(1,4,4,1)
                 if Type=='5by5'or Type=='5by5_test':
                     temp=temp.reshape(1,5,5,1)
                 if Type=='6by6'or Type=='6by6_test':
                     temp=temp.reshape(1,6,6,1) 
                 Input_X=np.concatenate((Input_X,temp),axis=0)
                 i=i.reshape(1,1)
                 Input_Y=np.concatenate((Input_Y,i),axis=0)
    print('\n*************  Done!!  *******************\n')
    return Input_X,Input_Y


# =============================================================================
# def X_loading(data_dir,num_file,counting):
#     print('\n**********************************************')
#     print('   Now Loading the data of Input   ')
#     print('**********************************************')
#     time.sleep(2)
#     DataBase_dir=data_dir+'dataset/Train_Data/InputX/'
#     Data=[] 
# #-------------------------------------------------------------
# #   the number of file in this folder
# #-------------------------------------------------------------    
#     for i in counting:        
#         conditions=((i/(num_file+1))*100)
#         count=int(conditions)
#         condition=str(count)+'%'      
#         if (conditions%5<=0.009):  
#             print('The loading condition of X :' +condition )  
#         temp=[]
#         File_dir=DataBase_dir+'InputX_'+str(i)+'.csv'
#         csv_file=csv.reader(open(File_dir,encoding='utf-8'))
#         for i0 in csv_file:
#             temp.append(i0)
#         temp=np.array(temp,dtype=float)
#         temp=temp.reshape(1,4,4,1)
#         if len(Data)==0:
#             Data=temp
#         else:
#             Data=np.concatenate((Data,temp),axis=0)
#     print('\n*************  Done!!  *******************\n')
#     time.sleep(1)
#     return Data
# 
# 
# def Y_loading(data_dir,num_file):
#     print('\n**********************************************')
#     print('   Now Loading the data of Output   ')
#     print('**********************************************')
#     time.sleep(1)
#     DataBase_dir=data_dir+'/dataset/Train_Data/InputY/'
#     Data=[]
#     counting=[]
# #-------------------------------------------------------------
# #   the number of file in this folder
# #-------------------------------------------------------------    
#     for i in range(1,(num_file+1)):        
#         conditions=((i/(num_file+1))*100)
#         count=int(conditions)
#         condition=str(count)+'%'      
#         if (conditions%5<=0.009):  
#             print('The loading condition of Y :' +condition )  
#         File_dir=DataBase_dir+'InputY_'+str(i)+'.csv'
#         csv_file=csv.reader(open(File_dir,encoding='utf-8'))
#         for i0 in csv_file:
#             i0=round(float(i0[0])*4.6,1)*10
#         if i0<=25 and i0>=6:
#             counting.append(i)            
#             i0=(np.array(i0,dtype=float)).reshape(1,1)
#             if len(Data)!=0 :
#                 Data=np.concatenate((Data,i0),axis=0)
#             else:
#                 Data=i0
#     print('\n*************  Done!!  *******************\n')
#     time.sleep(1)
#     return Data,counting
# =============================================================================

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
        if (conditions%5<=0.009):  
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
    
def save(Model,Base_dir,epochs,version,Type='Generator'):
    if Type=='Generator':
        save_dir=Base_dir+'GANs_result/H5_file/Gen_'+version+"_"+str(epochs)+'.h5'
    elif Type=='Discriminator':
        save_dir=Base_dir+'GANs_result/H5_file/Dis_'+version+"_"+str(epochs)+'.h5'
    Model.save(save_dir)

def save_Generate_Result(Base_dir,Data,labels):
    data_dir=Base_dir+'GANs_result/Predict_result/Structure'
    for i in range(len(Data)):
        Data_label=(labels[i]+6)/10
        Data_file=Data[i,:,:]    
        csv_dir=data_dir+'/Label_'+str(Data_label)+'_No_'+str(i)+'_Data.csv'
        np.savetxt(csv_dir,Data_file,fmt='%10.5f',delimiter=',')
    
# =============================================================================
# def Data_process(data_dir,num_file):
#     [InputY,count]=Y_loading(data_dir,num_file)
#     InputX=X_loading(data_dir,num_file,count)
#     return InputX,InputY
# =============================================================================
