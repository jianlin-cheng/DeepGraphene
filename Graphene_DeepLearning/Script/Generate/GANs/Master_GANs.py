# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 13:21:28 2018

@author: Herman Wu
"""

import os
import numpy as np
import Freeze_Result
import Data_processing_GANs
import ACGANs
import numpy as np
import time
from keras.models import load_model
from keras.models import Sequential
from keras.utils import plot_model

#if __name__==' __main__':
def main(version="V1",Epc=500,Type='4by4'):
    Freeze_Result.main()
    Local_dir = os.path.dirname(__file__)
    Base_dir=(Local_dir.split('Script'))[0]
    sample_data_num=1000
###########################################################################################################
    Input_X,Input_Y=Data_processing_GANs.load_data(Base_dir,'4by4')
    print(' \n           Finished Loading the Data!!  \n ')
    print('/**********************************************7*********/')
    time.sleep(2)
####0#############################################################G##########################################
    [Gen,Dis,train,test]=ACGANs.main(Base_dir,Input_X,Input_Y,nb_epochs=Epc,batch_size=256)
#    Epc_Change=''
#    Data_processing_GANs.save(Gen,Base_dir,Epc_Change,version,Type='Generator')
#    Data_processing_GANs.save(Dis,Base_dir,Epc_Change,version,Type='Discriminator')
    #Model=ResNet.main(Input_X,Input_Y,epoch=Epc,batch_size=40)
    print('            Finished Training the Network!!  \n ')
    print('/*******************************************************/')

#################################################################G##########################################
    rela_error_sum=0
    abs_error_sum=0
    per_error_sum=0

    for i in range(4):
        noise_dis=np.random.normal(0,0.5,(sample_data_num,100))
#        noise_dis = np.random.uniform(-1, 1, (sample_data_num, 100))
        sampled_labels=np.random.randint(0,20,sample_data_num)       
        generate_data=Gen.predict([noise_dis,sampled_labels],verbose=0)
        generated_data=generate_data.reshape(sample_data_num,4,4,1)
        generate_labels=(sampled_labels+6)/10
        for c in range(len(generated_data)):
            for i in range(4):
                for j in range(4):
                    if generated_data[c,i,j]>0.5:
                        generated_data[c,i,j]=1
                    else:
                        generated_data[c,i,j]=0
        
        Model_dir=Base_dir+'predict_h5file/GANs_Test/'+Type+'.h5'
        Test_Model=Sequential()
        Test_Model=load_model(Model_dir)
        Result=(Test_Model.predict(generated_data))*4.6
        rela_error=0
        abs_error=0
        per_error=0
        temp_ava=len(generate_labels)
        for index,i in enumerate(generate_labels):
            if Result[index]==0:
                temp_ava+=-1
            else:
                temp=abs(Result[index]-i)/Result[index]
                rela_error+=temp
        rela_error=rela_error/temp_ava
        
        for index,i in enumerate(generate_labels):
            if i!=0:
                temp=abs(Result[index]-i)
                abs_error+=temp
        abs_error=abs_error/1000
        per_error=rela_error*100
    
        rela_error_sum+=rela_error
        abs_error_sum+=abs_error
        per_error_sum+=per_error
        
    rela_error_sum=rela_error_sum/4
    abs_error_sum=abs_error_sum/4
    per_error_sum=per_error_sum/4
    print('/*******************************************************/')
    print(' \n     The  relative error is:'+str(rela_error_sum)+'   \n')
    print(' \n     The  Absolute error is:'+str(abs_error_sum)+'   \n')
    print(' \n     The  Percent error is : '+str(per_error_sum)+' %   \n')
    print('/*******************************************************/')

    if  per_error_sum<17.456772:
        Epc_Change='Best'        
        Data_processing_GANs.save(Gen,Base_dir,Epc_Change,version,Type='Generator')
        Data_processing_GANs.save(Dis,Base_dir,Epc_Change,version,Type='Discriminator')        
#    Gen_model_dir=Base_dir+'/GANs_result/Model_Image/Generator_'+str(Epc)+'.png'
#    Dis_model_dir=Base_dir+'/GANs_result/Model_Image/Discriminator_'+str(Epc)+'.png'
#    plot_model(Gen,to_file=Gen_model_dir)
#    plot_model(Dis,to_file=Dis_model_dir)
    #Data_processing_GANs.save_Generate_Result(Base_dir,generated_data,sampled_labels)                
    return rela_error_sum,abs_error_sum,per_error_sum