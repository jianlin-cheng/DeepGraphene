from os.path import expanduser
import numpy as np
import pandas as pd
import re
import collections
import time

class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("The running time of this code: %s " % self.elapsed(time.time() - self.start_time) )


def csv2ndarray(feature_path,feature_files):
    feature_input = []
    for i in range(len(feature_files)):
        new_mat = pd.read_csv(feature_path+feature_files[i], sep=',',index_col=False,header=None)
        new_mat = new_mat.as_matrix()
        new_mat = new_mat.reshape(new_mat.shape[0],new_mat.shape[1])
        feature_input.append(new_mat)

    feature_input_ndarray = np.array(feature_input,np.float)
    return(feature_input_ndarray)

def tsv2ndarray(feature_path,feature_files):
    feature_input = []
    for i in range(len(feature_files)):
        new_mat = pd.read_csv(feature_path+feature_files[i], sep='\t',index_col=False,header=None)
        new_mat = new_mat.as_matrix()
        new_mat = new_mat.reshape(new_mat.shape[0],new_mat.shape[1])
        feature_input.append(new_mat)

    feature_input_ndarray = np.array(feature_input)
    return(feature_input_ndarray)
    
def get_abs_path(input_path):
    home = expanduser("~")
    if re.match(r'^[A-Z]',home) :
        home = home + '\\Documents'
    input_path = re.sub(r'~',home,input_path)
    return(input_path)

def viz_model(Data,base_dir,Type,epochs):
    Gen_history=Data['generator']
    Dis_history=Data['discriminator']
    Gen_history=np.array(Gen_history,dtype=float)
    Dis_history=np.array(Dis_history,dtype=float)  
    image_dir=base_dir+'GANs_result/Loss_value/'+Type+'_'+str(epochs)+'.jpg'
    import matplotlib.pyplot as plt
    plt.plot(Gen_history[:,1])
    plt.plot(Gen_history[:,2])
    plt.plot(Dis_history[:,1])
    plt.plot(Dis_history[:,2])
    plt.title('GrapheneGANs_loss')
    plt.ylabel('Loss_Value')
    plt.xlabel('Epoch_Num')
    plt.legend(['Gen_Loss','Gen_aux','Dis_Loss','Dis_aux'], loc='upper left')
        
    plt.savefig(image_dir)  
    plt.show()
    plt.close()
        

def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el
            
def cartesian_iterative(pools):
  result = [[]]
  for pool in pools:
    result = [x+[y] for x in result for y in pool]
  return result