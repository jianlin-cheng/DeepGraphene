from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd

def csv2ndarray(feature_path):
    feature_files = [f for f in listdir(feature_path) if isfile(join(feature_path, f))]
    feature_input = []
    feature_files.sort(key=lambda f: int(''.join(list(filter(str.isdigit, f)))))
    for i in range(len(feature_files)):
        new_mat = pd.read_csv(feature_path+feature_files[i], sep=',',index_col=False,header=None)
        new_mat = new_mat.as_matrix()
        new_mat = new_mat.reshape(new_mat.shape[0],new_mat.shape[1])
        feature_input.append(new_mat)

    feature_input_ndarray = np.array(feature_input,np.float)
    return(feature_input_ndarray)

def viz_model(history,outfile):
    history=history.history
    import matplotlib.pyplot as plt
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    
    plt.savefig(outfile)  
    plt.show()
    plt.close()