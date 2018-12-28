# DeepGraphene
 Paper: `Deep Learning Bandgaps of Topologically Doped Graphene` Yuan Dong, Chuhan Wu et.al <br/>
         Url: https://arxiv.org/abs/1809.10860  < Under reviewed Form >
## Introduction:
*   This Repo contain the source code of Paper `Deep Learning Bandgaps of topologically Doped Graphene` , it contains all algorithms which we use to predict graphene supercells' bandgap values (Graphene-SVR, VCN, RCN, CCN). Meanwhile it contains the latest data of graphene supercell ( 4by4: 13018, 5by5: 79647, 6by6: 6382). 
*   DeepGraphene is an interdiscipline research that implemented Machine Learning methods toward the bandgap values prediction problem. It described different type of Graphene supercell structure into 2-D matrix, them input these data into Deep Networks or SVR algorithm to extract their spatial and hidden features. Therefore we can predict graphene supercells data band-gap values based on its 2-D structure matrix.  
*  Brief workflow chart: ![](https://raw.githubusercontent.com/q145492675/DCGANs-DeepConvolutionalGANs-keras/master/image1.png)


## Requirement:
    * Tensorflow (1.11.0)
    * Keras (2.2.4)
    * Matlab (2017a version)
## Index :
* [Data set :](./Graphene_DeepLearning/dataset) (This folder contain the data of Graphene supercells)
    * Data
    * data_script
    * Original_data
    * Processed_Dataset
* GANs_result : 
    * H5_file
    * Loss_value
    * Model_Image
    * Predict_resul
* [Predict_h5file :](./Graphene_DeepLearning/) (This folder contain all Deep neural netowkr models we have trained [except Graphene_SVR])
    * GAN_h5file.zip
    * h5_file.zip
* [Script :](./Graphene_DeepLearning/Script)  (This folder contain all scripts we use)
    * Generate
    * Predict 
        * [DeepGraphene ](./Graphene_DeepLearning/Script/Predict/DeepGraphene)  (This folder contains Deep Neural Network algorithms' script: VCN, RCN, CCN)
        * [Graphene_SVR ](./Graphene_DeepLearning/Script/Predict/Graphene_SVR) (This folder contains traditional machine learning algorithm' script: Graphene_SVR)
