# DeepGraphene
 Paper: `Deep Learning Bandgaps of Topologically Doped Graphene` Yuan Dong, Chuhan Wu, Chi Zhang, Yingda Liu, Jian Lin, JianLin Cheng  <br/>
         Url: https://arxiv.org/abs/1809.10860  < Under reviewed Form >

## Information:
|Author|Chuhan Wu|Dong Yuan|JianLin cheng|Jian Lin|
|---|---|---|---|---
|E-mail|cwdg9@mail.missouri.edu|dongyu@missouri.edu|chengji@missouri.edu|linjian@missouri.edu

 

## Introduction:
*   This Repo contain the source code of Paper `Deep Learning Bandgaps of topologically Doped Graphene` , it contains all algorithms which we use to predict graphene supercells' bandgap values (Graphene-SVR, VCN, RCN, CCN). Meanwhile it contains the latest data of graphene supercell ( 4by4: 13018, 5by5: 79647, 6by6: 6382). 
*   DeepGraphene is an interdiscipline research that implemented Machine Learning methods toward the bandgap values prediction problem. It described different type of Graphene supercell structure into 2-D matrix, them input these data into Deep Networks or SVR algorithm to extract their spatial and hidden features. Therefore we can predict graphene supercells data band-gap values based on its 2-D structure matrix.  

## Brief workflow chart : <br/>
**How to describe Graphene supercell structure:** ![](https://github.com/jianlin-cheng/DeepGraphene/blob/master/Image/image1.png)
 <br/> <br/>
**The workflow chart (VCN)** ![](https://github.com/jianlin-cheng/DeepGraphene/blob/master/Image/image2.png)


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
* [Predict_h5file :](./Graphene_DeepLearning/) (This folder contain all Deep neural netowkr models we have trained [except Graphene_SVR])
    * h5_file.zip
* [Script :](./Graphene_DeepLearning/Script)  (This folder contain all scripts we use)
    * Predict 
        * [DeepGraphene ](./Graphene_DeepLearning/Script/Predict/DeepGraphene)  (This folder contains Deep Neural Network algorithms' script: VCN, RCN, CCN)
        * [Graphene_SVR ](./Graphene_DeepLearning/Script/Predict/Graphene_SVR) (This folder contains traditional machine learning algorithm' script: Graphene_SVR)
        
## Usage:
* Clone the whole repo into your local address.
* If you want to train DeepGraphene Neural netowrk. Go into [Script folader](./Graphene_DeepLearning/Script/Predict/DeepGraphene) , click [Master.py](./Graphene_DeepLearning/Script/Predict/DeepGraphene/Master.py) . The `line 23` to `line 29` You can set which algorithm you want to choice: `VCN`, `RCN` or `CCN`, if you select one, please annotate other algorithms. 
    * Meanwhile you can set the epoch you want to train and whether you need to use `Transfer Learning` toward the problem, transfer learning is good at `single size training problem`, if you want to do that. Set `VCN` as an example, please go to `line 45` to anti-annotate this line and annotate `line 46` too. 
    * This Script is `Training all graphene data together` (4by4, 5by5, 6by6 data together) and the testing data are randomly selected from these data, each size have 1000 test data. The performance of predicting test data will show in the `console window` after you have trained this model.
    * Once your DeepGraphene algorithm's training process is finished, this model will preserve in this [folder](./Graphene_DeepLearning/Predict_h5file) and named as `total_TF_'algorithm you choice'.h5` (with transfer learning) or `total_Non-TF_'algorithm you choice'.h5`  (without transfer learning) automatically.   
* If you want to train Graphene_SVR, Go into [Script folader](./Graphene_DeepLearning/Script/Predict/Graphene_SVR), Use `Matlab2017a` to open the script [SVM_Regression](./Graphene_DeepLearning/Script/Predict/Graphene_SVR/SVM_Regression.m). From `line 8` to `line 18` you can select the size of Graphene structure you want to train, please do the annotate and anti-anotate process toward this command line. After that,run it,them you can get the model you want, the performance will show in the windows after you have trained the model.

## Result:
Setting 4by4 and 5by5 data as an example, the training process toward testing data is shown as below `training single Graphene size data's performance`
![](https://github.com/jianlin-cheng/DeepGraphene/blob/master/Image/image3.jpg)
