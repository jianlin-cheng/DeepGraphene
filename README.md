# DeepGraphene
 Paper: `Deep Learning Bandgaps of Topologically Doped Graphene` Yuan Dong, Chuhan Wu et.al <br/>
         Url: https://arxiv.org/abs/1809.10860  < Under reviewed Form >
## Introduction:
*   This Repo contain the source code of Paper `Deep Learning Bandgaps of topologically Doped Graphene` , it contains all algorithms which we use to predict graphene supercells' bandgap values (Graphene-SVR, VCN, RCN, CCN). Meanwhile it contains the latest data of graphene supercell ( 4by4: 13018, 5by5: 79647, 6by6: 6382).
<br/><br/>
***Repo Structure*** 
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
* [Predict_h5file :](./Graphene_DeepLearning/)
    * GAN_h5file.zip
    * h5_file.zip
* [Script :](./Graphene_DeepLearning/Script)  (This folder contain all scripts we use)
    * Generate
    * Predict 
        * [DeepGraphene ](./Graphene_DeepLearning/Script/Predict/DeepGraphene)  (This folder contains Deep Neural Network algorithms' script: VCN, RCN, CCN)
        * [Graphene_SVR ](./Graphene_DeepLearning/Script/Predict/Graphene_SVR) (This folder contains traditional machine learning algorithm' script: Graphene_SVR)
