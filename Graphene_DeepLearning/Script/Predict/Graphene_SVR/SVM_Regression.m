clear all
clc
%-------------------------------------------------------------------------
local_dir=pwd;
data_dir=strsplit(local_dir,'Script');
data_dir=strcat(char(data_dir(1)),'dataset\Data\');
%%
% 4by4 Data
Train_dir=strcat(data_dir,'Train_Data_4by4/');
Test_dir=strcat(data_dir,'Test_Data_4by4/');
%-------------------------------------------------------------------------
% 5by5 Data
%Train_dir=strcat(data_dir,'Train_Data_5by5/');
%Test_dir=strcat(data_dir,'Test_Data_5by5/');
%-------------------------------------------------------------------------
% 6by6 Data
%Train_dir=strcat(data_dir,'Train_Data_6by6/');
%Test_dir=strcat(data_dir,'Test_Data_6by6/');

%%
Train_X_Dir=strcat(Train_dir,'InputX/');
Train_Y_Dir=strcat(Train_dir,'InputY/');
Test_X_Dir=strcat(Test_dir,'InputX/');
Test_Y_Dir=strcat(Test_dir,'InputY/');
%new_X_Dir=strcat(new_dir,'InputX/');
Train_X_Info=dir(Train_X_Dir);
Train_X_filename=char(strcat(Train_X_Dir,{Train_X_Info.name}));
Train_Y_Info=dir(Train_Y_Dir);
Train_Y_filename=char(strcat(Train_Y_Dir,{Train_Y_Info.name}));
Test_X_Info=dir(Test_X_Dir);
Test_X_filename=char(strcat(Test_X_Dir,{Test_X_Info.name}));
Test_Y_Info=dir(Test_Y_Dir);
Test_Y_filename=char(strcat(Test_Y_Dir,{Test_Y_Info.name}));
%new_X_Info=dir(new_X_Dir);
%new_X_filename=char(strcat(new_X_Dir,{new_X_Info.name}));
%-------------------------------------------------------------------------
Length_Train=size(Train_X_filename);
Length_Test=size(Test_X_filename);
%Length_new=size(new_X_filename);

%%
for i=3:Length_Train(1)
    if (i==3)
        Train_X=csvread(Train_X_filename(i,:));
        Train_Y=csvread(Train_Y_filename(i,:));
    else 
        tempx=csvread(Train_X_filename(i,:));
        tempy=csvread(Train_Y_filename(i,:));
        Train_X=[Train_X;tempx];
        Train_Y=[Train_Y;tempy];
    end
end

for i=3:Length_Test(1)
    if (i==3)
        Test_X=csvread(Test_X_filename(i,:));
        Test_Y=csvread(Test_Y_filename(i,:));
    else 
        tempx=csvread(Test_X_filename(i,:));
        tempy=csvread(Test_Y_filename(i,:));
        Test_X=[Test_X;tempx];
        Test_Y=[Test_Y;tempy];
    end
end

%for i=3:Length_new(1)
%    if (i==3)
%        new_X=csvread(new_X_filename(i,:));
%    else 
%        tempx=csvread(new_X_filename(i,:));
%        new_X=[new_X;tempx];
%    end
%end

Test_Y=Test_Y.*4.6;
%-------------------------------------------------------------------------
fprintf('**********************************************************\n');
fprintf('    Finsh Load the data! \n');
fprintf('**********************************************************\n');
clearvars local_dir data_dir Test_dir Train_dir Train_X_Dir Train_Y_Dir Test_X_Dir Test_Y_Dir new_X_dir
clearvars Train_X_Info Train_Y_Info Test_X_Info Test_Y_Info new_X_Info
clearvars Train_X_filename Train_Y_filename Test_X_filename Test_Y_filename new_X_filename
clearvars tempx tempy Length_Test Length_train Length_new

%%
%--------------------------------------------------------------------------
% gpuDeviceCount           %print how many GPU are installed in this node
% gpu=gpuDevice                % print the information of the GPU
%--------------------------------------------------------------------------
ModelSVM=fitrsvm(Train_X,Train_Y,'standardize',true,'KernelFunction','polynomial');
                                        % Training the SVM model
fprintf('**********************************************************\n');
fprintf(' The Converge condition of MOdel (if value=1-> converged)\n' );
ModelSVM.ConvergenceInfo.Converged	% Evaluate if the SVM is Converge or not
fprintf('**********************************************************\n');

MSE_Loss_value=resubLoss(ModelSVM);
fprintf('**********************************************************\n');
fprintf(' The MSE loss function of this model is %f \n', MSE_Loss_value);
fprintf('**********************************************************\n\n');

%%
Predict_Y=predict(ModelSVM,Test_X);
Predict_Y=Predict_Y.*4.6;
error_percent=0;
ava_error=0;
abs_error=0;
for i=1:length(Predict_Y)
    ava_error=ava_error+abs(Predict_Y(i,:)-Test_Y(i,:))/Predict_Y(i,:);
    abs_error=abs_error+abs(Predict_Y(i,:)-Test_Y(i,:));
    error_percent=error_percent+abs(Predict_Y(i,:)-Test_Y(i,:))/Test_Y(i,:);
end
ava_error=ava_error/1000;
abs_error=abs_error/1000;
error_percent=error_percent*100/1000;
fprintf('**********************************************************\n');
fprintf(' The Relative error is :  %f \n', ava_error);
fprintf(' The Absolute error is :  %f \n', abs_error);
fprintf(' The Avvuracy error is :  %f \n', error_percent);
fprintf('**********************************************************\n');
%--------------------------------------------------------------------------
Predict_new=predict(ModelSVM,new_X)
Predict_new=Predict_new.*4.6;



