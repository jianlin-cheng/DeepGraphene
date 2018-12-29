clear all
local_dir=pwd
file_dir=strsplit(local_dir,'data_Script');
load_OrginalData_6by6
%-------------------------------------------------------------------------
fprintf('**********************************************************\n');
fprintf(' Now we Crate Original 6by6 Graphene Data \n');
fprintf('**********************************************************\n\n');
Original_X=zeros(length(InputX),6,6);
Test_X=zeros(1000,6,6);
Test_Y=zeros(1000,1);
Train_X=zeros((length(InputX)-1000),6,6);
Train_Y=zeros((length(InputX)-1000),1);
count_test=1;
count_train=1;
random_index=randsample(length(InputX),1000);
%-------------------------------------------------------------------------
for i=1:length(InputX)
    P=InputX(i);
    S=dec2bin(P,36);
    Ps=zeros(6,6);
    for ai=1:6
        for aj=1:6
            Ps(ai,aj)=str2num(S((ai-1)*6+aj));
        end
    end
    Original_X(i,:,:)=Ps;  
end

for i=1:length(InputX)
    temp=InputY(i,:)/4.6;
	if any(random_index==i)
		Test_X(count_test,:,:)=Original_X(i,:,:);
		Test_Y(count_test,:)=temp;
		Test_data_index(count_test,:)=InputX(i,:);
		count_test=count_test+1;
	else
		Train_X(count_train,:,:)=Original_X(i,:,:);
		Train_Y(count_train,:)=temp;
		count_train=count_train+1;	
    end
end
%-------------------------------------------------------------------------
for i=1:length(Train_X)
    tempx=Train_X(i,:,:);
    tempy=Train_Y(i,:);
    file_name_X=strcat('InputX\InputX_',strcat(num2str(i),'.csv'));
    file_name_Y=strcat('InputY\InputY_',strcat(num2str(i),'.csv'));
    filename=strcat(char(file_dir(1)),'Data\Train_Data_6by6\');
    data_dir_X=strcat(filename,file_name_X);
    data_dir_Y=strcat(filename,file_name_Y);
    csvwrite(data_dir_X,tempx);
    csvwrite(data_dir_Y,tempy);
end

for i=1:length(Test_X)
    tempx=Test_X(i,:,:);
    tempy=Test_Y(i,:);
    file_name_X=strcat('InputX\InputX_',strcat(num2str(i),'.csv'));
    file_name_Y=strcat('InputY\InputY_',strcat(num2str(i),'.csv'));
    filename=strcat(char(file_dir(1)),'Data\Test_Data_6by6\');
    data_dir_X=strcat(filename,file_name_X);
    data_dir_Y=strcat(filename,file_name_Y);
    csvwrite(data_dir_X,tempx);
    csvwrite(data_dir_Y,tempy);
end

%Realfile_name='RealValue_Y.csv'
%file_name='Test_Index_6by6.csv';
%data_dir_Index=strcat('./Test_Data_5by5/',file_name);
%Realdata_dir_Index=strcat('C:/Users/Herman Wu/Desktop/Predict/',Realfile_name);
%csvwrite(data_dir_Index,Test_data_index);
%csvwrite(Realdata_dir_Index,Test_Y);
fprintf('**********************************************************\n');
fprintf(' 6*6 Data Finished!! \n ');
fprintf('**********************************************************\n\n');