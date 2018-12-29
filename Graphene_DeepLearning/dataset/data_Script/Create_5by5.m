clear all
clc
%load_OrginalData_5by5
%-------------------------------------------------------------------------
Maximum_Num=2^25-1;
%Remain_Index=zeros(Maximum_Num-length(InputX),1);
%Remain_Index=zeros(Maximum_Num,1);
count_index=1;
count_test=1;
count_train=1;
%-------------------------------------------------------------------------
%for i=1:Maximum_Num
%	if any(InputX==i)
%		;
%	else
%		Remain_Index(count_index,1)=i;
%		count_index=count_index+1;
%end
Test_X_Index=randsample(Maximum_Num,1000);
%-------------------------------------------------------------------------
temp0=0;
for i=1:length(Test_X_Index)
    P=Test_X_Index(i);
    S=dec2bin(P,25);
    Ps=zeros(5,5);
    count=0;
    for ai=1:5
        for aj=1:5
            Ps(ai,aj)=str2num(S((ai-1)*5+aj));
            if (str2num(S((ai-1)*5+aj))==1)
                count=count+1;
            end
        end
    end
    if 16<=count & count<=23
        if (temp0==0)
            temp0=temp0+1;
            Test_X=Ps;
            Test_X=reshape(Test_X,[1,5,5]);
            Index_X=P;
        else
            Ps=reshape(Ps,[1,5,5]);
            Test_X=cat(1,Test_X,Ps);
            Index_X=cat(1,Index_X,P);
        end
    end
end
%-------------------------------------------------------------------------
for i=1:length(Test_X)
    tempx=Test_X(i,:,:);
    file_name_X=strcat('Test_',strcat(num2str(i),'.csv'));
    data_dir_X=strcat('D:/Working_Application/DropBox_File/Dropbox/DeepGraphene/Graphene_DeepLearning/dataset/Data/Create_Data_5by5/high_Data/',file_name_X);
    csvwrite(data_dir_X,tempx);
end

file_name='Test_5by5_high_Index.csv';
data_dir_Index=strcat('D:/Working_Application/DropBox_File/Dropbox/DeepGraphene/Graphene_DeepLearning/dataset/Data/Create_Data_5by5/',file_name);
csvwrite(data_dir_Index,Index_X);
dlmwrite(data_dir_Index,Index_X,'precision','%9.2f')

