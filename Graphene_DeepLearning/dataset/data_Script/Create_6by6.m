%clear all
clc
%load_OrginalData_6by6
%-------------------------------------------------------------------------
%Maximum_Num=2^36-1;
%Test_X=zeros(1000,6,6);
%Remain_Index=zeros(Maximum_Num-length(InputX),1);
Create_X=zeros(length(InputX),6,6);
count_index=1;
count_test=1;
count_train=1;
%-------------------------------------------------------------------------
%for i=1:Maximum_Num
%	if any(InputX==i)
%		;
%	else
%		Remain_Index(count_index,1)=i;
%	end
%end
%Test_X_Index=randsample(Remain_Index,1000);
%-------------------------------------------------------------------------
for i=1:length(Create_X)
    P=InputX(i);
    S=dec2bin(P,36);
    Ps=zeros(6,6);
    for ai=1:6
        for aj=1:6
            Ps(ai,aj)=str2num(S((ai-1)*6+aj));
        end
    end
    Create_X(i,:,:)=Ps;  
end
%-------------------------------------------------------------------------
for i=1:length(Create_X)
    tempx=Create_X(i,:,:);
    file_name_X=strcat('Test_',strcat(num2str(i),'.csv'));
    data_dir_X=strcat('./Create_Data_6by6/Data/',file_name_X);;
    csvwrite(data_dir_X,tempx);
end

%file_name='Test_6by6_Index.csv';
%data_dir_Index=strcat('./Create_Data_6by6/',file_name);
%csvwrite(data_dir_Index,Test_X_Index);
%dlmwrite(data_dir_Index,Test_X_Index,'precision','%9.2f')