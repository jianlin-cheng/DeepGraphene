clear all
clc
load_OrginalData
%-------------------------------------------------------------------------
Maximum_Num=2^25-1;
Test_X=zeros(1000,5,5);
Remain_Index=zeros(Maximum_Num-length(InputX),1);
count_index=1;
count_test=1;
count_train=1;
%-------------------------------------------------------------------------
for i=1:Maximum_Num
	if any(InputX==i)
		;
	else
		Remain_Index(count_index,1)=i;
		count_index=count_index+1;
	end
end
Test_X_Index=randsample(Remain_Index,1000);
%-------------------------------------------------------------------------
for i=1:length(Test_X_Index)
    P=Test_X_Index(i);
    S=dec2bin(P,25);
    Ps=zeros(5,5);
    for ai=1:5
        for aj=1:5
            Ps(ai,aj)=str2num(S((ai-1)*5+aj));
        end
    end
    Test_X(i,:,:)=Ps;  
end
%-------------------------------------------------------------------------
for i=1:length(Test_X)
    tempx=Test_X(i,:,:);
    file_name_X=strcat('Test_',strcat(num2str(i),'.csv'));
    data_dir_X=strcat('./Create_Data_5by5/Data/',file_name_X);;
    csvwrite(data_dir_X,tempx);
end

file_name='Test_5by5_Index.csv';
data_dir_Index=strcat('./Create_Data_5by5/',file_name);
%csvwrite(data_dir_Index,Test_X_Index);
dlmwrite(data_dir_Index,Test_X_Index,'precision','%9.2f')

