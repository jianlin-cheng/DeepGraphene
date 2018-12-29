clear all
clc
load_OrginalData
%-------------------------------------------------------------------------
Maximum_Num= 65520;
Test_X=zeros(1000,4,4);
Remain_Index=zeros(Maximum_Num-length(InputX),1);
count_index=1;
count_test=1;
count_train=1;
%-------------------------------------------------------------------------
for i=0:Maximum_Num
	if any(InputX==i)
	else
		Remain_Index(count_index,1)=i;
		count_index=count_index+1;
	end
end
Test_X_Index=randsample(Remain_Index,1000);
%-------------------------------------------------------------------------
for i=1:length(Test_X_Index)
    P=Test_X_Index(i);
    S=dec2bin(P,16);
    Ps=zeros(4,4);
    for ai=1:4
        for aj=1:4
            Ps(ai,aj)=str2num(S((ai-1)*4+aj));
        end
    end
    Test_X(i,:,:)=Ps;  
end
%-------------------------------------------------------------------------
for i=1:length(Test_X)
    tempx=Test_X(i,:,:);
    file_name_X=strcat('Test_',strcat(num2str(i),'.csv'));
    data_dir_X=strcat('./Create_Data_4by4/Data/',file_name_X);;
    csvwrite(data_dir_X,tempx);
end

file_name='Test_4by4_Index.csv';
data_dir_Index=strcat('./Create_Data_4by4/',file_name);
csvwrite(data_dir_Index,Test_X_Index);
