%% Initialize variables.
local_dir=pwd;
file_dir=strsplit(local_dir,'data_Script');
filename=strcat(char(file_dir(1)),'Original_data\data\data55_1027.csv');
delimiter = ',';

%% Format for each line of text:
%   column1: double (%f)
% For more information, see the TEXTSCAN documentation.
formatSpecX = '%f%*s%[^\n\r]';
formatSpecY = '%*s%f%[^\n\r]';
%% Open the text file.
fileIDX = fopen(filename,'r');
fileIDY = fopen(filename,'r');
%% Read columns of data according to the format.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
XdataArray = textscan(fileIDX, formatSpecX, 'Delimiter', delimiter, 'TextType', 'string',  'ReturnOnError', false);
YdataArray = textscan(fileIDY, formatSpecY, 'Delimiter', delimiter, 'TextType', 'string',  'ReturnOnError', false);
%% Close the text file.
fclose(fileIDX);
fclose(fileIDY);
%% Post processing for unimportable data.
% No unimportable data rules were applied during the import, so no post
% processing code is included. To generate code which works for
% unimportable data, select unimportable cells in a file and regenerate the
% script.

%% Create output variable
InputX = [XdataArray{1:end-1}];
InputY = [YdataArray{1:end-1}];
%% Clear temporary variables
clearvars filename delimiter formatSpec fileID dataArray ans;