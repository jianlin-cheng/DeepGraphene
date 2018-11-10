%% Import data from text file.
% Script for importing data from the following text file:
%
%    D:\Working_Application\DropBox_File\Dropbox\Graphene_DeepLearning\dataset\Original_data\dataH.csv
%
% To extend the code to different selected data or a different text file,
% generate a function instead of a script.

% Auto-generated by MATLAB on 2018/02/28 17:00:36

%% Initialize variables.
filename = 'D:\Working_Application\DropBox_File\Dropbox\Graphene_DeepLearning\dataset\Original_data\data66_0926.csv';
%filename='.\Original_data\data_store_0304.csv'
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