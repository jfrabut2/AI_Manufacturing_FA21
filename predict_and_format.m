%% predict_and_format.m
% Names: Alisa Nguyen and Jacob Frabutt
% Date: 12/03/2021
% Description: This program uses a pretrained neural network along with new
%   data so that the output of the forward unet can be properly formatted. 
%   This can be used simply to use the network to predict new data, or to
%   generate data to use to predict the inverse network.
% NOTE: This program should only be run after unet.m is run and it saves
%   a fully trained network. Additionally, you should delete the data you
%   trained the model with, and generate all new data before running this
%   program.

clear
clc

%% Data Structuring

% load in the trained net from the working directory
load net

% The data files should be stored in a dataset directory located in 
% your current directory
dataSetDir = fullfile('.', 'dataset');
imageDir = fullfile(dataSetDir, 'original'); % original images
labelDir = fullfile(dataSetDir, 'distorted'); % distorted images

% Create an imageDatastore object to store the images
imds = imageDatastore(imageDir);
 
% Determine the number of samples
numSamples = numpartitions(imds);

% Select all the generated samples
testImages = imds.Files(1:numSamples);

% Create an image datastore for the samples
imdsTest = imageDatastore(testImages);

%% Using the network

% create a directory for the output files
mkdir('NetResults')

% run all the samples through the network and write to the file we created
% NOTE: you need to modify the path to the folder so that it is correct
%       on whatever device you are using.
pxdsResults = semanticseg(imdsTest, net, "WriteLocation", '/Users/jacob/Desktop/Research/NetResults/'); %MAC

%% Data formatting 

% The files outputed by the semanticseg function above have values of 
% 2 for the background and 1 for the shape. However, the the pixels should
% be 0 for the background and 255 for the shape. The following code makes
% new versions of the output files with these correct pixel values.

% create a driectory for the correctly formatted output files
mkdir('FormattedNetResults');

% Retrieve all the files from the output folder
filePattern = fullfile('NetResults', '*.png');
theFiles = dir(filePattern);
theFiles = theFiles(~[theFiles.isdir]);
resultSetDir = fullfile('Users','jacob','Desktop','Research', 'NetResults'); 

% iterate through all the files
for k = 1:length(theFiles) 
    
    % read in the kth file
    baseFileName = theFiles(k).name; 
    fullFileName = fullfile(theFiles(k).folder, baseFileName); 
    img = imread(fullFileName);
    
    % scale the 2s and 1s down to 1s and 0s
    imgLogical = mat2gray(img);
    
    % convert the logicals to unsigned ints and scale the 1s to 255s
    uint8Image = uint8(255*imgLogical);
    
    % invert the image so that the background values are 0 and the shape 
    % values are 255
    invertImage = 255-uint8Image;
    
    % format the name of the file to include preceding 0s. This ensures
    % that the order of the files does not get mixed up. Works for up to
    % 99,999 samples. Change the 5 for more samples.
    strPadded = sprintf('%05d', k);
    
    % write our correctly formated image to the new folder
    imwrite(invertImage,['FormattedNetResults/' strPadded 'r.png']);
end

% Delete the unformatted net results folder 
system('rm -rf NetResults');