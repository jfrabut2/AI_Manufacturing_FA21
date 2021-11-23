%% unet.m
% Names: Alisa Nguyen and Jacob Frabutt
% Date: 12/03/2021
% Description: This program trains a neural network (specifically a unet),
%   using data from generateDataset_2D.m to predict the shrinkage of
%   a set of images.
%   Reference documentation for a unet can be found at this page:
%   https://www.mathworks.com/help/vision/ref/unetlayers.html
% NOTE: This program should only be run after generateDataset_2D is run.

clear
clc

%% Data Structuring

% The data files should be stored in a dataset directory located in 
% your current directory
dataSetDir = fullfile('.', 'dataset');
imageDir = fullfile(dataSetDir, 'original'); % original images
labelDir = fullfile(dataSetDir, 'distorted'); % distorted images

% Create an imageDatastore object to store the images
imds = imageDatastore(imageDir);
 
% Determine the number of samples
numSamples = numpartitions(imds);       
 
% Determine Indices for training, testing, and validation:
% Use 70% of images for training
numTrain = round(numSamples*0.7);
trainInd = 1:numTrain;
 
% Use 15% of images for validation
numVal = round(numSamples*0.15);
valInd = (numTrain+1:numTrain+numVal);
 
% Use 15% of images for testing
numTest = round(numSamples*0.15);
testInd = (numTrain+numVal+1:numSamples);
 
% Get the images for each stage
trainingImages = imds.Files(trainInd);
valImages = imds.Files(valInd);
testImages = imds.Files(testInd);

% Create image datastores for training, validation, and testing
imdsTrain = imageDatastore(trainingImages);
imdsVal = imageDatastore(valImages);
imdsTest = imageDatastore(testImages);

% Define the classes and labels of our data
% Pixels are either white or black
classNames = ["shapes", "background"];
labelIDs = [1 0];
 
% Create a pixelLabelDatastore to store the ground truth pixel labels
pxds = pixelLabelDatastore(labelDir, classNames, labelIDs); 

% Separate the files for each stage
trainingLabels = pxds.Files(trainInd);
valLabels = pxds.Files(valInd);
testLabels = pxds.Files(testInd);

% Create pixel label datastores for training, validation, and testing
pxdsTrain = pixelLabelDatastore(trainingLabels, classNames, labelIDs);
pxdsVal = pixelLabelDatastore(valLabels, classNames, labelIDs);
pxdsTest = pixelLabelDatastore(testLabels, classNames, labelIDs);
 
% Create a datastore for training and validating the network
dsTrain = combine(imdsTrain, pxdsTrain);
dsVal = combine(imdsVal, pxdsVal);


%% Create the U-Net network

% Image size depends on size of image set in generateDataset_2D
% Note: must be a multiple of 2^EncoderDepth
imageSize = [80 80];

% This is the number of classes we defined previously
numClasses = 2; 

% Create the layers for the unet
% We are using the default EncoderDepth of 4
lgraph = unetLayers(imageSize, numClasses);

% Set training parameters
options = trainingOptions('sgdm',...
    'InitialLearnRate',0.05, ...
    'Momentum',0.9,...
    'L2Regularization',0.0001,...
    'ValidationData',dsVal,...
    'MaxEpochs',100,...
    'MiniBatchSize',16,...
    'LearnRateSchedule','piecewise',...    
    'Shuffle','every-epoch',...
    'GradientThresholdMethod','l2norm',...
    'GradientThreshold',0.05, ...
    'Plots','training-progress', ...
    'ValidationPatience', 4, ...
    'VerboseFrequency',20);

%% Train the network

net = trainNetwork(dsTrain,lgraph,options);

% To view the layers, uncomment this line: 
% analyzeNetwork(net)

% To display the u-net network, uncomment this line:
% plot(lgraph)            

%% Visualize the results

% The following code block alows you to viusalize what the model is doing
% by demonstrating the shrinkage the network predicts. To run this block of
% code, you must have a sample image and its corresponding distorted image
% in the working directory. Here '1.png' is the original image and '1d.png'
% is the corresponding distorted image.
% However, this section can all be commented out if you would like.

% load in the images
I = imread('1.png');
Id = 255 * uint8(imread('1d.png'));

% run the input through the network
[C,scores] = semanticseg(I,net);

% create an overlay and montage
%   The top left image will be the original image
%   The top right image will highlight the pixels the model removed
%   The bottom left image will highlight the discrepancy between the
%       predicted shirnkage and the ground truth shrinkage (error)
%   The bottom right image is the ground truth shrinkage
B = labeloverlay(I,C);
Accuracy = labeloverlay(Id, C);
montage({I,B,Accuracy,Id})

%% Network Statistics

% Create an label store of the ground truth test images
pxdsTruth = pixelLabelDatastore(testLabels,classNames,labelIDs);

% Run the testing images through the network
pxdsResults = semanticseg(imdsTest, net);

% Run metrics comparing ground truth images to the network's output
metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTruth);

% Display the confusion matrix
metrics.ConfusionMatrix

%% Save the network

% the command saves the trained network to the current working directory
save('net.mat', 'net');