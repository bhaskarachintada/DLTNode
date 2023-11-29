% Script for preparing the data for the training: Loads original and
% TNode-processed OCT volumes and randomly picks 100 training pairs.

%% Training and Testing cardinality;
nTrain = 100; %Number of BscanStacks for training
seed = rng('default'); % specify the seed for random number generator

%% Working folders
% Matlab functions
addpath(genpath('../../matlab')); % Functions folder
% tomogram Intensity data path name
datasetName = 'FingerSkin';
dataPath = fullfile('../../Data/',[datasetName,'TomInt.mat']);
% TNode processed tomogram Intensity data path name
TNodeDataPath = fullfile('../../Data/',[datasetName,'TomIntTNode.mat']);
% output folder to save the training pairs 
outputFolder = fullfile('../../Output/',datasetName);
[~, ~, ~] = mkdir(outputFolder);
filenamePreffix= datasetName;

%% params used in TNode processing
hSearch = 8; % Search Window used to generate ground truth TNode dat

%% load the raw and TNode-processed OCT volumes
% load tomogram intensity volume
load(dataPath);

% load TNode processed tomogram intensity volume 
load(TNodeDataPath);

%% Save Bscan Stacks for the training
% concatenate TomIntSum and tomIntSumTNode
imageData = cat(2,10*log10(tomIntSum),10*log10(tomIntSumTNode));

for idx = hSearch+1:size(imageData,3)-hSearch-1
    hSearchRange = idx-hSearch:idx+hSearch;
    stack= imageData(:,1:end/2,hSearchRange);
    % stack the Raw Bscans of size: [height x width x hSearchRange] and
    % TNode despeckled Bscan of size: [height x width x 1] in 3rd dimension
    % to make it [height x width x hsearchRange+1]
    bscanStack = cat(3, stack, imageData(:,end/2+1:end,idx));
    % saving them as .mat files for now--> one could save them as HDF5 file
    % to reduce the data reading overheads for the network.
    fileName = fullfile(outputFolder, [filenamePreffix,'_',num2str(idx),'.mat']); % filename
    save(fileName,'bscanStack') %Save
end

%% Split the bscanStack files into training and testing randomly
[~,~,~]= mkdir(outputFolder,'train'); % create train folder
numberofBscanStacks = length(hSearch+1:size(imageData,3)-hSearch-1);
rng(seed) % helps you use same speed while debugging
idx = randperm(numberofBscanStacks); % creates the random indices of files
% split the data into train, validation and test folders
% Randomly select #nTRain BscanStacks for the training
trainIdx = idx(1:nTrain); % idx(1:ceil(n*0.8)); 
% Remaining BscanStacks for testing if needed
testIdx = idx(nTrain:numberofBscanStacks);

% Destination and Source Folders to copy and paste the BscanStacks
destFolder = fullfile(outputFolder,'train');
sourceFolder = outputFolder;

fileList = dir(fullfile(sourceFolder,'*.mat')); %list all the files with given exension

for k = 1:length(trainIdx)
  source = fullfile(sourceFolder, fileList(trainIdx(k)).name);
  copyfile(source, destFolder);
end

%%


