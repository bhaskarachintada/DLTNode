% Matlab code to compute quality metrics and to create figures for the
% manuscript
addpath(genpath('../../matlab')); % Functions folder
datasetName = 'FingerSkin';
% output folder that contains the data pairs 
testdataPathRoot = fullfile('../../Output/',datasetName);  
predictiondataPathRoot = fullfile('../../Output/Inference/',datasetName);
intCMap = gray(256);
logLimPrediction = [70 120]; % Log limit used during inference 
showFigs = 0;
%%
predictionMatFilesList= dir(fullfile(predictiondataPathRoot,'*.mat'));
MatFilesList= dir(fullfile(testdataPathRoot,'*.mat'));
thisStack = load(fullfile(testdataPathRoot, MatFilesList(5).name));
try 
  speckleStack = thisStack.bscanStack;
catch
  speckleStack = thisStack.image_stack;
end
tomWidth = size(speckleStack,2); 
tomDepth = size(speckleStack,1);
zeropadHalfYwidth = round(0.5*(1024-size(speckleStack,2)));
zeropadZdepth = round(0.5*(1024-size(speckleStack,1)));
%%
filenameString = predictionMatFilesList(1).name;
saveFileName = split(filenameString,'.mat');
saveFileName = saveFileName{1,1};
saveFileName = saveFileName(1:end-3);
sliceIdx = zeros(length(predictionMatFilesList),1);
for i=1:length(predictionMatFilesList)
  
  filenameString = predictionMatFilesList(i).name;
  splitFilenameString = split(filenameString,"_");
  splitFilenameString = splitFilenameString{end,1};
  sliceIdx(i,1) = str2num(splitFilenameString(1:end-4));
end
 sliceIdx = sliceIdx-min(sliceIdx)+1;
%%
speckleStack = single(zeros(length(predictionMatFilesList),1024,1024*3));
for i=1:length(predictionMatFilesList)
  thisStack = load(fullfile(predictiondataPathRoot, predictionMatFilesList(i).name));
  speckleStack(sliceIdx(i),:,:) = thisStack.outputStack;
  if showFigs
    imagesc(squeeze(speckleStack(sliceIdx(i),:,:))); colormap(gray(2^16-1));
    draw now
  end
end

thisImageYIdx = zeropadHalfYwidth+1:1024-zeropadHalfYwidth;
imageIdx = [thisImageYIdx 1024+thisImageYIdx 2048+thisImageYIdx];
speckleStack = uint16(speckleStack(:,zeropadZdepth+1:end-zeropadZdepth,imageIdx));
% convert the data back to log scale. 
tomLogStack = single(speckleStack)/65535 * diff(logLimPrediction) + logLimPrediction(1);
