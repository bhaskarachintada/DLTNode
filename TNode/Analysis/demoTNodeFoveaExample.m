% This is an example on how to use TNode with OCT intensity tomograms
% First, add to the MATLAB folder the path to main functions
addpath(genpath('../matlab'));
dataPath = '../Data';

%% Read data
% Here we read the data file with the tomogram. This tomogram has been
% saved in linear intensity using single precision, (NO logarithmic
% transformation has been performed)
load(fullfile(dataPath, 'tomFoveaDemo.mat'))

% The only loaded variable is tomROI. The tomogram axes are assumed to be 
% organized in the following order: (Z,X,Y, W) with Z the depth dimension,
% X the fast scanning axis, Y the slow scanning axis, and W an additional
% arbitrary dimension
% This tomogram consist of a small ROI of the fovea having 128 depth 
% samples and 256 Alines per Bscan with 256 Bscans and 2 orthogonal
% illumination polarization states
[nZ, nX, nY, ~, nToms] = size(tomROI);

%% Let's apply TNode to the following en-face plane +- 7:
logLim = [65 100];
thisZ = 22;
% Tomogram 1
figure(1), subplot(221), imagesc(10 * log10(squeeze(tomROI(thisZ, :, :, :, 1)))', logLim),
axis image, colormap(gray(256)), xticks([]), yticks([])
xlabel('X'), ylabel('Y'), title('Original en-face, W = 1')
% Tomogram 2
figure(1), subplot(222), imagesc(10 * log10(squeeze(tomROI(thisZ, :, :, :, 2)))', logLim),
axis image, colormap(gray(256)), xticks([]), yticks([])
xlabel('X'), ylabel('Y'), title('Original en-face, W = 2')

%% Define paremeter for TNode
% Search window half-size [Z,X,Y,W]; total size 2*hSearch+1.
hSearch = [8, 8, 8, 0];
% Similarity window half-size [Z,X,Y,W], total size 2*hSimi+1
hSimi = [3, 3, 3, 0];
% Apodization for the similarity window
hSimiKernel = 'unitary';
% Apodization for the search window
hSearchKernel = 'unitary';
% Filtering parameters h0 and h1. TNode produces an output tomogram per
% [h0, h1] pair. Here we have two pairs.
h0 = [110 70] * 1e-3; % Base speckle reduction
h1 = [0 30] * 1e-3; % SNR-dependent speckle reduction
% Post-processing options for the similarity criterion
% Prunning between 0 and 100. Each prunning value is applied to each pair
% [h0, h1]. So the total number of output tomograms is Number of [h0, h1]
% pairs times number of prunning values
pruningPercentileVec = [50 30];
pruningPercentileVecStr = num2str(pruningPercentileVec, '%d,');
simPostProcessing = strcat('pruning=', pruningPercentileVecStr(1:end - 1));
% Show additional information in the algorithm?
verbosity = 4; % Verbosity level (1 up to 4)
showFigs = false; % Show figures of first slice in each subVolume?
% If so, request the range for plotting intermediate results
imRange = [70 120]; % Images range in dB
figFunc = 'truesize';
% Noise floor [dB]
noiseFloorDb = 72;
% Maximum simultaneous processed voxels in each dimension [Z,X,Y]
% Adjust according to GPU or RAM size
blockSize = [7, 32, 64];
% Central pixel behavior. True: sets self-similarty to 1
% False sets self-similarity to the maximum weight
normalizeSelfSimilarity = false;
% Re-scale weights
rescaleWeights = true;
% ROI to despeckle; the despeckling is perform to each voxel within this
% range. In this case, TNode will be applied to sub-volume with 7 depth
% samples z = {thisZ - 3, thisZ + 3}
iniSlice = [thisZ - hSimi(1), 1, 1, 1, 1]; % Initial slice to filter in ZXY
finSlice = [thisZ + hSimi(1), nX, nY, 1, nToms]; % Final slice to filter in ZXY
% Despeckling direction: It does not change output results but different
% directions can be more convinient to different ROI. In this case, XY is
% more convinent because we have few z-slices.
direction = 'XY'; % Options are 'ZX', 'ZY', 'enface' or 'XY'
% Similarity window normalization to compensate for attenuation of light
% This may require prior correction of motion artifacts or a flatten
% tomogram
hSimNorm = false; 
% Use GPU device?
useGPU = true;
% Index of GPU divice to use
gpuIdx = 1;
% Edge behavior
edgePadding = 'symmetric';
% Set to true to output information during processing into .txt file
fileStdout = false;
fileStdoutPath = '.'; % Path to file
% Convert options and variables to struct
options = struct('hSearch', hSearch, 'hSimilarity', hSimi, ...
  'verbosity', verbosity, 'showFigs', showFigs, 'iniSlice', iniSlice,...
  'noiseFloorDb', noiseFloorDb, 'finSlice', finSlice, 'direction', direction,...
  'imRange', imRange, 'h0', h0, 'h1', h1, 'edgePadding', edgePadding,...
  'blockSize', blockSize, 'normalizeSelfSimilarity', normalizeSelfSimilarity,...
  'hSimiKernel', hSimiKernel, 'hSearchKernel', hSearchKernel,...
  'rescaleWeights', rescaleWeights, 'simPostProcessing', simPostProcessing,...
  'useGPU', useGPU, 'hSimNorm', hSimNorm, 'figFunc', figFunc,...
  'fileStdout', fileStdout, 'fileStdoutPath', fileStdoutPath, 'gpuIdx', gpuIdx);

%%% Run TNode 3D
tomTNode = PerformTNode(tomROI, options);
% Get ROIs according to ini and fin slices. Everything outside is zero
zROIinTom = iniSlice(1): finSlice(1);
xROIinTom = iniSlice(2): finSlice(2);
yROIinTom = iniSlice(3): finSlice(3);

% The output tomogram have the same dimension of the input tomogram, but
% the 6th dimension is equal to the number of [h0, h1] pairs times the
% number of purnning values. In this examples, 2 x 2 = 4;

%%% Alternative: Run TNode 3D in two GPUs in parallel
% In this example, we have 2 tomograms in the 5-th dimension which we want
% to process independently. An alternative is to process each tomogram
% indepedently and simultaneously in two different GPUs, if available.
parallelDim = 5; % Dimension along which we want to process independently
options.gpuIdx = [1 2];
% UNCOMMENT FOLLOWING LINE TO RUN TNode in GPUs in parallel
tomTNode =  RunParallelTNode(tomROI, options, parallelDim);

%% Iterate to show output tomograms
% there are four combinations of [h0, h1] and prunning
nH0 = numel(h0);
nPrunings = numel(pruningPercentileVec);
iter = 1;
for thisPruningIter = 1:nPrunings
  for thisH0Iter = 1:nH0
    % Get ROI just after ini and fin slice have been applied and real ROI
    thisTomTNode = tomTNode(zROIinTom, xROIinTom,...
      yROIinTom, :, :, nH0 * (thisPruningIter - 1) + thisH0Iter);
    % Tomogram 1
    figure(2), subplot(2,2,iter), imagesc(10 * log10(squeeze(thisTomTNode(4, :, :, 1)))', logLim),
    axis image, colormap(gray(256)), xticks([]), yticks([])
    xlabel('X'), ylabel('Y'), title(sprintf('h0 = %.2f, h1 = %.2f \n %d%% Prunning',...
      h0(thisH0Iter), h1(thisH0Iter), pruningPercentileVec(thisPruningIter)))
    % Tomogram 2
    figure(3), subplot(2,2,iter), imagesc(10 * log10(squeeze(thisTomTNode(4, :, :, 2)))', logLim),
    axis image, colormap(gray(256)), xticks([]), yticks([])
    xlabel('X'), ylabel('Y'), title(sprintf('h0 = %.2f, h1 = %.2f \n %d%% Prunning',...
      h0(thisH0Iter), h1(thisH0Iter), pruningPercentileVec(thisPruningIter)))
    iter = iter + 1;
  end
end