function options = GetDefaultOptionsTNode(tom)

% Create default options to run the tomographic non-local means despeckling TNOde to OCT.
% Those parameters seems to work fine in almost any image, if user decides
% to manually change them, the algorithm will overwrite the choosed ones

h0 = 8e-3; % base despeckling parameter
h1 = 35e-3; % SNR-dependent despeckling parameter
% Similarity post-processing parms
simPostProcessing = 'none';
hSimilarity = [3, 3, 3]; % Using speckle size instead
hSearch = [6, 8, 2]; % Search window half size [Z,X,Y], total size 2*hSearch+1
verbosity = 4; % Show additional information in the algorithm
showFigs = 1;
imRange = [10*log10(min(tom(:))), 10*log10(max(tom(:)))]; % Images range in dB
nAverageBScans = 1;
noiseFloorDb = 0;
figFunc = 'axis image';

% Bscans to filter in volume
iniSlice = [1, 1, 1]; % Initial slice to filter in ZXY
finSlice = size(tom); % Final slice to filter in ZXY
direction = 'ZX'; % Filtering plane 'ZX' or 'ZY'

% Edge behavior
edgePadding = 'replicate'; % Can be 0, 'replicate', 'symmetric' or 'circular'

% Maximum aline block size
blockSize = [80, 80, 48];

% Set central pixel weigth to one
normalizeSelfSimilarity = true;

% Kernels
hSimiKernel = 'unitary';
hSearchKernel = 'unitary';

% Whether to use the GPU or not
useGPU = false;

% Remove mean intensity while computing
hSimNorm = false; 

% Print output into a txt file and its path
fileStdout = false;
fileStdoutPath = '.';

% Pack them into a struct and return
options = struct('hSearch', hSearch, 'h0', h0, 'h1', h1, ...
  'verbosity', verbosity, 'iniSlice', iniSlice, 'finSlice', finSlice, ...
  'hSimilarity', hSimilarity, 'hSimNorm', hSimNorm, ...
  'direction', direction, 'imRange', imRange,...
  'edgePadding', edgePadding, 'blockSize', blockSize,...
  'nAverageBScans', nAverageBScans,'noiseFloorDb', noiseFloorDb,...
  'figFunc', figFunc, 'normalizeSelfSimilarity', normalizeSelfSimilarity,...
  'simPostProcessing', simPostProcessing, 'useGPU', useGPU, 'showFigs', showFigs,...
  'fileStdout', fileStdout, 'fileStdoutPath', fileStdoutPath);
end
