function [tomDespeckled, varargout] = PerformTNode(tom, options)
  
  % PerformTNode Takes a tomogram and the options set before
  % applying the Tomographic non-local despeckling TNode.
  
  % Inputs:
  % tom: the tomogram to despeckle it must be organized in (Z,X,Y), where Z is the axial
  % direction, X the fast scanning axis and Y the slow scanning axis. Note that it uses
  % intensity speckle statistics, so the tomogram must be in linear scale.
  %
  % Options: A set of options for the algorithm, those must be a struct:
  %   -hSearch: Search window half-size in (Z,X,Y), Y=0 correspond to TNode2D,
  %             default is (5,7,2).
  %   -hSimilarity: Similarity window half-size in (Z,X,Y), default is (3,3,3).
  %   -h0: Base speckle reduction parameter, default 0
  %   -h1: SNR-dependent parameter, default 38
  %   -verbosity: Show additional information in the algorithm, if set to 1 it will show
  %     the current B-scan, options vary form 0 to 4.
  %   -showFigs: Create figure while algorithm is running, default true.
  %   -imRange: Range for plotting intermediate results
  %   -nAverageBScans: Number of previously averaged B-scans, default 1.
  %   -noiseFloorDb: Noise floor in dB, default 0.
  %   -blockSize: Processed subVolume maximum size, default [32,32,32].
  %   -normalizeSelfSimilarity: Central pixel behavier, true assigns 1 as the weight of the central
  %   patch, false assigns the maximum weight. Default true.
  %   -iniSlice: Initial slice to despeckle in ZXY. Default [1,1,1].
  %   -finSlice: Final slice to despeckle in ZXY. Default [X,Y,Z] size of tomogram.
  %   -direction: 'ZX' or 'ZY' a specific direction to use the windows, default 'ZX'.
  %   -edgePadding: Edges behavior, default 'circular'.
  %   -useGPU: Wheter to use a MATLAB compatible GPU device, default false.
  %   -hSimNorm: Remove mean intensity while computing, default true. This
  %     option requires a motion free or flatten tomogram, if you see
  %     unusual blurring in the final tomogram, turn this option off.
  %     Default true.
  %   -fileStdout: Write output into a txt file, default false.
  %   -fileStdoutPath: If writting txt file, define its path. Default '.'
  
  %
  % Outputs:
  %   tomDespeckled: The tomogram after despeckling.
  %   varargout: If requested, returns a matrix of the size of subVolume with
  %     the amount of significant neighbors.
  %
  % See also ApplyTNodeToSubBscan
  
  % This script and its functions follow the coding style that can be
  % summarized in:
  % * Variables have lower camel case
  % * Functions upper camel case
  % * Constants all upper case
  % * Spaces around operators
  %
  % Authors:  Carlos Cuartas-Vélez {1*}, Sebastián Ruiz-Lopera, René
  % Restrepo {1} and Néstor Uribe-Patarroyo {2}.
  % CCV - SRL - RR:
  %	 1. Applied Optics Group, Universidad EAFIT, Carrera 49 # 7 Sur-50,
  %     Medellín, Colombia.
  %
  % NUP:
  %  2. Wellman Center for Photomedicine, Harvard Medical School, Massachusetts
  % General Hospital, 40 Blossom Street, Boston, MA, USA.
  %
  %	* <ccuarta1@eafit.edu.co>
  
  % MGH-HMS-EAFIT OCT Postprocessing Project
  %
  % Changelog:
  %
  % V1.0 (2018-03-28): Initial version released
  % V2.0 (2019-03-06): Optimized version
  % V3.0 (2022-11-14): Version with pruning, multiple h0/pruning options,
  % scaling
  %
  % Copyright Carlos Cuartas-Vélez, Sebastián Ruiz-Lopera, René
  % Restrepo and Néstor Uribe-Patarroyo (2022).
  
  
  % This function is a previous step for the TNode algorithm. In this step
  % we treat the volumetric data as subvolumes which contains information
  % of current processed B-scan with padding. We further divide each
  % subvolume into subBscans (which are in reality 3D). The final volume is
  % then assembled from processed subBscans.
  
  initTime = tic;
  %% Delfault data and automated parameters
  
  % Default parameters
  if nargin == 0
    error('At the least one input is required');
  elseif nargin == 1
    % If nargin is 1, then only noisy image was sent, set all parameters to
    % default
    defaultOptions = GetDefaultOptionsTNode(tom(:, :, 1));
    StructToVars(defaultOptions);
  else
    % Now get default set of options for denoising
    defaultOptions = GetDefaultOptionsTNode(tom(:, :, 1));
    StructToVars(defaultOptions);
    % And replace the defaults
    StructToVars(options);
    if exist('centralPXisOne', 'var')
      error('Change input: centralPXisOne for normalizeSelfSimilarity');
    end
  end
  
  % Using 'enface' or 'XY' must be equivalent
  if strcmp(direction, 'XY')
    direction = 'enface';
  end
  %% Set gpuIdx if need be
  if useGPU && (~exist('gpuIdx', 'var') || isempty(gpuIdx))
    hGPU = gpuDevice;
    gpuIdx = hGPU.Index;
    gpuIdx
    return
  else
    hGPU = gpuDevice(gpuIdx);
  end
  
  %% Set logging to file if desired
  if exist('fileStdout', 'var') && fileStdout
    if useGPU
      logFilename = sprintf('TNode-%s-GPU_%d.log', datestr(now, 'yyyymmdd-HHMMSS'), gpuIdx);
    else
      logFilename = sprintf('TNode-%s-CPU_%d.log', datestr(now, 'yyyymmdd-HHMMSSFFF'), labindex);
    end
    if ~exist('fileStdoutPath', 'var') || isempty(fileStdoutPath)
      fileStdoutPath = '.';
    else
      [~, ~, ~] = mkdir(fileStdoutPath);
    end
    logFileId = fopen(fullfile(fileStdoutPath, logFilename), 'w', 'native', 'UTF-8');
  end
  
  %% Determine additional options that affect output size
  % Discard lower x-percentile of weights
  if contains(simPostProcessing, 'pruning')
    doPruning = true;
    % Surprisingly effective at preserving low-contrast structures in
    % mostly homogeneous regions
    if ~contains(simPostProcessing, 'pruning=')
      % Default to median
      percentile = 50;
      nPrunings = 1;
    else
      %       tokenOut = regexpi(simPostProcessing, 'pruning=([0-9]+),?', 'tokens');
      tokenOutFirst = regexpi(simPostProcessing, 'pruning=([0-9]+,?)+', 'tokens');
      tokenOut = regexpi(tokenOutFirst{1}{1}, '([0-9])+,?+', 'tokens');
      nPrunings = numel(tokenOut);
      percentile = zeros(nPrunings, 1, 'single');
      for thisToken = 1:nPrunings
        percentile(thisToken) = str2double(tokenOut{thisToken}{1});
      end
      percentile = sort(percentile);
    end
    if nPrunings == 1 && percentile == 0
      doPruning = false;
    end
  else
    doPruning = false;
    nPrunings = 1;
  end
  %% Extend volume if required
  % First get a cell array describing the indexing of the whole original volume
  % that will be despeckled
  idxVol = arrayfun(@colon, ones([1, 5]), size(tom, 1:5), 'UniformOutput', false);
  % Number of h0 values to despeckle at once
  nOutputs = numel(h0) * nPrunings;
  idxVol{6} = 1:nOutputs;
  
  % Pad the data
  % Preallocate output
  tomDespeckled = zeros(size(tom, 1), size(tom, 2), size(tom, 3),...
    size(tom, 4), size(tom, 5), nOutputs, 'like', tom);
  % If noisefloor is a scalar, make it an array along dims 1 and 5 so we
  % can pad and index it
  if isscalar(noiseFloorDb)
    noiseFloorDb = noiseFloorDb * ones(size(tom, 1), 1, 1, 1, size(tom, 5), 'single');
  elseif size(noiseFloorDb, 1) == 1 && size(noiseFloorDb, 5) == size(tom, 5)
    noiseFloorDb = noiseFloorDb * ones(size(tom, 1), 1, 1, 1, 1, 'single');
  elseif size(noiseFloorDb, 5) == 1 && size(noiseFloorDb, 1) == size(tom, 1)
    noiseFloorDb = noiseFloorDb * ones(1, 1, 1, 1, size(tom, 5), 'single');
  end
  if nargout > 1
    % Preload the number of similar neighbors
    % Similar neighbor refers to those that have a normalized contribution
    % higher than the average of the normalized weights. We will have this number
    % for each Z, X, Y, W pixel and h0
    nSimNeighbors = zeros(size(tom, 1), size(tom, 2), size(tom, 3), 1,...
      size(tom, 5), nOutputs, 'uint16');
  end
  % Get parameters according to processing direction
  if strcmp(direction, 'ZX')
    % Initial and final Bscans have less information, create a padded version
    % of the tomogram for them to work
    % In ZX, we have everything organized
    extraZ = hSearch(1) + hSimilarity(1);
    extraX = hSearch(2) + hSimilarity(2);
    extraY = hSearch(3) + hSimilarity(3);
    % We also need some extra spectral windows
    extraSpectralWndw = hSearch(4) + hSimilarity(4);
    % Pad
    tom = padarray(tom, [extraZ, extraX, extraY, 0, extraSpectralWndw], edgePadding);
    noiseFloorDb = padarray(noiseFloorDb, [extraZ, 0, 0, 0, extraSpectralWndw], edgePadding);
    % Fix idxVol
    idxVol = cellfun(@plus, idxVol, {extraZ, extraX, extraY, 0, extraSpectralWndw, 0}, 'UniformOutput', false);
    % Block sizes
    zBlockSize = blockSize(1);
    xBlockSize = blockSize(2);
    yBlockSize = blockSize(3);
    % And also ini and fin slices in ZXY
    iniSliceZ = iniSlice(1);
    finSliceZ = finSlice(1);
    iniSliceX = iniSlice(2);
    finSliceX = finSlice(2);
    iniSliceY = iniSlice(3);
    finSliceY = finSlice(3);
    iniSliceS = iniSlice(4);
    finSliceS = finSlice(4);
  elseif strcmp(direction, 'ZY')
    % Initial and final planes have less information, create a padded version
    % of the tomogram for them to work
    % In ZY, we must permute X and Y
    extraZ = hSearch(1) + hSimilarity(1);
    extraX = hSearch(3) + hSimilarity(3);
    extraY = hSearch(2) + hSimilarity(2);
    % We also need some extra spectral windows
    extraSpectralWndw = hSearch(4) + hSimilarity(4);
    % Pad
    tom = padarray(tom, [extraZ, extraY, extraX, 0, extraSpectralWndw], edgePadding);
    noiseFloorDb = padarray(noiseFloorDb, [extraZ, 0, 0, 0, extraSpectralWndw], edgePadding);
    % Fix idxVol
    idxVol = cellfun(@plus, idxVol, {extraZ, extraY, extraX, 0, extraSpectralWndw, 0}, 'UniformOutput', false);
    % Permute the tomogram
    tom = permute(tom, [1, 3, 2, 4, 5]);
    noiseFloorDb = permute(noiseFloorDb, [1, 3, 2, 4, 5]);
    % Permute the indexing cell
    idxVol = idxVol([1, 3, 2, 4, 5]);
    % Permute windows
    hSearch = hSearch([1, 3, 2, 4]);
    hSimilarity = hSimilarity([1, 3, 2, 4]);
    % Swap block sizes between X and Y
    zBlockSize = blockSize(1);
    xBlockSize = blockSize(3);
    yBlockSize = blockSize(2);
    % And also ini and fin slices in ZXY
    iniSliceZ = iniSlice(1);
    finSliceZ = finSlice(1);
    iniSliceX = iniSlice(3);
    finSliceX = finSlice(3);
    iniSliceY = iniSlice(2);
    finSliceY = finSlice(2);
    iniSliceS = iniSlice(4);
    finSliceS = finSlice(4);
  elseif strcmp(direction, 'enface')
    % As we are using XY planes, we may need some extra planes in the XY
    extraZ = hSearch(2) + hSimilarity(2);
    extraX = hSearch(3) + hSimilarity(3);
    extraY = hSearch(1) + hSimilarity(1);
    % We also may need some extra spectral windows
    extraSpectralWndw = hSearch(4) + hSimilarity(4);
    % Same for all other relevant dimensions
    % Pad
    tom = padarray(tom, [extraY, extraZ, extraX, 0, extraSpectralWndw], edgePadding);
    noiseFloorDb = padarray(noiseFloorDb, [extraZ, 0, 0, 0, extraSpectralWndw], edgePadding);
    % Fix idxVol
    idxVol = cellfun(@plus, idxVol, {extraY, extraZ, extraX, 0, extraSpectralWndw, 0}, 'UniformOutput', false);
    % Now, similar to the tomogram filtered in ZY, we can shift dimensions
    % and just use the ZX algorithm in the appropiated direction
    tom = permute(tom, [2, 3, 1, 4, 5]);
    noiseFloorDb = permute(noiseFloorDb, [2, 3, 1, 4, 5]);
    % Permute the indexing cell
    idxVol = idxVol([2, 3, 1, 4, 5, 6]);
    % Permute windows
    hSearch = hSearch([2, 3, 1, 4]);
    hSimilarity = hSimilarity([2, 3, 1, 4]);
    % Swap block sizes among Z, X and Y
    zBlockSize = blockSize(2);
    xBlockSize = blockSize(3);
    yBlockSize = blockSize(1);
    % And also ini and fin slices in ZXY
    iniSliceZ = iniSlice(2);
    finSliceZ = finSlice(2);
    iniSliceX = iniSlice(3);
    finSliceX = finSlice(3);
    iniSliceY = iniSlice(1);
    finSliceY = finSlice(1);
    iniSliceS = iniSlice(4);
    finSliceS = finSlice(4);
  else
    % Not those directions?
    error('Unknow processing direction, only ZX or ZY avaliable');
  end
  
  %% Divide the volume into subVolumes and subBscans
  iter = 1; % A counter for iterations
  
  % Number of Spectral Windows AFTER padding
  nSpectralWindowsPadded = size(tom, 5);
  % Number of spectral windows
  nSpectralWindows = numel(idxVol{5});
  
  if verbosity > 0
    runTimeElapsed = zeros(numel(iniSliceY: finSliceY), 1);
  end
  
  % Limit the amount of Alines that uses the algorithm, the subvolume will be
  % divided into Alines of that given size in order to prevent using too much
  % memory
  
  % First estimate memory required. blocks is only extended by the
  % similarity window (not the search window), but then is multiplied by
  % the search window.
  elementsInMem = size(tom, 4) * (min(zBlockSize, numel(iniSliceZ: finSliceZ)) + 2 * hSimilarity(1)) *...
    (min(xBlockSize, numel(iniSliceX: finSliceX)) + 2 * hSimilarity(2)) *...
    (min(yBlockSize, numel(iniSliceY: finSliceY) + 1) + 2 * hSimilarity(3)) *...
    (min(1, nSpectralWindows) + 2 * hSimilarity(4)) *... % sBlockSize == 1
    prod(1 + 2 * hSearch);
  classSizeBytes = isa(tom, 'single') * 4 + isa(tom, 'double') * 8;
  % Required memory in bytes
  memReq = elementsInMem * classSizeBytes;
  
  if useGPU
    % If using GPU, check that this number is safe and reduce it otherwise
    thisGPU = gpuDevice(gpuIdx);
    memGPU = thisGPU.AvailableMemory;
    % Try to guess if this GPU is already being used in the system or not
    if thisGPU.AvailableMemory / thisGPU.TotalMemory > 0.9
      % Probably completely free for us
      if doPruning
        memOverhead = 7.00; % This is due to array operations needing more memory
      else
        memOverhead = 4.50; % This is due to array operations needing more memory
      end
      gpuMessage = 'considering GPU is not shared';
    else
      % Probably shared, be more conservative on overhead
      if doPruning
        memOverhead = 7.00 * 1.05; % This is due to array operations needing more memory
      else
        memOverhead = 4.50 * 1.05; % This is due to array operations needing more memory
      end
      gpuMessage = 'considering GPU *is* shared';
    end
    logString = sprintf('Estimated GPU RAM needed: %4.1f GiB (%s), GPU RAM available: %4.1f GiB\n',...
      memOverhead * memReq / 1024 ^ 3, gpuMessage, memGPU / 1024 ^ 3);
    fprintf(logString);
    if fileStdout
      fprintf(logFileId, logString);
    end
    overheadVec = [(0 + 2 * (hSimilarity(1) + hSearch(1))) / zBlockSize,...
      (0 + 2 * (hSimilarity(2) + hSearch(2))) / xBlockSize,...
      (0 + 2 * (hSimilarity(3) + hSearch(3))) / yBlockSize];
    logString = sprintf('Current overhead indices are: z=%5.2f, x=%5.2f, y=%5.2f.\n',...
      overheadVec(1), overheadVec(2), overheadVec(3));
    fprintf(logString);
    if fileStdout
      fprintf(logFileId, logString);
    end
    if memGPU < memOverhead * memReq % Add empirical buffer for computations overhead
      memRatio = memGPU / (memOverhead * memReq);
      % Find a suitable blocksize to reduce
      % Sort them by overhead
      [~, overheadPreferredIdx] = sort(overheadVec);
      blocksizeChangeSuccess = false;
      while ~blocksizeChangeSuccess
        switch overheadPreferredIdx(1)
          case 1
            if zBlockSize < finSliceZ - iniSliceZ + 1
              zBlockSize = memRatio * (zBlockSize + 2 * hSimilarity(1)) - 2 * hSimilarity(1);
              zBlockSize = max(1, floor(zBlockSize));
              logString = sprintf('zBlockSize too big for GPU memory, reducing to %d\n', zBlockSize);
              fprintf(logString);
              if fileStdout
                fprintf(logFileId, logString);
              end
              blocksizeChangeSuccess = true;
            else
              overheadPreferredIdx(1) = [];
            end
          case 2
            if xBlockSize < finSliceX - iniSliceX + 1
              xBlockSize = memRatio * (xBlockSize + 2 * hSimilarity(2)) - 2 * hSimilarity(2);
              xBlockSize = max(1, floor(xBlockSize));
              logString = sprintf('xBlockSize too big for GPU memory, reducing to %d\n', xBlockSize);
              fprintf(logString);
              if fileStdout
                fprintf(logFileId, logString);
              end
              blocksizeChangeSuccess = true;
            else
              overheadPreferredIdx(1) = [];
            end
          case 3
            if yBlockSize < finSliceY - iniSliceY + 1
              yBlockSize = memRatio * (yBlockSize + 2 * hSimilarity(3)) - 2 * hSimilarity(3);
              yBlockSize = max(1, floor(yBlockSize));
              logString = sprintf('yBlockSize too big for GPU memory, reducing to %d\n', yBlockSize);
              fprintf(logString);
              if fileStdout
                fprintf(logFileId, logString);
              end
              blocksizeChangeSuccess = true;
            else
              overheadPreferredIdx(1) = [];
            end
        end
        if isempty(overheadPreferredIdx)
          logString = sprintf(strcat('Arrays too large for GPU memory, but unable to find',...
            'a suitable BlockSize to reduce footprint. Will try to proceed\n'));
          fprintf(logString);
          if fileStdout
            fprintf(logFileId, logString);
          end
          blocksizeChangeSuccess = true;
        end
      end
      overheadVecNew = [(0 + 2 * (hSimilarity(1) + hSearch(1))) / zBlockSize,...
        (0 + 2 * (hSimilarity(2) + hSearch(2))) / xBlockSize,...
        (0 + 2 * (hSimilarity(3) + hSearch(3))) / yBlockSize];
      logString = sprintf(['New overhead indices are: z=%5.2f, x=%5.2f, y=%5.2f,',...
        ' if too disparate tweak block sizes by hand to equalize them.\n'],...
        overheadVecNew(1), overheadVecNew(2), overheadVecNew(3));
      fprintf(logString);
      if fileStdout
        fprintf(logFileId, logString);
      end
    elseif memGPU > 1.1 * memOverhead * memReq % If more than 10% RAM not used
      logString = sprintf(...
        'Consider increasing block sizes as GPU seems to have more memory to reduce overhead.\n');
      fprintf(logString);
      if fileStdout
        fprintf(logFileId, logString);
      end
    end
  else
    % If using CPU, show how much memory we expect to require, but do not enforce
    % a change in zBlockSize
    memOverhead = 2;
    logString = sprintf('Estimated RAM needed: %4.1f GiB\n', memOverhead * memReq / 1024 ^ 3);
    fprintf(logString);
    if fileStdout
      fprintf(logFileId, logString);
    end
  end
  
  logString = sprintf('\nRange in tomogram to process:\nZ=%d:%d (%d), X=%d:%d (%d), Y=%d:%d (%d)\n', iniSlice(1),...
    finSlice(1), numel(iniSlice(1): finSlice(1)), iniSlice(2), finSlice(2),...
    numel(iniSlice(2): finSlice(2)), iniSlice(3), finSlice(3), numel(iniSlice(3): finSlice(3)));
  fprintf(logString);
  if fileStdout
    fprintf(logFileId, logString);
  end
  
  % Number of Sub-Bscans
  nSubBscans = ceil(numel(iniSliceX: finSliceX) / xBlockSize);
  
  thisYBlock = 0;
  for thisY = iniSliceY: yBlockSize: finSliceY
    
    runTime = tic;
    thisYBlock = thisYBlock + 1;
    
    % Current first slice in Y in tom padded
    thisIniYInTomPad = thisY + extraY;
    % Current last slice in Y in tom padded
    thisFinYInTomPad = ...
      min(finSliceY + 2 * extraY, thisIniYInTomPad + yBlockSize + extraY - 1);
    % Now we are processing a set of Bscans insted of one-at-the-time in
    % order to take advantage of the memory overhead we already have
    % Range in tom padded
    ySliceRangeInTomPad = thisIniYInTomPad + ...
      (-extraY: thisFinYInTomPad - thisIniYInTomPad);
    % ySlice range in real tomogram
    ySliceRangeInTom = thisY: min(thisY + yBlockSize - 1, finSliceY);
    
    % Some extra information
    if verbosity >= 1
      logString = sprintf('\nCurrent %s range (%d:%d), procesing run: %d/%d\n', ...
        direction, ySliceRangeInTom(1), ySliceRangeInTom(end),...
        thisYBlock, ceil(numel(iniSliceY: finSliceY) / yBlockSize));
      fprintf(logString);
      if fileStdout
        fprintf(logFileId, logString);
      end
    end
    
    % Now take the subvolume consisting of all the neighbooring Bscans
    % subVolume = tom(:, :, ySliceRangeInTomPad);
    % Get a cell array describing the indexing of the subvolume
    % that will be processed
    idxSubVol = idxVol;
    % We need the index of those
    idxSubVol{3} = thisIniYInTomPad + ...
      (0: min(yBlockSize - 1, finSliceY + extraY - thisIniYInTomPad));
    
    % As we may have too many Alines, the subvolume is divided into
    % subBscans and TNode will be used on each of those
    % For each sub-bscan
    thisXBlock = 0;
    for thisX = iniSliceX: xBlockSize: finSliceX
      subvolTime = tic;
      thisXBlock = thisXBlock + 1;
      if verbosity >= 2
        logString = sprintf('\tRun subvolume %d/%d\t', thisXBlock, nSubBscans);
        fprintf(logString);
        if fileStdout
          fprintf(logFileId, logString);
        end
      end
      % Current X in tom padded
      thisIniXInTomPad = thisX + extraX; % thisSliceInTomPad
      % Last X in tom padded
      thisFinXInTomPad = ...
        min(finSliceX + 2 * extraX, thisIniXInTomPad + xBlockSize + extraX - 1);
      % Now we are processing a set of Bscans instead of one-at-the-time in
      % order to take advantage of the memory overhead we already have
      % Range in tom padded
      xSliceRangeInTomPad = thisIniXInTomPad + ...
        (-extraX: thisFinXInTomPad - thisIniXInTomPad);
      % xSlice range in real tomogram
      xSliceRangeInTom = thisX: min(thisX + xBlockSize - 1, finSliceX);
      
      % Now take the subvolume consisting of all the neighbooring Bscans
      % We already have tom padded, so we just take the range in tom padded
      idxSubBscan = idxSubVol;
      idxSubBscan{2} = thisIniXInTomPad + (0: min(xBlockSize - 1,...
        finSliceX + extraX - thisIniXInTomPad));
      
      % Split the subBscan into smaller Bscans whose Z axis is just
      % smaller
      thisZBlock = 0;
      for thisZ = iniSliceZ: zBlockSize: finSliceZ
        thisZBlock = thisZBlock + 1;
        if verbosity >= 3
          if thisZBlock == 1
            logString = sprintf(' [Sub-block (/%d): %d', ceil(numel(iniSliceZ: finSliceZ) / zBlockSize), thisZBlock);
          elseif thisZBlock == ceil(numel(iniSliceZ: finSliceZ) / zBlockSize)
            logString = sprintf('; %d]', thisZBlock);
          else
            logString = sprintf('; %d', thisZBlock);
          end
          fprintf(logString);
          if fileStdout
            fprintf(logFileId, logString);
          end
        end
        
        % Current Z in tom padded
        thisIniZInTomPad = thisZ + extraZ;
        % Last Z in tom padded
        thisFinZInTomPad = ...
          min(finSliceZ + 2 * extraZ, thisIniZInTomPad + zBlockSize + extraZ - 1);
        % z range in tom padded (from thisZInTomPad)
        zSliceRangeInTomPad = thisIniZInTomPad + ...
          (-extraZ: thisFinZInTomPad - thisIniZInTomPad);
        % z slice in real tom
        zSliceRangeInTom = thisZ: min(thisZ + zBlockSize - 1, finSliceZ);
        % SubBscan in new z ROI
%         subZRangeBscan = ...
%           tom(zSliceRangeInTomPad, xSliceRangeInTomPad, ySliceRangeInTomPad, :, :);
        % New relative index
        idxSubZRangeBscan = idxSubBscan;
        % Save new index
        idxSubZRangeBscan{1} = thisIniZInTomPad + ...
          (0: min(zBlockSize - 1, finSliceZ + extraZ - thisIniZInTomPad));
        
        % Split the subBscan into subBscan based only on the required spectral
        % windows
        
        % Apply TNode to each spectral window
        % Sub Bscan filtered only contains information of the central Bscan,
        % but it possesses the same spectral windows as the subVolume
        subVolumeDespeckled = zeros(cellfun(@numel, idxSubZRangeBscan), 'like', tom);
        % idxVol{5} is the actual number of spectral windows
        
        if nargout > 1
          subBscanSimNeigh = zeros([size(subVolumeDespeckled, 1:3), 1, size(subVolumeDespeckled, 5:6)], 'like', subVolumeDespeckled);
        end
        
        % Loop through spectral windows, the idea is to make an analog to the
        % subBscan, so that we transfer the least amount of spectral windows to
        % the algorithm, reducing the required memory
        thisSBlock = 0;
        nSBlocks = finSliceS - iniSliceS + 1;
        for thisSpecW = iniSliceS: finSliceS
          % The loop is only for the real spectral windows in idxVol, notice
          % that the size of tom in this dimension maybe bigger, as it has been
          % padded
          
          % Get required spectral windows around thisSpecW
          specWnRange = thisSpecW + extraSpectralWndw + (-extraSpectralWndw: extraSpectralWndw);
          
          % Take spectralWin ROI from subBscan
          subSpecBscan = tom(zSliceRangeInTomPad, xSliceRangeInTomPad, ySliceRangeInTomPad, :, specWnRange);
          if strcmp(direction, 'ZX') || strcmp(direction, 'ZY')
            subSpecNoiseFloorDb = noiseFloorDb(zSliceRangeInTomPad, :, :, :, specWnRange);
          elseif strcmp(direction, 'enface')
            subSpecNoiseFloorDb = noiseFloorDb(:, :, ySliceRangeInTomPad, :, specWnRange);
          end            
          % Now calculate subBscan indexing cell of the region under processing
          % relative to the subBscan
          % Z Range
          idxSubSpecBscanRel{1} = extraZ + (1: numel(zSliceRangeInTom));
          % X Range
          idxSubSpecBscanRel{2} = extraX + (1:numel(xSliceRangeInTom));
          % Y Range
          idxSubSpecBscanRel{3} = extraY + (1: numel(ySliceRangeInTom));
          % Stokes components are just the same
          idxSubSpecBscanRel{4} = idxSubZRangeBscan{4};
          % Spectral windows Range
          idxSubSpecBscanRel{5} = extraSpectralWndw + 1;
          % And process each spectral window independently, but with the required
          % additional information.
          if nargout == 1
            subVolumeDespeckled(:, :, :, :, thisSpecW, :) =...
              ApplyTNodeToSubBscan(subSpecBscan, hSimilarity, hSearch,...
              h0, h1, nAverageBScans,...
              subSpecNoiseFloorDb,...
              normalizeSelfSimilarity,...
              hSearchKernel, hSimiKernel, simPostProcessing, useGPU,...
              hGPU, idxSubSpecBscanRel, rescaleWeights, hSimNorm, direction);
          elseif nargout > 1
            [subVolumeDespeckled(:, :, :, :, thisSpecW, :),...
              subBscanSimNeigh(:, :, :, :, thisSpecW, :)] =...
              ApplyTNodeToSubBscan(subSpecBscan, hSimilarity, hSearch,...
              h0, h1, nAverageBScans,...
              subSpecNoiseFloorDb,...
              normalizeSelfSimilarity,...
              hSearchKernel, hSimiKernel, simPostProcessing, useGPU,...
              hGPU, idxSubSpecBscanRel, rescaleWeights, hSimNorm, direction);
          end
        end
        
        
        % Reconstruct the filtered tomogram with the subBscans filtered
        % It depends of the direction
        if strcmp(direction, 'ZX')
          % In ZX only replace data for the ALine range in the current B-scan
          tomDespeckled(zSliceRangeInTom, xSliceRangeInTom, ySliceRangeInTom, :, :, :) = ...
            subVolumeDespeckled;
          
          if nargout > 1
            nSimNeighbors(zSliceRangeInTom, xSliceRangeInTom, ySliceRangeInTom, :, :, :) = ...
              subBscanSimNeigh;
          end
        elseif strcmp(direction, 'ZY')
          % In ZX permute before replacing data for the ALine range in the current ZY-plane
          tomDespeckled(zSliceRangeInTom, ySliceRangeInTom, xSliceRangeInTom, :, :, :) = ...
            permute(subVolumeDespeckled, [1, 3, 2, 4, 5, 6]);
          
          if nargout > 1
            nSimNeighbors(zSliceRangeInTom, ySliceRangeInTom, xSliceRangeInTom, :, :, :) = ...
              permute(subBscanSimNeigh, [1, 3, 2, 4, 5, 6]);
          end
        elseif strcmp(direction, 'enface')
          % If we used en face projections, we need to shift the third
          % dimension
          tomDespeckled(ySliceRangeInTom, zSliceRangeInTom, xSliceRangeInTom, :, :, :) =...
            permute(subVolumeDespeckled, [3, 1, 2, 4, 5, 6]);
          
          if nargout > 1
            nSimNeighbors(ySliceRangeInTom, zSliceRangeInTom, xSliceRangeInTom, :, :, :) = ...
              permute(subBscanSimNeigh, [3, 1, 2, 4, 5, 6]);
          end
        end
      end
      
      % Show some additional information
      if verbosity >= 2
        logString = newline;
        fprintf(logString);
        if fileStdout
          fprintf(logFileId, logString);
        end
      end
      subvolTimeElapsed = toc(subvolTime);
      logString = sprintf('\t\tAll sub-blocks for subvolume %d done, subvolume processing time: %.2f s\n', ...
        thisXBlock, subvolTimeElapsed);
      fprintf(logString);
      if fileStdout
        fprintf(logFileId, logString);
      end
    end
    if nargout > 1
      varargout{1} = nSimNeighbors;
    end
    clear subBscan subBscanDespeckled subVolume
    
    %% Show additional information?
    if verbosity > 0
      % Verbosity level 1: show timers
      runTimeElapsed = toc(runTime);
      logString = sprintf('\tAll subvolumes for %s run %d done, run processing time: %.2f min\n', ...
        direction, thisYBlock, runTimeElapsed / 60);
      fprintf(logString);
      if fileStdout
        fprintf(logFileId, logString);
      end
    end
    % Convert to double to prevent error when showFigs is boolean
    showFigs = double(showFigs);
    if showFigs
      % Define config
      if contains(figFunc, 'truesize')
        FigFunc = @() truesize;
      elseif contains(figFunc, 'axis image')
        FigFunc = @() axis('image');
      else
        FigFunc = @() 1;
      end
      if contains(figFunc, 'ch=')
        tokenOutFirst = regexpi(figFunc, 'ch=([0-9]+,?)+', 'tokens');
        tokenOut = regexpi(tokenOutFirst{1}{1}, '([0-9])+,?+', 'tokens');
        plotCh = str2double(tokenOut{1}{1});
      else
        plotCh = 1;
      end
      % Verbosity level 2: Create figures with speckle and filtered Bscans
      imaSpeckleName = 'Original '; % Image with speckle name
      imaFiltName = 'TNode '; % Despeckled image name
      
      % Noisy intensity images
      figure2(showFigs);
      switch direction
        case 'ZX'
          imagesc(10*log10(FlattenArrayTo2D(tom(idxSubVol{1}(iniSliceZ:finSliceZ),...
            idxSubVol{2}(iniSliceX:finSliceX), thisIniYInTomPad, plotCh,...
            idxVol{5}))),imRange)
          colormap(gray(256)), colorbar;
          xlabel({'X', 'W'}), ylabel({'Z', 'I'});
        case 'ZY'
          imagesc(10*log10(FlattenArrayTo2D(...
            squeeze(tom(idxSubVol{1}(iniSliceZ:finSliceZ),...
            idxSubVol{2}(iniSliceX:finSliceX), thisIniYInTomPad, plotCh,...
            idxSubVol{5})))),imRange)
          colormap(gray(256)), colorbar;
          xlabel({'Y', 'W'}), ylabel({'Z', 'I'});
        case 'enface'
          imagesc(10*log10(FlattenArrayTo2D(...
            permute(squeeze(tom(idxSubVol{1}(iniSliceZ:finSliceZ),...
            idxSubVol{2}(iniSliceX:finSliceX), thisIniYInTomPad, plotCh,...
            idxSubVol{5})), [2 1 3 4]))),imRange), colormap(gray(256)), colorbar;
          xlabel({'X', 'W'}), ylabel({'Y', 'I'});
      end
      title(horzcat(imaSpeckleName, direction, ': ', num2str(thisY)));
      FigFunc();
      drawnow;
      % Despeckled intensity images
      [h0Mat, pruningMat] = ndgrid(h0(:), percentile(:));
      [h1Mat, ~] = ndgrid(h1(:), percentile(:));
      for thisOutput = 1:nOutputs
        figure2(showFigs + 1 + (thisOutput - 1));
        switch direction
          case 'ZX'
            imagesc(10*log10(FlattenArrayTo2D(...
              tomDespeckled(iniSliceZ:finSliceZ, iniSliceX:finSliceX,...
              thisY, plotCh, :, thisOutput))), imRange)
            colormap(gray(256)), colorbar;
            xlabel({'X', 'W'}), ylabel({'Z', 'I'});
          case 'ZY'
            imagesc(10*log10(FlattenArrayTo2D(...
              squeeze(tomDespeckled(iniSliceZ:finSliceZ, thisY,...
              iniSliceX:finSliceX, plotCh, :, thisOutput)))), imRange)
            colormap(gray(256)), colorbar;
            xlabel({'Y', 'W'}), ylabel({'Z', 'I'});
          case 'enface'
            imagesc(10*log10(FlattenArrayTo2D(...
              permute(squeeze(tomDespeckled(thisY, iniSliceZ:finSliceZ,...
              iniSliceX:finSliceX, plotCh, :, thisOutput)), [2 1 3 4]))), imRange)
            colormap(gray(256)), colorbar;
            xlabel({'X', 'W'}), ylabel({'Y', 'I'});
        end
        title(horzcat(imaFiltName, direction, ': ', num2str(thisY),...
          sprintf(', h0=%.3f, h1=%.3f, pr=%d%%', h0Mat(thisOutput), h1Mat(thisOutput), pruningMat(thisOutput))));
        FigFunc();
      end
      drawnow;
    end
    iter = iter + 1;
  end
  
  timeTot = toc(initTime);
  logString = sprintf('\n\nAll data processed, TNode took %.2f minutes\n', timeTot/60);
  fprintf(logString);
  if fileStdout
    fprintf(logFileId, logString);
  end
  %% Turn off logging to file if enabled
  if exist('fileStdout', 'var') && fileStdout
    fclose(logFileId);
  end
end
