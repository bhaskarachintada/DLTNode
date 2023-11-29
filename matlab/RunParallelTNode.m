function [tomDespeckled, nSimNeighbors] =...
    RunParallelTNode(tom, options, parallelDim)
  %RunParallelTNode Wrapper function to run PerformTNode in parallel in
  %multiple GPUs
  %   If not given, determine parallelDim: the dimension with the most work
  %   (including overhead) that's not singleton.
  % 
  if nargin == 2 || isempty(parallelDim)
    if iscell(options.hSearch)
      % Just pick the first as representative of all
      overhead = 2 * options.hSimilarity + 1 + 2 * options.hSearch{1} + 1;
      singleton = (options.finSlice - options.iniSlice) == 0;
      [~, parallelDim] = max(~singleton .* ((options.finSlice - options.iniSlice + 1) + overhead));
    else
      overhead = 2 * options.hSimilarity + 1 + 2 * options.hSearch + 1;
      singleton = (options.finSlice - options.iniSlice) == 0;
      [~, parallelDim] = max(~singleton .* ((options.finSlice - options.iniSlice + 1) + overhead));
    end
  end
  nGPUs = numel(options.gpuIdx);
  batchSizeParallel = ceil((options.finSlice(parallelDim) - options.iniSlice(parallelDim) + 1) / nGPUs);
  nBatches = ceil((options.finSlice(parallelDim) - options.iniSlice(parallelDim) + 1) / batchSizeParallel);
  nZ = size(tom, 1);
  nX = size(tom, 2);
  nY = size(tom, 3);
  nPolChs = size(tom, 4);
  nS = size(tom, 5);
  nDimsMax = 5;
  colonOp = repmat({':'}, [1 nDimsMax]);

  if nGPUs > 1 && nBatches > 1
    % Split volume in several GPUs. Consider if there's a way to define
    % codistributed arrays for this; difficult with the need to have
    % overlapping elements.
    tomDespeckled = zeros(nZ, nX, nY, nPolChs, nS, 'single');
    nSimNeighbors = zeros(nZ, nX, nY, 1, nS, 'uint16');
    spmd(nBatches)
      % Determine slice limits for processing in this worker
      iniBatchParallel = batchSizeParallel * (labindex - 1) + 1;
      finBatchParallel = min(options.finSlice(parallelDim), batchSizeParallel * labindex);
      theseOptions = options;
      theseOptions.iniSlice(parallelDim) = iniBatchParallel;
      theseOptions.finSlice(parallelDim) = finBatchParallel;
      % Choose desired GPU
      theseOptions.gpuIdx = options.gpuIdx(labindex);
      % Process and save data in Composite variables
      [thisTomDespeckled, thisNSimNeighbors] = PerformTNode(tom, theseOptions);
    end
    % Now we need a for loop to gather the data from each worker in the
    % Composite variables
    for batchIdx = 1:nBatches
      % Recalculate the limits as those vars were Composites
      iniBatchParallel = batchSizeParallel * (batchIdx - 1) + 1;
      finBatchParallel = min(options.finSlice(parallelDim), batchSizeParallel * batchIdx);
      % We need temp vars because Composites cannot be indexed the way we
      % need
      batchTomDespeckled = thisTomDespeckled{batchIdx};
      batchNSimNeighbors = thisNSimNeighbors{batchIdx};
      % Now get the data from each results with the corresponding range
      thisColonOp = colonOp;
      if parallelDim > 3
        outputParallelDim = parallelDim + 1;
      else
        outputParallelDim = parallelDim;
      end
      thisColonOp{outputParallelDim} = iniBatchParallel:finBatchParallel;
      thisColonOp{parallelDim} = iniBatchParallel:finBatchParallel;
      tomDespeckled(thisColonOp{:}) = batchTomDespeckled(thisColonOp{:});
      nSimNeighbors(thisColonOp{:}) = batchNSimNeighbors(thisColonOp{:});
    end
  else
    % Run in single GPU
    theseOptions = options;
    % Make sure only one GPU is selected
    theseOptions.gpuIdx = options.gpuIdx(1);
    [tomDespeckled, nSimNeighbors] = PerformTNode(tom, theseOptions);
  end
end

