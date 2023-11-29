function [subVolumeDespeckled, varargout] = ApplyTNodeToSubBscan(subVolume, hSimiWindow,...
    hSearchWindow, h0, h1, nAverageBScans, noiseFloorDb,...
    normalizeSelfSimilarity, hSearchKernel, hSimiKernel, simPostProcessing,...
    useGPU, hGPU, idxSubSpecBscanRel, rescaleWeights, hSimNorm, direction)

  % ApplyPrCoherentDenoisingToSubBscan Applies the PrCoherentDenoising procedure to the subVolume of the tomogram.
  
  % Inputs:
  %   subVolume: is the set of Bscans required to despeckle the central one.
  %   hSimiWindow: Similarity window half-size.
  %   hSearchWindow: Search window half-size.
  %   h0: Base speckle reduction parameter.
  %   h1: SNR-dependent parameter.
  %   nAverageBScans: Amount of previously averaged Bscans.
  %   noiseFloorDb: Noise floor in dB.
  %   normalizeSelfSimilarity: Central pixel behavior.
  %   useGPU: Wheter to use GPU device.
  %   hSimNorm: Remove mean intensity before processing. Accounts for
  %     depth-intensity fall-off.
  %   idxSubSpecBscanRel: Indices of subVolume referenced to the tomogram.
  %
  % Outputs:
  %   bscanFiltered: The B-scan after despeckling.
  %   varargout: If requested, returns a matrix of the size of subVolume with
  %     the amount of significant neighbors.
  %
  % See also PerformPrCoherentDenoising
  
  %
  % Authors:  Carlos Cuartas-Vélez {1*}, René Restrepo {1}, B. E. Bouma {2}
  %            and Néstor Uribe-Patarroyo {2}.
  % CCV - RR:
  %	 1. Applied Optics Group, Universidad EAFIT, Carrera 49 # 7 Sur-50,
  %     Medellín, Colombia.
  %
  % BEB - NUP:
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
  % V3.0 (2021-02-04): Version based on PSPrCoherentDenoising code base
  %
  % Copyright Carlos Cuartas-Vélez (2018), René Restrepo (2021), Sebastián
  % Ruiz-Lopera (2021) and Néstor Uribe-Patarroyo (2021).
  %

  % Use one dimensional convolutions? Using 2D- or 3D-dimensional
  % convolutions seems extremely slow, using 1D-convolutions multiple times
  % is just a little bit better
  use1DConvn = ~(range(hSimiWindow(1:2)) == 0);
  
  % Get similarity window size in each direction, as well as search window
  % We have all the input in a vector, but if we are interested on ZX or ZY
  % it should use different parameters
  % Z
  hSimiWinZ = hSimiWindow(1);
  hSearchWinZ = hSearchWindow(1);
  % X
  hSimiWinX = hSimiWindow(2);
  hSearchWinX = hSearchWindow(2);
  % Y
  hSimiWinY = hSimiWindow(3);
  hSearchWinY = hSearchWindow(3);
  % But we may have some extra dimensions over the similarity and
  % search window
  hSearchWinSpec = hSearchWindow(4);
  hSimiWinSpec = hSimiWindow(4);
  
  % PerformBlockShift4D requires indexing subVolume many times across all
  % dimensions. This is faster in the GPU only for large arrays; otherwise,
  % the CPU is more efficient. The empirical limit is ~85,000 elements in
  % subVolume. Here, we determine whether to use the CPU or GPU for block
  % shifting. If we use the CPU, we will transfer blocks to the GPU after
  % shifting. This is relevant with blocksizes found in small Bscans, where
  % the CPU can be 2x faster than the GPU.
  GPU_SHIFT_SIZE_THRESH = 85000;
  if numel(subVolume) >= GPU_SHIFT_SIZE_THRESH
    performBlockShiftIn = 'GPU';
  else
    performBlockShiftIn = 'CPU';
  end
  
  % Transfer subVolume to GPU before block shifting
  if useGPU && strcmp(performBlockShiftIn, 'GPU')
    subVolume = gpuArray(subVolume);
  end
  
  % Here is where we need to shift each similarity window within the search
  % window. To do this in a vectorized manner, we create a 4D array,
  % blocks, containing all the possible shifts. This makes the variable
  % very large, with prod(2 * hSearchWindow + 1) elements in the 4th
  % dimension. The ZXYS dimensions will have the same size as subVolume
  % minus the search window sizes along each dimension. Therefore, the
  % sizes are the same as the final subVolume plus the similarity window
  % sizes.
  
  % Use the function to create blocks
  dimsFilt = num2cell(cellfun(@numel, idxSubSpecBscanRel));
  subVolumeFinalSizeIdx = cellfun(@colon, repmat({1}, 1, 5), dimsFilt, 'UniformOutput', false);
  blocks = PerformBlockShift4DOptim(subVolume, hSearchWindow, hSearchWinZ, hSearchWinX,...
    hSearchWinY, hSearchWinSpec, hSimiWinZ, hSimiWinX, hSimiWinY, hSimiWinSpec,...
    dimsFilt);
  
  % After block shifting in CPU, transfer subVolume and blocks to GPU
  if useGPU && strcmp(performBlockShiftIn, 'CPU')
    subVolume = gpuArray(subVolume);
    blocks = gpuArray(blocks);
  end
  % Save the position of centered image
  CENTRAL = (size(blocks, 4) + 1) / 2;
  
  % Normalize intensities referenced to the search or similarity window?
  if hSimNorm
    % If we normalize by the similarity window, this means we try to compensate
    % for the light attenuation in tissue. After despeckling, we put the mean
    % value of each similarity window back.
    % Just an average over Z, X and Y
    % Kernel
    if ~use1DConvn
      hSimMeanIntKernel = ones(2 * hSimiWindow + 1) /...
        prod(2 * hSimiWindow + 1);
      % Get mean intensity over I
      hSimMeanInt = imfilter(subVolume, hSimMeanIntKernel);
      % Remove this from subVolume
      subVolume = bsxfun(@rdivide, subVolume, hSimMeanInt);
      % And continue
    else
      % Z
      hSimMeanInt = imfilter(subVolume,...
        ones(2 * hSimiWinZ + 1, 1, 1) / (2 * hSimiWinZ + 1));
      % X
      hSimMeanInt = imfilter(hSimMeanInt,...
        ones(1, 2 * hSimiWinX + 1, 1) / (2 * hSimiWinX + 1));
      % Y
      hSimMeanInt = imfilter(hSimMeanInt,...
        ones(1, 1, 2 * hSimiWinY + 1) / (2 * hSimiWinY + 1));
      % Remove this from subVolume
      subVolume = bsxfun(@rdivide, subVolume, hSimMeanInt);
    end
  end
  
  
  % Now we compute the similarity criterion for the whole subvolume. We do
  % the calculation in two steps because it is more memory efficient by
  % avoiding the creation of TWO variables of size simCriterion. The ZXYS
  % sizes are the same as the final subVolume plus the similarity window
  % sizes.
  
  % Here we're not dividing by h yet
  logSimCriterionInv = blocks(:, :, :, CENTRAL, :, :) + blocks;
  logSimCriterionInv = (0.5 * logSimCriterionInv) .^ 2 ./ blocks(:, :, :, CENTRAL, :, :);
  logSimCriterionInv = nAverageBScans * log(logSimCriterionInv ./ blocks);
  
  % Later, we will use the blocks variable to perform the actual weighted
  % average. For this, we do not need to preserve the full extent of the
  % similarity window, just the actual subvolume. So get rid of unnecessary
  % blocks so this matches the size of the final subVolume
  blocks = blocks(...
    hSimiWinZ + 1:end - hSimiWinZ,...
    hSimiWinX + 1:end - hSimiWinX,...
    hSimiWinY + 1:end - hSimiWinY,...
    :, :,...
    hSimiWinSpec + 1:end - hSimiWinSpec);
  
  % If we have h1 > 0, we are using the adaptive version. Pre-calculate
  % hActual now
  adaptiveH = h1 ~= 0;
  % Promote h0 to 7th dimension
  h0 = permute(single(h0(:)), [2 3 4 5 6 7 1]);
  h1 = permute(single(h1(:)), [2 3 4 5 6 7 1]);
  if any(adaptiveH, 'all')
    % We have seen most of the time that using the similarity window does not
    % make sense, so we use mean intensities over search windows
    % Get mean intensity for each search window, which is larger and fulfills
    % the purpose of correcting for SNR due to light attenuation in tissue
    
    % Kernel
    subVolumeMeanIntKernel = ones(...
      2 * hSearchWinZ + 1, 2 * hSearchWinX + 1, 2 * hSearchWinY + 1, 1, 2 * hSearchWinSpec + 1) / ...
      prod(2 * hSearchWindow + 1);
    
    % Trim noiseFloorDb spectrally as we do not need the additional
    % search window elements
    noiseFloorDb = noiseFloorDb(:, :, :, :, hSearchWinSpec + 1:end - hSearchWinSpec, :);
    % Also trim in Z
    if strcmp(direction, 'ZX') || strcmp(direction, 'ZY')
      noiseFloorDb = noiseFloorDb(hSearchWinZ + 1:end - hSearchWinZ, :, :, :, :, :);
    elseif strcmp(direction, 'enface')
      noiseFloorDb = noiseFloorDb(:, :, hSearchWinY + 1:end - hSearchWinY, :, :, :);
    end

    % If using hSimNorm need to put it back in
    if hSimNorm
      if ~use1DConvn
        % Same here, no need to put original intensity back in
        subVolumeMeanIntSNR = convn(...
          bsxfun(@times, subVolume, hSimMeanInt), ...
          subVolumeMeanIntKernel, 'valid') ./ 10 .^ (noiseFloorDb / 10);
      else
        % Add mean intensity back
        % Z
        subVolumeMeanIntSNR = convn(...
          bsxfun(@times, subVolume, hSimMeanInt), ...
          ones(2 * hSearchWinZ + 1, 1, 1) / (2 * hSearchWinZ + 1), 'valid');
        % X
        subVolumeMeanIntSNR = convn(subVolumeMeanIntSNR, ...
          ones(1, 2 * hSearchWinX + 1, 1) / (2 * hSearchWinX + 1), 'valid');
        % Y
        subVolumeMeanIntSNR = convn(subVolumeMeanIntSNR, ...
          ones(1, 1, 2 * hSearchWinY + 1) / (2 * hSearchWinY + 1),...
          'valid') ./ 10 .^ (noiseFloorDb / 10);
      end
    else
      if ~use1DConvn
        % Same here, no need to put original intensity back in
        subVolumeMeanIntSNR = convn(subVolume, subVolumeMeanIntKernel,...
          'valid') ./ 10 .^ (noiseFloorDb / 10);
      else
        % Do not add mean intensity back
        % Z
        subVolumeMeanIntSNR = convn(subVolume, ...
          ones(2 * hSearchWinZ + 1, 1, 1) / (2 * hSearchWinZ + 1), 'valid');
        % X
        subVolumeMeanIntSNR = convn(subVolumeMeanIntSNR, ...
          ones(1, 2 * hSearchWinX + 1, 1) / (2 * hSearchWinX + 1), 'valid');
        % Y
        subVolumeMeanIntSNR = convn(subVolumeMeanIntSNR, ...
          ones(1, 1, 2 * hSearchWinY + 1) / (2 * hSearchWinY + 1),...
          'valid') ./ 10 .^ (noiseFloorDb / 10);
      end
    end
		% Make sure SNR is positive
    subVolumeMeanIntSNR = max(0, subVolumeMeanIntSNR);
    % Asymptotic parametrization
    hActual = h0 + h1 ./ (1 + 1 ./ subVolumeMeanIntSNR .^ 0.5);
    % Now move the last 2 dimensions to make space for the shift dimension in blocks
    hActual = permute(hActual, [1 2 3 6 4 5 7]);
    % Trim it
    hActual = hActual(hSimiWinZ + 1:end - hSimiWinZ,...
      hSimiWinX + 1:end - hSimiWinX,...
      hSimiWinY + 1:end - hSimiWinY,...
      :, :, hSimiWinSpec + 1:end - hSimiWinSpec, :);
  else
    % If not using adaptive h, just get h0
    hActual = h0;
  end
  clear subVolumeMeanIntSNR

  % We now start compounding the similarity criterion by summing (because
  % it's the log probability) over the similarity window with a kernel
  % (it's a "running" average), and for this reason we could do it with
  % convn. However, because the size of the convolution kernel is usually
  % small, it is much faster to use loops in which we shift the position
  % for each kernel pixel manually.
  
  % We no longer compound probability over the spectral dimension if we
  % have multiple spectral windows because we allow for arbitrary kernel
  % shapes too

  % We also use the for loop to iterate through the kernel size, not the
  % blocksize, adn that would be very inefficient.
  
  % The order of the convolutions do not matter. Here, we start with Z as
  % it is more likely we will reduce the array sizes this way at the first
  % calc instead of the last one. TO BE DONE, RIGHT NOW WE START WITH S.
  
  % Define a non-unitary kernel for the similarity window. All kernels are
  % column vectors, no need to define them along the corresponding
  % dimension. Because we are compounding probabilities, windows should not
  % be normalized (with sum()), instead, they should have 1s for regular
  % compounding, and <1 for weaker compounding.
  if isempty(hSimiKernel) || strcmpi(hSimiKernel, 'unitary')
    simKernelZ = ones(2 * hSimiWinZ + 1, 1, 'like', real(subVolume));
    simKernelX = ones(2 * hSimiWinX + 1, 1, 'like', real(subVolume));
    simKernelY = ones(2 * hSimiWinY + 1, 1, 'like', real(subVolume));
    simKernelS = ones(2 * hSimiWinSpec + 1, 1, 'like', real(subVolume));
  elseif strcmpi(hSimiKernel, 'triangular')
    simKernelZ = cast(triang(2 * hSimiWinZ + 1), 'like', real(subVolume));
    simKernelX = cast(triang(2 * hSimiWinX + 1), 'like', real(subVolume));
    simKernelY = cast(triang(2 * hSimiWinY + 1), 'like', real(subVolume));
    simKernelS = cast(triang(2 * hSimiWinSpec + 1), 'like', real(subVolume));
  elseif strcmpi(hSimiKernel, 'optfilt')
    simKernelZ = cast(AnisotropicGaussianExp2Diameter([1, floor(hSimiWinZ) * 2 + 1],...
      1, hSimiWinZ), 'like', real(subVolume));
    simKernelX = cast(AnisotropicGaussianExp2Diameter([1, floor(hSimiWinX) * 2 + 1],...
      1, hSimiWinX), 'like', real(subVolume));
    simKernelY = cast(AnisotropicGaussianExp2Diameter([1, floor(hSimiWinY) * 2 + 1],...
      1, hSimiWinY), 'like', real(subVolume));
    simKernelS = cast(AnisotropicGaussianExp2Diameter([1, floor(hSimiWinSpec) * 2 + 1],...
      1, hSimiWinSpec), 'like', real(subVolume));
    % Create a filter similar to the optimum filter: one towards the center,
    % zeros towards the edge, with a smooth transition
    % Change constant to zero to have unsmoothed weights
    simKernelZ = simKernelZ ./ (simKernelZ + (0.01 * max(simKernelZ, [], 'all')));
    simKernelX = simKernelX ./ (simKernelX + (0.01 * max(simKernelX, [], 'all')));
    simKernelY = simKernelY ./ (simKernelY + (0.01 * max(simKernelY, [], 'all')));
    simKernelS = simKernelS ./ (simKernelS + (0.01 * max(simKernelS, [], 'all')));
  elseif isvector(hSimiKernel)
    simKernelZ = cast(hSimiKernel(:), 'like', real(subVolume));
    simKernelX = cast(hSimiKernel(:), 'like', real(subVolume));
    simKernelY = cast(hSimiKernel(:), 'like', real(subVolume));
    simKernelS = cast(hSimiKernel(:), 'like', real(subVolume));
  else
    error('Unknown similarity kernel type')
  end
  
  % First along S
  if hSimiWinSpec > 1
    logSimCriterionInvConvOverS = zeros(size(logSimCriterionInv, 1), size(logSimCriterionInv, 2),...
      size(logSimCriterionInv, 3), size(logSimCriterionInv, 4), size(logSimCriterionInv, 5),...
      size(logSimCriterionInv, 6) - 2 * hSimiWinSpec, 'like', logSimCriterionInv);
    
    % And convolve manually differently
    for thisShift = -hSimiWinSpec:hSimiWinSpec
      % Shifted index
      idxS = subVolumeFinalSizeIdx{5} + thisShift + hSimiWinSpec;
      % Sum over S Similarity window
      logSimCriterionInvConvOverS = logSimCriterionInvConvOverS + ...
        simKernelS(hSimiWinSpec + 1 + thisShift) .* logSimCriterionInv(:, :, :, :, :, idxS) / (2 * hSimiWinSpec + 1);
      if useGPU
        wait(hGPU)
      end
    end
  else
    logSimCriterionInvConvOverS = logSimCriterionInv;
  end
  clear logSimCriterionInv
  
  % Second along Y
  if hSimiWinY > 1
    logSimCriterionInvConvOverSY = zeros(size(logSimCriterionInvConvOverS, 1), size(logSimCriterionInvConvOverS, 2),...
      size(logSimCriterionInvConvOverS, 3) - 2 * hSimiWinY, size(logSimCriterionInvConvOverS, 4), size(logSimCriterionInvConvOverS, 5),...
      size(logSimCriterionInvConvOverS, 6), 'like', logSimCriterionInvConvOverS);
    
    % And convolve manually differently
    for thisShift = -hSimiWinY:hSimiWinY
      % Shifted index
      idxY = subVolumeFinalSizeIdx{3} + thisShift + hSimiWinY;
      % Sum over Y Similarity window
      logSimCriterionInvConvOverSY = logSimCriterionInvConvOverSY + ...
        simKernelY(hSimiWinY + 1 + thisShift) .* logSimCriterionInvConvOverS(:, :, idxY, :, :, :) / (2 * hSimiWinY + 1);
      if useGPU
        wait(hGPU)
      end
    end
  else
    logSimCriterionInvConvOverSY = logSimCriterionInvConvOverS;
  end
  clear logSimCriterionInvConvOverS
  
  % Third along X
  if hSimiWinX > 1
    logSimCriterionInvConvOverSYX = zeros(size(logSimCriterionInvConvOverSY, 1), size(logSimCriterionInvConvOverSY, 2) - 2 * hSimiWinX,...
      size(logSimCriterionInvConvOverSY, 3), size(logSimCriterionInvConvOverSY, 4), size(logSimCriterionInvConvOverSY, 5),...
      size(logSimCriterionInvConvOverSY, 6), 'like', logSimCriterionInvConvOverSY);
    
    % And convolve manually differently
    for thisShift = -hSimiWinX:hSimiWinX
      % Shifted index
      idxX = subVolumeFinalSizeIdx{2} + thisShift + hSimiWinX;
      % Sum over Y Similarity window
      logSimCriterionInvConvOverSYX = logSimCriterionInvConvOverSYX + ...
        simKernelX(hSimiWinX + 1 + thisShift) .* logSimCriterionInvConvOverSY(:, idxX, :, :, :, :) / (2 * hSimiWinX + 1);
      if useGPU
        wait(hGPU)
      end
    end
  else
    logSimCriterionInvConvOverSYX = logSimCriterionInvConvOverSY;
  end
  clear logSimCriterionInvConvOverSY
  
  % Fourth along Z
  if hSimiWinZ > 1
    logSimCriterionInvConvOverSYXZ = zeros(size(logSimCriterionInvConvOverSYX, 1) - 2 * hSimiWinZ, size(logSimCriterionInvConvOverSYX, 2),...
      size(logSimCriterionInvConvOverSYX, 3), size(logSimCriterionInvConvOverSYX, 4), size(logSimCriterionInvConvOverSYX, 5),...
      size(logSimCriterionInvConvOverSYX, 6), 'like', logSimCriterionInvConvOverSYX);
    
    % And convolve manually differently
    for thisShift = -hSimiWinZ:hSimiWinZ
      % Shifted index
      idxZ = subVolumeFinalSizeIdx{1} + thisShift + hSimiWinZ;
      % Sum over Y Similarity window
      logSimCriterionInvConvOverSYXZ = logSimCriterionInvConvOverSYXZ + ...
        simKernelZ(hSimiWinZ + 1 + thisShift) .*  logSimCriterionInvConvOverSYX(idxZ, :, :, :, :, :) / (2 * hSimiWinZ + 1);
      if useGPU
        wait(hGPU)
      end
    end
  else
    logSimCriterionInvConvOverSYXZ = logSimCriterionInvConvOverSYX;
  end
  clear logSimCriterionInvConvOverSYX
  
  % Apply h to simCriterion. This is not memory efficient yet, but at least allows to
  % filter with h0 as a vector
  logSimCriterionInvConvOverSYXZ = logSimCriterionInvConvOverSYXZ ./ hActual;
  clear hActual
  % Compound probability over 4th dimension of tom
  if size(logSimCriterionInvConvOverSYXZ, 5) > 1
    logSimCriterionInvConvOverSYXZ = mean(logSimCriterionInvConvOverSYXZ, 5);
  end

  % Compute the weights for each pixel in the Bscan for all shifts. We use
  % the exp(-logSimCriterion) to undo the inverse we had before (which was
  % used to save on GPU memory usage).
  weights = exp(-logSimCriterionInvConvOverSYXZ);
  clear logSimCriterionInvConvOverSYXZ
  
  % Set behavior of central pixel
  if normalizeSelfSimilarity
    % Make the central patch the same value as the next maximum patch
    weights(:, :, :, CENTRAL, :, :, :) = nan;
    weights(:, :, :, CENTRAL, :, :, :) = max(weights, [], 4, 'omitnan');
  else
    % Central patch should already be 1
    weights(:, :, :, CENTRAL, :, :, :) = 1;
  end
  
  % Now normalize weights
  % Cummulative weights for the denominator of the NLM average
  cumWeights = sum(weights, 4);
  weights = weights ./ cumWeights;
  
  % Now w is an array with z and x coordinates in 1st and 2nd index, 3rd
  % index is y coordinate, 4th index is shifts, 5th index is one among
  % which probability is *not* compounded and 6th is spectral window where
  % it *is* compounded
  
  % Smooth weights along the search window to avoid edge artifacts
  if strcmpi(hSearchKernel, 'triangular')
    searchKernelZ = cast(triang(2 * hSimiWinZ + 1), 'like', real(subVolume));
    searchKernelX = cast(triang(2 * hSimiWinX + 1), 'like', real(subVolume));
    searchKernelY = cast(triang(2 * hSimiWinY + 1), 'like', real(subVolume));
    searchKernelZXY = searchKernelZ .* permute(searchKernelX, [2 1]) .* permute(searchKernelY, [3 2 1]);
  elseif strcmpi(hSearchKernel, 'optfilt')
    searchKernelZ = cast(AnisotropicGaussianExp2Diameter([1, floor(hSearchWinZ) * 2 + 1],...
      1, hSearchWinZ), 'like', real(subVolume));
    searchKernelX = cast(AnisotropicGaussianExp2Diameter([1, floor(hSearchWinX) * 2 + 1],...
      1, hSearchWinX), 'like', real(subVolume));
    searchKernelY = cast(AnisotropicGaussianExp2Diameter([1, floor(hSearchWinY) * 2 + 1],...
      1, hSearchWinY), 'like', real(subVolume));
    searchKernelZXY = searchKernelZ .* permute(searchKernelX, [2 1]) .* permute(searchKernelY, [3 2 1]);
    % Create a filter similar to the optimum filter: one towards the center,
    % zeros towards the edge, with a smooth transition
    % Change constant to zero to have unsmoothed weights
    searchKernelZXY = searchKernelZXY ./ (searchKernelZXY + (0.001 * max(searchKernelZXY, [], 'all')));
  elseif isvector(hSearchKernel) && ~ischar(hSearchKernel)
    searchKernelZ = cast(hSearchKernel(:), 'like', real(subVolume));
    searchKernelX = cast(hSearchKernel(:), 'like', real(subVolume));
    searchKernelY = cast(hSearchKernel(:), 'like', real(subVolume));
    searchKernelZXY = searchKernelZ .* permute(searchKernelX, [2 1]) .* permute(searchKernelY, [3 2 1]);
  elseif ~strcmpi(hSearchKernel, 'unitary')
    error('Unknown search kernel type')
  end
  if ~isempty(hSearchKernel) && ~strcmpi(hSearchKernel, 'unitary')
    % Calculate new weights with the smoothing filter
    weights = bsxfun(@times, weights, permute(searchKernelZXY(:), [2 3 4 1]));
    % and normalize again
    weights = weights ./ sum(weights, 4);
  end
  
  % Discard lower x-percentile of weights
  if contains(simPostProcessing, 'pruning')
    % Surprisingly effective at preserving low-contrast structures in
    % mostly homogeneous regions
    if ~contains(simPostProcessing, 'pruning=')
      % Default to median
      pruningPercentileVec = 50;
      nPrunings = 1;
    else
      %       tokenOut = regexpi(simPostProcessing, 'pruning=([0-9]+),?', 'tokens');
      tokenOutFirst = regexpi(simPostProcessing, 'pruning=([0-9]+,?)+', 'tokens');
      tokenOut = regexpi(tokenOutFirst{1}{1}, '([0-9])+,?+', 'tokens');
      nPrunings = numel(tokenOut);
      pruningPercentileVec = zeros(nPrunings, 1, 'single');
      for thisToken = 1:nPrunings
        pruningPercentileVec(thisToken) = str2double(tokenOut{thisToken}{1});
      end
      % Need to sort to be able to do this cummulatively
      pruningPercentileVec = sort(pruningPercentileVec);
    end
  else
    nPrunings = 1;
    pruningPercentileVec = 0;
  end
  
  subBscanSimNeigh = zeros([SafeSize(weights, 1:3), 1, SafeSize(weights, 5:6), size(weights, 7) * nPrunings], 'like', weights);
  subVolumeDespeckled = zeros([SafeSize(weights, 1:3), 1, SafeSize(blocks, 5:6), size(weights, 7) * nPrunings], 'like', weights);

  if rescaleWeights
    weights = (weights - min(weights, [], 4)) ./ (max(weights, [], 4) - min(weights, [], 4));
    weights = weights ./ sum(weights, 4);
  end
  
  for thisPruning = 1:nPrunings
    if pruningPercentileVec(thisPruning) ~= 0
      thisThreshold = prctile(weights, pruningPercentileVec(thisPruning), 4);
      weightsLowIdx = weights < thisThreshold;
      weights(weightsLowIdx) = 0;
      weights = weights ./ sum(weights, 4);
    end
    % Get number of similar patches if requested
    if nargout > 1
      % Return the amount of shifts in which weights were higher than 5%
      % Significant weight threshold
      % It seems like mean weight for a subvolume is ~0.0125, let's use 0.011
      % instead of 0.05
      signiWThresh = 1 / size(weights, 4); % Those that contribute more than average, was 0.01, why??
      subBscanSimNeigh(:, :, :, :, :, :,...
        numel(h0) * (thisPruning - 1) + (1:numel(h0))) =...
        sum(weights >= signiWThresh, 4) - 1; % -1 to subtract central
        % patch
    end
    % Weight every single patch, we do all spectral windows and Stokes parms in
    % parallel
    subVolumeDespeckled(:, :, :, :, :, :,...
        numel(h0) * (thisPruning - 1) + (1:numel(h0))) = sum(bsxfun(@times, weights, blocks), 4);
  end
  % Move dims > 3 down
  subVolumeDespeckled = permute(subVolumeDespeckled, [1 2 3 5 6 7 4]);
  
  clear weights blocks

  % If we normalized by sim window mean intensity, put it back in
  if hSimNorm
    subVolumeDespeckled = bsxfun(@times,...
      subVolumeDespeckled, hSimMeanInt(idxSubSpecBscanRel{:}));
  end
  
  % Move data back to CPU if necessary
  if useGPU
    subVolumeDespeckled = gather(subVolumeDespeckled);
  end
  
  if nargout > 1
    if useGPU
      varargout{1} = gather(subBscanSimNeigh);
    else
      varargout{1} = subBscanSimNeigh;
    end
  end
end
