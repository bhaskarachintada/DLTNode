function blocks = PerformParallelBlockShift4D(subVolume, hSearchWindow, hSearchWinZ, hSearchWinX,...
    hSearchWinY, hSearchWinSpec, hSimiWinZ, hSimiWinX, hSimiWinY, hSimiWinSpec,...
    dimsFilt)
  
  % number of elements of each kind for final despeckled subvolume
  [nZFilt, nXFilt, nYFilt, N_STOKES, nSpectralWsFilt] = deal(dimsFilt{:});
  
  nZSimi = nZFilt + 2 * hSimiWinZ;
  nXSimi = nXFilt + 2 * hSimiWinX;
  nYSimi = nYFilt + 2 * hSimiWinY;
  nSpecSimi = nSpectralWsFilt + 2 * hSimiWinSpec;
  blocks = zeros(nZSimi, nXSimi, nYSimi, prod(2 * hSearchWindow(1:2) + 1),...
    N_STOKES, nSpecSimi, 'like', subVolume);
  
  for dS = -hSearchWinSpec: hSearchWinSpec
    % Shift over spectral windows
    sVec = hSearchWinSpec + dS + (1: nSpecSimi);
    parfor (k = 1:1 + 2 * hSearchWinY)
      dY = -hSearchWinY + k - 1;
      shiftCount = 1;
      theseBlocks = zeros(nZSimi, nXSimi, nYSimi,...
        (1 + 2 * hSearchWinX) * (1 + 2 * hSearchWinZ), N_STOKES, nSpecSimi,...
        'like', subVolume);
      % Shift over Y
      yVec = hSearchWinY + dY + (1: nYSimi);
      for dX = -hSearchWinX: hSearchWinX
        % Shift over X
        xVec = hSearchWinX + dX + (1: nXSimi);
        for dZ = -hSearchWinZ: hSearchWinZ
          % Shift over z
          zVec = hSearchWinZ + dZ + (1: nZSimi);
          theseBlocks(:, :, :, shiftCount, :, :) = subVolume(zVec, xVec, yVec, :, sVec);
          % Increase counter
          shiftCount = shiftCount + 1;
        end
      end
      blocks(:, :, :, :, k) = theseBlocks;
    end
    blocks = blocks(:, :, :, :);
  end
end