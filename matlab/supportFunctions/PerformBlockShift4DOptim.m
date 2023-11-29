function blocks = PerformBlockShift4DOptim(subVolume, hSearchWindow, hSearchWinZ, hSearchWinX,...
    hSearchWinY, hSearchWinSpec, hSimiWinZ, hSimiWinX, hSimiWinY, hSimiWinSpec,...
    dimsFilt)
  
  % number of elements of each kind for final despeckled subvolume
  [nZFilt, nXFilt, nYFilt, N_STOKES, nSpectralWsFilt] = deal(dimsFilt{:});
  
  nZSimi = nZFilt + 2 * hSimiWinZ;
  nXSimi = nXFilt + 2 * hSimiWinX;
  nYSimi = nYFilt + 2 * hSimiWinY;
  nSpecSimi = nSpectralWsFilt + 2 * hSimiWinSpec;
  blocks = zeros(nZSimi, nXSimi, nYSimi, prod(2 * hSearchWindow + 1),...
    N_STOKES, nSpecSimi, 'like', subVolume);
  blocksZ = zeros(nZSimi, nXSimi, nYSimi, prod(2 * hSearchWindow(1) + 1),...
    N_STOKES, nSpecSimi, 'like', subVolume);
  
  % Create a "new" volume for each displacement and save it into blocks
  % Index for shifts
  shiftCount = 1;
  
  for dS = -hSearchWinSpec: hSearchWinSpec
    % Shift over spectral windows
    sVec = hSearchWinSpec + dS + (1: nSpecSimi);
    for dY = -hSearchWinY: hSearchWinY
      % Shift over Y
      yVec = hSearchWinY + dY + (1: nYSimi);
      for dX = -hSearchWinX: hSearchWinX
        % Shift over X
        xVec = hSearchWinX + dX + (1: nXSimi);
        shiftCountZ = 1;
        for dZ = -hSearchWinZ: hSearchWinZ
          % Shift over z
          zVec = hSearchWinZ + dZ + (1: nZSimi);
          blocksZ(:, :, :, shiftCountZ, :, :) = subVolume(zVec, xVec, yVec, :, sVec);
          % Increase counter
          shiftCountZ = shiftCountZ + 1;
          shiftCount = shiftCount + 1;
        end
        blocks(:, :, :, shiftCount + (-2 * hSearchWinZ - 1:-1), :, :) = blocksZ;
      end
    end
  end
end