function blocks = PerformBlockShift3D(subVolume, hSearchWindow, hSearchWinZ, hSearchWinX,...
    hSearchWinY, hSimiWinZ, hSimiWinX, hSimiWinY, idxSubSpecBscanRel)
  
  % Dimensions after TNode
  dimsFilt = num2cell(cellfun(@numel, idxSubSpecBscanRel));
  % number of elements of each kind for final despeckled subvolume
  [nZFilt, nXFilt, nYFilt] = deal(dimsFilt{:});
  
  nZSimi = nZFilt + 2 * hSimiWinZ;
  nXSimi = nXFilt + 2 * hSimiWinX;
  nYSimi = nYFilt + 2 * hSimiWinY;
  blocks = zeros(nZSimi, nXSimi, nYSimi, prod(2 * hSearchWindow + 1),...
    'like', subVolume);
  
  % Create a "new" volume for each displacement and save it into blocks
  % Index for Y
  shiftCount = 1;
  
  % Calculate the z indices we need. We always want nZ, but we restrict this to
  % valid indices. We repeat indices if they go out of the range ('repeat'). We
  % should later add options for 'circular' and 'symmetric'.
  % Here, we only need repeat option realistically, we have already padded
  % the initial tomogram. For all subvolumes we already have additional
  % information from the tomogram itself
  for dY = -hSearchWinY: hSearchWinY
    % Shift over Y
    yVec = hSearchWinY + dY + (1: nYSimi);
    for dX = -hSearchWinX: hSearchWinX
      % Shift over X
      xVec = hSearchWinX + dX + (1: nXSimi);
      for dZ = -hSearchWinZ: hSearchWinZ
        % Shift over z
        zVec = hSearchWinZ + dZ + (1: nZSimi);
        blocks(:, :, :, shiftCount) = subVolume(zVec, xVec, yVec);
        % Increase counter
        shiftCount = shiftCount + 1;
      end
    end
  end
