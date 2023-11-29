function sizeVect = SafeSize(tom, idxVect)
% This function extends 'size' function in old MATLAB versions to return
% ones in non-existent dimensions (equivalently to 'size' in MATLAB 2020)

% Make a vector of ones, that way we make sure that output is filled with
% ones in non-existent dimensions
sizeVect = ones(1, idxVect(end));
% Assign the corresponding size to existent dimensions
sizeVect(1:ndims(tom)) = size(tom);
% Return only the size for the dimensions of interest
sizeVect = sizeVect(idxVect);

end