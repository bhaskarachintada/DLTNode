function [ val ] = FlattenArrayTo2D(array, varargin)
  %FlattenArray Functional form of (:, :)
  %   Useful when indexing has already taken place and (:, :) cannot be used in an expression
  %   Optional second argument ind makes the indexing as (1:ind(1):end, 1:ind(2):end) 
  
  if nargin == 2 && isvector(varargin{1})
    ind = varargin{1};
  else
    ind = [1 1];
  end
  
  val = array(1:ind(1):end, 1:ind(2):end);
end

