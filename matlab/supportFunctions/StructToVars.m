function StructToVars(varsStruct)
  %UNTITLED Summary of this function goes here
  %   Detailed explanation goes here
  
  varNames = fieldnames(varsStruct);
    
  for k = 1:numel(varNames)
    assignin('caller', varNames{k}, varsStruct.(varNames{k}));
  end
  
  
end

