function [hists, param ] = vec2gist( wimgs, param )
%IM2GIST Summary of this function goes here
%   Detailed explanation goes here

    sz = size(wimgs,2);
    
    hists = zeros(256, sz);
        
    for i = 1:sz
        
%         gists(:,i) = LMgist(reshape(wimgs(:,i),32,32), '', param);
        hists(:,i) = imhist(uint8(reshape(wimgs(:,1),32,32)))/size(wimgs(:,1),1);
        
    end
    
end

