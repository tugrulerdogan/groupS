function [ hists, param ] = im2hist( wimgs, param )
%IM2GIST Summary of this function goes here
%   Detailed explanation goes here

    sz = size(wimgs,3);
    
    hists = zeros(256, sz);

    for i = 1:sz
        
%         gists(:,i) = LMgist(wimgs(:,:,i), '', param);
        hists(:,i) = imhist(uint8(wimgs(:,:,i)));

        
    end
    


end

