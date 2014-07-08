function [ gists, param ] = im2gist( wimgs, param )
%IM2GIST Summary of this function goes here
%   Detailed explanation goes here

    sz = size(wimgs,3);
    
    gists = zeros(496, sz);

    for i = 1:sz
        
%         gists(:,i) = LMgist(wimgs(:,:,i), '', param);
        a = vl_hog(single(wimgs(:,:,i)), param.orientationsPerScale(1));
        gists(:,i) = a(:);
       
    end
    


end

