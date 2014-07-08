function [ gists, param ] = vec2gist( wimgs, param )
%IM2GIST Summary of this function goes here
%   Detailed explanation goes here

    sz = size(wimgs,2);
    gists = zeros(496, sz);
        
    for i = 1:sz
        
%         gists(:,i) = LMgist(reshape(wimgs(:,i),32,32), '', param);
        a = vl_hog(single(reshape(wimgs(:,i),32,32)), param.orientationsPerScale(1));
        gists(:,i) = a(:);
        
    end
  
end

