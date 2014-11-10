function [ hists, param ] = im2histcolor( wimgs, param )
%IM2GIST Summary of this function goes here
%   Detailed explanation goes here

% %     wm = wimgs(:,:,:,1) + wimgs(:,:,:,2) + wimgs(:,:,:,3);
%     wm = reshape(wimgs, [size(wimgs,1)*size(wimgs,2)*size(wimgs,4),  size(wimgs,3)]);
%     
%     wimgs = wm;

    sz = size(wimgs,3);
    ch_num = size(wimgs,4);
    bin_num=256;

    
    hists = zeros(bin_num*ch_num, sz);

    for i = 1:sz
                
        chnl = zeros(bin_num, ch_num);
        for j = 1:ch_num
%           gists(:,i) = LMgist(wimgs(:,:,i), '', param);
            chnl(:,j) = imhist(uint8(wimgs(:,:,i,j)),bin_num);
        end
        
        hists(:,i) = chnl(:);
        
    end
    


end


% function [hists, param ] = im2histcolor( wimgs, param )
% %IM2GIST Summary of this function goes here
% %   Detailed explanation goes here
% 
% % %     wm = wimgs(:,:,1) + wimgs(:,:,2) + wimgs(:,:,3);
% %     wm = reshape(wimgs, [size(wimgs,1)*size(wimgs,3),  size(wimgs,2)]);
%     
% %     wimgs = wm;
% %     
%     sz = size(wimgs,3);
%     ch_num = size(wimgs,4);
%     
%     hists = zeros(bin_num, sz);
%         
%     for i = 1:sz
%                 
%         chnl = zeros(bin_num, ch_num);
%         for j = 1:ch_num
%         
%     %         gists(:,i) = LMgist(reshape(wimgs(:,i),32,32), '', param);
%             chnl(:,j) = imhist(uint8(reshape(wimgs(:,:,i,j),32,32)));
%         end
%     end
%     
% end
