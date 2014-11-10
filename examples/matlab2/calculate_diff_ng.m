function [ diff mean nums2] = calculate_diff_ng(groundtruth , rests, m_size )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here


    f =  size(groundtruth,1);

    diff = zeros(f,1);
    mean = zeros(f,1);
    
    nums2 = 0;


    for i = 1:f

        orig = groundtruth(i,:);


        grp = rests(i,:);

        if sum(grp) ~= 0 
            
            ox1 = max(min(orig(1), m_size(1)),0);
            oy1 = max(min(orig(2), m_size(2)),0);
            ox2 = max(min(orig(1)+orig(3), m_size(1)),0);
            oy2 = max(min(orig(2)+orig(4), m_size(2)),0);
            
            gx1 = max(min(grp(1), m_size(1)),1);
            gy1 = max(min(grp(2), m_size(2)),1);
            gx2 = max(min(grp(5), m_size(1)),1);
            gy2 = max(min(grp(6), m_size(2)),1);
            
            osch = zeros(m_size);
            gsch = zeros(m_size);
%             try
            osch(ox1:ox2,oy1:oy2) = 1;
            gsch(gx1:gx2,gy1:gy2) = 1;
            
            
            intersection = osch & gsch;
            union = osch | gsch;
%             catch
%                osch; 
%             end

%             w = orig(3);
%             h = orig(4);
% 
% 
%             w_p = grp(3) - grp(1) ;
%             h_p = grp(8) - grp(2);
%             
%             lefttopx = max(orig(1),grp(1));
%             lefttopy = max(orig(2),grp(2));
%             rigthbottomx = min(orig(1)+orig(3),grp(5));
%             rigthbottomy = min(orig(2)+orig(4),grp(6));
% 
%             owidth = rigthbottomx - lefttopx;
%             iheight = rigthbottomy - lefttopy;
% 
%              diff(i) = abs(owidth*iheight - w_p*h_p)/owidth*iheight;
             
            diff(i) = sum(sum(intersection))/sum(sum(union));

             if  (diff(i) >= 0.5)
                 
                nums2 = nums2 + 1; 
                
             end
%             diff(i) = (owidth*iheight)/(2*w*h-owidth*iheight);
            m1x = (orig(1)+orig(3) + orig(1))/2;
            m1y = (orig(2) + orig(2)+orig(4))/2;
            m2x = (grp(1) + grp(5))/2;
            m2y = (grp(2) + grp(6))/2;
            
            mean(i) = ((m1x-m2x)^2+(m1y-m2y)^2)^(1/2);
%             mean(i) = abs(m1x-m2x)+abs(m1y-m2y);
%             
            
        end

    end
    
    nums2 = nums2/f;
end

