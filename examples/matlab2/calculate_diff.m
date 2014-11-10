function [ diff mean nums2] = calculate_diff(groundtruth , rests )
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

            w = orig(3);
            h = orig(4);


            w_p = grp(3) - grp(1) ;
            h_p = grp(8) - grp(2);
            
            lefttopx = max(orig(1),grp(1));
            lefttopy = max(orig(2),grp(2));
            rigthbottomx = min(orig(1)+orig(3),grp(5));
            rigthbottomy = min(orig(2)+orig(4),grp(6));

            owidth = rigthbottomx - lefttopx;
            iheight = rigthbottomy - lefttopy;

             diff(i) = abs(owidth*iheight - w_p*h_p)/owidth*iheight;
             
             if  (1.0*(diff(i))/owidth*iheight >= 0.5)
                 
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

