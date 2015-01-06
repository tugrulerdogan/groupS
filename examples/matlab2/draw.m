% ===============================
% ======= INITIALIZATIONS =======

% The base path of result files
base = 'D:\vot7\rests'      

% The base path of image frames
image_base = 'D:\mtest\sequences' 

% The name of the set
tname = 'iceskater'

% The frame number of visualisation
% frame_no = 15

% The test methods
itrts = {[1,2]}


% ===============================

colorstring = 'brycykm';
labels = {'adaptive', 'intensity', 'Hog'};

% open the frame imge
h = figure(1); cla;
imshow(sprintf('%s\\%s\\%08d.jpg', image_base, tname, frame_no));

% draw the ground-truth
load(sprintf('%s\\%s\\gt.mat', base, tname));
gt = groundtruth(frame_no,:);
hold on
plot([gt(1) gt(1)+gt(3) ],[gt(2) gt(2)],'g','linewidth', 2.5)
plot([gt(1)+gt(3) gt(1)+gt(3)],[ gt(2) gt(2)+gt(4)],'g','linewidth', 2.5)
plot([gt(1)+gt(3) gt(1)],[gt(2)+gt(4) gt(2)+gt(4)],'g','linewidth', 2.5)
plot([gt(1) gt(1)],[gt(2)+gt(4) gt(2)],'g','linewidth', 2.5)


% draw the results

i = 1;
for iterations = itrts
    iterations = cell2mat(iterations);

    load(sprintf('%s\\result_%s_%s.mat', base, tname, strtrim(sprintf('%d', iterations))));

    rst = results(frame_no,:);

    plot([rst(1) rst(3)], [rst(2) rst(4)], colorstring(i), 'linewidth', 2.5)
    plot([rst(3) rst(5)], [rst(4) rst(6)], colorstring(i), 'linewidth', 2.5)
    plot([rst(5) rst(7)], [rst(6) rst(8)], colorstring(i), 'linewidth', 2.5)
    plot([rst(7) rst(1)], [rst(8) rst(2)], colorstring(i), 'linewidth', 2.5)
    
    for itr = iterations
        
        i = i+1;
        load(sprintf('%s\\result_%s_%s.mat', base, tname, strtrim(sprintf('%d', itr))));

        rst = results(frame_no,:);

        plot([rst(1) rst(3)], [rst(2) rst(4)], colorstring(i), 'linewidth', 2.5)
        plot([rst(3) rst(5)], [rst(4) rst(6)], colorstring(i), 'linewidth', 2.5)
        plot([rst(5) rst(7)], [rst(6) rst(8)], colorstring(i), 'linewidth', 2.5)
        plot([rst(7) rst(1)], [rst(8) rst(2)], colorstring(i), 'linewidth', 2.5)
        xlabel(sprintf('%s', labels{i}),'FontSize',12, 'Color', colorstring(i)); 
        
    end

   
end

saveas(h,sprintf('%s\\jpgs\\result_%s_results_%s_%d.jpg', base, tname, strtrim(sprintf('%d', iterations)), frame_no));