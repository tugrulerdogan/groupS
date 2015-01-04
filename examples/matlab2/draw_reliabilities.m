% ===============================
% ======= INITIALIZATIONS =======

% The base path of result files
base = 'D:\vot7\rests'      

% The base path of image frames
image_base = 'D:\mtest\sequences' 

% The name of the set
tname = 'torus'

% The frame number of visualisation
frame_no = 15

% The test methods
itrts = {[1,2]}


% ===============================


% draw the results
colorstring = 'gbrycykm';
labels = {'intensity', 'Hog'};

for iterations = itrts
    iterations = cell2mat(iterations);

    % load weights from test results
    load(sprintf('%s\\weights_%s_%s.mat', base, tname, strtrim(sprintf('%d', iterations))));
    
    % create a blank graphing surface
    h = figure(1); cla;
    hold on
    
    % for each cue
    for i = iterations
         
        % TODO: each label prints same place so it erases previous ones
        xlabel(sprintf('%s', labels{i}),'FontSize',12, 'Color', colorstring(i));  
        
        % plot the weights
        plot(wights(:, i), 'Color', colorstring(i), 'linewidth', 1.45)
            
    end
    
    % save drawed graph as image on disk
    saveas(h,sprintf('%s\\jpgs\\result_%s_diffs_%s.jpg', base, tname, strtrim(sprintf('%d', iterations))));

end