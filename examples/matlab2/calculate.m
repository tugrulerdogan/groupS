

basename = 'face'
w_dir = fullfile('D:\vot7\rests', basename)



% % files = dir([w_dir '\*.mat'])
% % 
% % for i = 1:numel(files) 
% %     
% %     path = fullfile(w_dir, files(i).name)
% %     
% %     load path
% %     
% % end
% 
% 
% load 'D:\vot7\rests\face\gt.mat'
% load 'D:\vot7\rests\face\rests.mat'
% load 'D:\vot7\rests\face\results.mat'
% 
% 
% [diff1 mean1 nm1] = calculate_diff(groundtruth , rests );
% [diff2 mean2 nm2] = calculate_diff(groundtruth , results );

tname  = 'hand'
output = {};


load(sprintf('D:\\vot7\\rests\\%s\\gt.mat', tname));

indx=1;
load(sprintf('D:\\vot7\\rests\\result_%s_1.mat', tname)); %results
[diff2 mean2 nm] = calculate_diff(groundtruth , results );
ouput{indx} = {mean(diff2) mean(mean2) nm};

indx=2;
load(sprintf('D:\\vot7\\rests\\result_%s_2.mat', tname)); %results
[diff2 mean2 nm] = calculate_diff(groundtruth , results );
ouput{indx} = {mean(diff2) mean(mean2) nm};

indx=3;
load(sprintf('D:\\vot7\\rests\\result_%s_3.mat', tname)); %results
[diff2 mean2 nm] = calculate_diff(groundtruth , results );
ouput{indx} = {mean(diff2) mean(mean2) nm};

indx=4;
load(sprintf('D:\\vot7\\rests\\result_%s_12.mat', tname)); %results
[diff2 mean2 nm] = calculate_diff(groundtruth , results );
ouput{indx} = {mean(diff2) mean(mean2) nm};

indx=5;
load(sprintf('D:\\vot7\\rests\\result_%s_13.mat', tname)); %results
[diff2 mean2 nm] = calculate_diff(groundtruth , results );
ouput{indx} = {mean(diff2) mean(mean2) nm};

indx=6;
load(sprintf('D:\\vot7\\rests\\result_%s_23.mat', tname)); %results
[diff2 mean2 nm] = calculate_diff(groundtruth , results );
ouput{indx} = {mean(diff2) mean(mean2) nm};

indx=7;
load(sprintf('D:\\vot7\\rests\\result_%s_123.mat', tname)); %results
[diff2 mean2 nm] = calculate_diff(groundtruth , results );
ouput{indx} = {mean(diff2) mean(mean2) nm};




