

basename = 'face'
w_dir = fullfile('D:\vot7\rests', basename)

draw_graph =0 



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

% tnames = {'juice' 'iceskater' 'gymnastics' 'face' 'diving' 'david'};
% tnames = {'sunshade' 'torus' 'woman' 'singer'};
tnames = {'singer'};

mns = zeros(size(tnames,2),4);
indx = 0;
output = zeros(size(tnames,2),7); 
output1 = zeros(size(tnames,2),7);
output2 = zeros(size(tnames,2),7);
output3 = zeros(size(tnames,2),7);

iii=0;
for tname=tnames
    
    iii=iii+1;
%  tname  = 'jump'
tname = tname{:}
cal_size = [240,320];
% output = {};
warning('off','MATLAB:colon:nonIntegerIndex');

try
load(sprintf('D:\\vot7\\rests\\%s\\gt.mat', tname));
catch
    
    groundtruth = importdata(sprintf('D:\\mtest\\sequences\\%s\\groundtruth.txt', tname));
end
indx=1;
% load(sprintf('D:\\vot7\\rests\\result_%s_1.mat', tname)); %results
load(sprintf('D:\\vot7\\rests\\result_%s_3.mat', tname)); %results
[diff2 mean2 nm] = calculate_diff_ng(groundtruth , results, cal_size );

% diffs = [[1:size(diff2,1)]; diff2'];
% means = [[1:size(mean2,1)]; mean2'];
diffs = diff2';
means = mean2';

ouput{indx} = {mean(diff2) mean(mean2) nm};
output1(indx,iii) = mean(diff2);
output2(indx,iii) = mean(mean2);
output3(indx,iii) = nm;

return


indx=2;
load(sprintf('D:\\vot7\\rests\\result_%s_2.mat', tname)); %results
[diff2 mean2 nm] = calculate_diff_ng(groundtruth , results, cal_size );

diffs = [diffs; diff2'];
means = [means; mean2'];


ouput{indx} = {mean(diff2) mean(mean2) nm};
output1(indx,iii) = mean(diff2);
output2(indx,iii) = mean(mean2);
output3(indx,iii) = nm;

indx=3;
load(sprintf('D:\\vot7\\rests\\result_%s_3.mat', tname)); %results
[diff2 mean2 nm] = calculate_diff_ng(groundtruth , results, cal_size );

diffs = [diffs; diff2'];
means = [means; mean2'];

ouput{indx} = {mean(diff2) mean(mean2) nm};
output1(indx,iii) = mean(diff2);
output2(indx,iii) = mean(mean2);
output3(indx,iii) = nm;

indx=4;
load(sprintf('D:\\vot7\\rests\\result_%s_12.mat', tname)); %results
[diff2 mean2 nm] = calculate_diff_ng(groundtruth , results, cal_size );

diffs = [diffs; diff2'];
means = [means; mean2'];

ouput{indx} = {mean(diff2) mean(mean2) nm};
output1(indx,iii) = mean(diff2);
output2(indx,iii) = mean(mean2);
output3(indx,iii) = nm;

indx=5;
load(sprintf('D:\\vot7\\rests\\result_%s_13.mat', tname)); %results
[diff2 mean2 nm] = calculate_diff_ng(groundtruth , results, cal_size );

diffs = [diffs; diff2'];
means = [means; mean2'];

ouput{indx} = {mean(diff2) mean(mean2) nm};
output1(indx,iii) = mean(diff2);
output2(indx,iii) = mean(mean2);
output3(indx,iii) = nm;

indx=6;
load(sprintf('D:\\vot7\\rests\\result_%s_23.mat', tname)); %results
[diff2 mean2 nm] = calculate_diff_ng(groundtruth , results, cal_size );

diffs = [diffs; diff2'];
means = [means; mean2'];

ouput{indx} = {mean(diff2) mean(mean2) nm};
output1(indx,iii) = mean(diff2);
output2(indx,iii) = mean(mean2);
output3(indx,iii) = nm;

indx=7;
load(sprintf('D:\\vot7\\rests\\result_%s_123.mat', tname)); %results
[diff2 mean2 nm] = calculate_diff_ng(groundtruth , results, cal_size );

diffs = [diffs; diff2'];
means = [means; mean2'];

diffs = [[1:7]', diffs];
means = [[1:7]', means];

ouput{indx} = {mean(diff2) mean(mean2) nm};
output1(indx,iii) = mean(diff2);
output2(indx,iii) = mean(mean2);
output3(indx,iii) = nm;

% mns = zeros(size(tnames,2

mns(:,1) = 1:size(tnames,2);
mns(iii, 2) = mean(diff2);
mns(iii, 3) = mean(mean2);
mns(iii, 4) = nm;

if draw_graph == 1
colorstring = 'ymcrgbk';
h = figure(1); cla;
hold on
for i = 1:7
  plot(diffs(i, 2:end), 'Color', colorstring(i))
end

saveas(h,sprintf('D:\\vot7\\rests\\jpgs\\result_%s_diffs_org_all.jpg', tname))

for i = 1:7
    h = figure(1); cla;
    hold on
    xlabel(sprintf('%d', i),'FontSize',12, 'Color', colorstring(i));  
    plot(diffs(i, 2:end), 'Color', colorstring(i))
    saveas(h,sprintf('D:\\vot7\\rests\\jpgs\\result_%s_diffs_org_%d.jpg', tname, i));
end

h = figure(1); cla;
hold on
for i = 1:7
  plot(means(i, 2:end), 'Color', colorstring(i))
end

saveas(h,sprintf('D:\\vot7\\rests\\jpgs\\result_%s_means_org_all.jpg', tname));

for i = 1:7
    h = figure(1); cla;
    hold on
    xlabel(sprintf('%d', i),'FontSize',12, 'Color', colorstring(i));  
    plot(means(i, 2:end), 'Color', colorstring(i))
    saveas(h,sprintf('D:\\vot7\\rests\\jpgs\\result_%s_means_org_%d.jpg', tname, i));
end

end






for iiii=2:size(diff2,1)+1
   
    
    diffs = sortrows(diffs,iiii);
    diffs(:,iiii) = 1:7;
    diffs = sortrows(diffs,1);
    
    means = sortrows(means,iiii);
    means(:,iiii) = 1:7;
    means = sortrows(means,1);
    
%     diffs = sortrows(diffs,iiii);
%     diffs(:,iiii) = 1:size(diff2,1);
%     diffs = sortrows(diffs,1);
%     
%     means = sortrows(means,iiii);
%     means(:,iiii) = 1:size(diff2,1);
%     means = sortrows(means,1);
    
end


if draw_graph == 1
colorstring = 'ymcrgbk';
h = figure(1); cla;
hold on
for i = 1:7
  plot(diffs(i, 2:end), 'Color', colorstring(i))
end

saveas(h,sprintf('D:\\vot7\\rests\\jpgs\\result_%s_diffs_all.jpg', tname))

for i = 1:7
    h = figure(1); cla;
    hold on
    xlabel(sprintf('%d', i),'FontSize',12, 'Color', colorstring(i));  
    plot(diffs(i, 2:end), 'Color', colorstring(i))
    saveas(h,sprintf('D:\\vot7\\rests\\jpgs\\result_%s_diffs_%d.jpg', tname, i));
end

h = figure(1); cla;
hold on
for i = 1:7
  plot(means(i, 2:end), 'Color', colorstring(i))
end

saveas(h,sprintf('D:\\vot7\\rests\\jpgs\\result_%s_means_all.jpg', tname));

for i = 1:7
    h = figure(1); cla;
    hold on
    xlabel(sprintf('%d', i),'FontSize',12, 'Color', colorstring(i));  
    plot(means(i, 2:end), 'Color', colorstring(i))
    saveas(h,sprintf('D:\\vot7\\rests\\jpgs\\result_%s_means_%d.jpg', tname, i));
end
end


mean(means(:, 2:end),2);
mean(diffs(:, 2:end),2);

% ouput{:}
% output{iii} = ouput;

end

mns
output1'
output2'
output3'

opr1 = [[1:7]' output1];
opr2 = [[1:7]' output2];
opr3 = [[1:7]' output3];

for iiii=2:size(opr1,1)+1
    mns = sortrows(opr1,iiii);
    mns(:,iiii) = 1:7;
    mns = sortrows(opr1,1);
end
for iiii=2:size(opr2,1)+1
    mns = sortrows(opr2,iiii);
    mns(:,iiii) = 1:7;
    mns = sortrows(opr2,1);
end
for iiii=2:size(opr2,1)+1
    mns = sortrows(opr2,iiii);
    mns(:,iiii) = 1:7;
    mns = sortrows(opr2,1);
end


% sonuclari genislet
%  saliya weight hesabi yalinnhaldeki featureler icin 3
% weight grafigi ciz zamana bagli