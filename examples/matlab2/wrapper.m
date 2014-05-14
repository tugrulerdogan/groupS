function wrapper()
%% VOT integration example for MeanShift tracker

% *************************************************************
% VOT: Always call exit command at the end to terminate Matlab!
% *************************************************************
onCleanup(@() 1+2 );

% *************************************************************
% VOT: Set random seed to a different value every time.
% *************************************************************

load(fullfile(fileparts(mfilename('fullpath')), 'cur.mat'));
% cd(currentFolder);
RandStream.setGlobalStream(RandStream('mt19937ar', 'Seed', sum(clock)));

tracker_directory = fullfile(fileparts(mfilename('fullpath')), 'tracker');
% rmpath(tracker_directory);
addpath(tracker_directory);

% **********************************
% VOT: Read input data
% **********************************
[images, region] = vot_initialize();

%% Initialize tracker variables
index_start = 1;
% Similarity Threshold
f_thresh = 0.16;
% Number max of iterations to converge
max_it = 5;

count = size(images,1);

im0 = imread(images{1});
height = size(im0,1);
width = size(im0,2);

results = zeros(count, 4);

results(1, :) = region;

T = imcrop(im0, region);
x = region(1);
y = region(2);
W = region(3);
H = region(4);

x = x + W/2;
y = y + H/2;


addpath('tracker');
trackparam;                                                       % initial position and affine parameters
opt.tmplsize = [32 32];                                           % [height width]
sz = opt.tmplsize;
n_sample = opt.numsample;

p = [x, y, W, H, 0]

param0 = [p(1), p(2), p(3)/sz(2), p(5), p(4)/p(3), 0];
p0 = p(4)/p(3);
param0 = affparam2mat(param0);
param = [];
param.est = param0';

num_p = 50;                                                         % obtain positive and negative templates for the SDC
num_n = 200;
[dataPath,dname,dext] = fileparts(images{1})
[A_poso A_nego] = affineTrainG(dataPath, sz, opt, param, num_p, num_n, forMat, p0);        
A_pos = A_poso;
A_neg = A_nego;                                                     

patchsize = [6 6];                                                  % obtain the dictionary for the SGM
patchnum(1) = length(patchsize(1)/2 : 2: (sz(1)-patchsize(1)/2));
patchnum(2) = length(patchsize(2)/2 : 2: (sz(2)-patchsize(2)/2));
Fisize = 50;
[Fio patcho] = affineTrainL(dataPath, param0, opt, patchsize, patchnum, Fisize, forMat);
Fi = Fio;    

% temp = importdata([dataPath 'datainfo.txt']);
num = count;%temp(3);
paramSR.lambda2 = 0;
paramSR.mode = 2;
alpha_p = zeros(Fisize, prod(patchnum), num);
result = zeros(num, 6);
results = zeros(num, 10);

%%******************************************* Do Tracking *********************************************%%

for f = 1:num
    f
    img_color = imread([dataPath '\\' sprintf('%8.8d',f) forMat]);
    if size(img_color,3)==3
        img	= rgb2gray(img_color);
    else
        img	= img_color;
    end
    
    %%----------------- Sparsity-based Discriminative Classifier (SDC) ----------------%%
    gamma = 0.4;
    
    [wimgs Y param] = affineSample(double(img), sz, opt, param);    % draw N candidates with particle filter
    
    YY = normVector(Y);                                             % normalization
    AA_pos = normVector(A_pos);
    AA_neg = normVector(A_neg);
    
    P = selectFeature(AA_pos, AA_neg, paramSR);                     % feature selection
    
    YYY = P'*YY;                                                    % project the original feature space to the selected feature space
    AAA_pos = P'*AA_pos;
    AAA_neg = P'*AA_neg;
    
    paramSR.L = length(YYY(:,1));                                   % represent each candidate with training template set
    paramSR.lambda = 0.01;
    beta = mexLasso(YYY, [AAA_pos AAA_neg], paramSR);
    beta = full(beta);
    
    rec_f = sum((YYY - AAA_pos*beta(1:size(AAA_pos,2),:)).^2);      % the confidence value of each candidate
    rec_b = sum((YYY - AAA_neg*beta(size(AAA_pos,2)+1:end,:)).^2);
    con = exp(-rec_f/gamma)./exp(-rec_b/gamma);                     

%     %%----------------- Sparsity-based Generative Model (SGM) ----------------%%
%     yita = 0.01;
%     
%     patch = affinePatch(wimgs, patchsize, patchnum);                % obtain M patches for each candidate
%     
%     Fii = normVector(Fi);                                           % normalization
%     
%     if f==1                                                         % the template histogram in the first frame and before occlusion handling
%         xo = normVector(patcho);
%         paramSR.L = length(xo(:,1));
%         paramSR.lambda = 0.01;
%         alpha_q = mexLasso(xo, Fii, paramSR);
%         alpha_q = full(alpha_q);
%         alpha_qq = alpha_q;
%     end
%     
%     temp_q = ones(Fisize, prod(patchnum));
%     sim = zeros(1,n_sample);
%     b = zeros(1,n_sample);
%     
%     for i = 1:n_sample
%         x = normVector(patch(:,:,i));                               % the sparse coefficient vectors for M patches 
%         paramSR.L = length(x(:,1));      
%         paramSR.lambda = 0.01;
%         alpha = mexLasso(x, Fii, paramSR);
%         alpha = full(alpha);
%         alpha_p(:,:,i) = alpha;      
%         
%         recon = sum((x - Fii*alpha).^2);                            % the reconstruction error of each patch
%         
%         thr = 0.04;                                                 % the occlusion indicator            
%         thr_lable = recon>=thr;   
%         temp = ones(Fisize, prod(patchnum));
%         temp(:, thr_lable) = 0;        
%         
%         p = temp.*abs(alpha);                                       % the weighted histogram for the candidate
%         p = reshape(p, 1, numel(p));
%         p = p./sum(p);
%         
%         temp_qq = temp_q;                                           % the weighted histogram for the template
%         temp_qq(:, thr_lable) = 0;
%         q = temp_qq.*abs(alpha_qq);     
%         q = reshape(q, 1, numel(q));
%         q = q./sum(q);
%         
%         lambda_thr = 0.00003;                                       % the similarity between the candidate and the template
%         a = sum(min([p; q]));
%         b(i) = lambda_thr*sum(thr_lable);
%         sim(i) = a + b(i);
%     end
    
    %%----------------- Collaborative Model ----------------%%
    likelihood = con;%.*sim;
    [v_max,id_max] = max(likelihood);
    
    
    param.est = affparam2mat(param.param(:,id_max));
    result(f,:) = param.est';
    displayResult_sf;                                               % display the tracking result in each frame
    
    results(f,:) =  corners(:)
%     rect= round(aff2image(param.est', sz_T));
%     inp	= reshape(rect,2,4);
%     
%     results(f,1)=round(mean(inp(2,:)));
%     results(f,2)=round(mean(inp(1,:)));
%     results(f,4)=inp(1,4)-inp(1,1);
%     results(f,3)=inp(2,4)-inp(2,1);
    
    
    save('D:\vot7\rests\results.mat', 'results')
    
    %%----------------- Update Scheme ----------------%%
    upRate = 5;
    if rem(f, upRate)==0
%         [A_neg alpha_qq] = updateDic(dataPath, sz, opt, param, num_n, forMat, p0, f, abs(alpha_q), abs(alpha_p(:,:,id_max)), (b(id_max)/lambda_thr)/prod(patchnum));
%         [A_neg alpha_q] = updateDic(dataPath, sz, opt, param, num_n, forMat, p0, f, ones(50,196), abs(alpha_p(:,:,id_max)), (b(id_max)/lambda_thr)/prod(patchnum));

    end
end



for t = 1:count
    afnv	= tracking_res(:,t)';

    rect= round(aff2image(afnv', sz_T));
    inp	= reshape(rect,2,4);
    
%     results(t,1)=inp(1,1)+H/2;
%     results(t,2)=inp(2,1)+W/2;
%     results(t,3)=inp(1,4)-inp(1,1);
%     results(t,4)=inp(2,4)-inp(2,1);
    
    results(t,1)=round(mean(inp(2,:)));
    results(t,2)=round(mean(inp(1,:)));
    results(t,4)=inp(1,4)-inp(1,1);
    results(t,3)=inp(2,4)-inp(2,1);

    results
 
end

% results=[1,2,3,4]

% save('d:\l1.mat', 'results');
csvwrite(output_file, results);
% vot_deinitialize(results);

