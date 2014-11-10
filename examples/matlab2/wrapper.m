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

% load('D:\vot7\rests\gists.mat')

itrts = {[1] [2] [3] [1, 2] [1, 3] [2, 3] [1, 2, 3]}
% itrts = {[1]}% [2] [3] [1, 2] [1, 3] [2, 3] [1, 2, 3]}
itrts = {[3]}

lambd = 6;

for iterations = itrts
iterations = cell2mat(iterations)


%% Initialize tracker variables
index_start = 1;
% Similarity Threshold
f_thresh = 0.16;
% Number max of iterations to converge
max_it = 5;


% sonuclari yeni normalizasyona gore yeniden kostur
% gamma degerini otomatik uret otalama recf, recb, recf-recb degerelerini
%      incele nasil bir gamma sececigmizi anlatacak 10 sekans icin NxT
%      kadar veri icin

%  whitening gama icin yada baska makina ogrenmesi



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

quality_cons = 0.01;
wight = ones(1,3)*(1/size(iterations,2));%)[1/3 1/3 1/3];

% beklentimi tekil ornegin iyi calistigi orneklerde gozlemel
% % % % % % % % % % normalizasyon probleminden dolayi yaklasik recons sonularini coz
%  recf recb cizdir min max ortalama, farklarini cizdir 
% % % % % % % % % % % % % % % % % % %  color tracker mutlaka calsmali
% onder hoca mail


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

param.imageSize = 32;
param.orientationsPerScale = [8 8 8 8]; % number of orientations per scale (from HF to LF)
param.numberBlocks = 4;
param.fc_prefilt = 4

num_p = 50;                                                         % obtain positive and negative templates for the SDC
num_n = 200;
[dataPath,dname,dext] = fileparts(images{1})
[bf,tname,bf] = fileparts(dataPath)
[A_pos A_neg] = affineTrainG(dataPath, sz, opt, param, num_p, num_n, forMat, p0); 

[dataPath '00000001' forMat]
img_color = imread([dataPath '\\' sprintf('%8.8d',1) forMat]);


[A_posr A_negr] = affineTrainG_RGB(dataPath, sz, opt, param, num_p, num_n, forMat, p0, img_color(:,:,1)); 
[A_posg A_negg] = affineTrainG_RGB(dataPath, sz, opt, param, num_p, num_n, forMat, p0, img_color(:,:,2)); 
[A_posb A_negb] = affineTrainG_RGB(dataPath, sz, opt, param, num_p, num_n, forMat, p0, img_color(:,:,3)); 

A_posRGB = reshape([A_posr A_posg A_posb], [size(A_posr),3]);
A_negRGB = reshape([A_negr A_negg A_negb], [size(A_negr),3]);


A_posRGB = reshape([A_pos A_pos A_pos], [size(A_posr),3]);
A_negRGB = reshape([A_neg A_neg A_neg], [size(A_negr),3]);
    
figure(6);
imshow(uint8(mMergeVect(A_posRGB, [32,32,3])));



% [dic_pos{2}  param] = vec2gist(dic_pos{1}, param);
% [dic_neg{2}  param] = vec2gist(dic_neg{1}, param);
% 
% [dic_pos{3} param] = vec2hist(dic_pos{1}, param);
% [dic_neg{3} param] = vec2hist(dic_neg{1}, param);


vecfunctions = {@vec2main @vec2gist, @vec2hist};
imfunctions = {@im2main @im2gist @im2hist};

for i=iterations%2:size(vecfunctions,2)
    
   [dic_pos{i},  param] = vecfunctions{i}(A_pos, param);
   [dic_neg{i},  param] = vecfunctions{i}(A_neg, param);
   dic_pos{i} = normVector(dic_pos{i});
   dic_neg{i} = normVector(dic_neg{i});

    
    
end

   [dic_pos{3},  param] = vec2histcolor(A_posRGB, param);
   [dic_neg{3},  param] = vec2histcolor(A_negRGB, param);
   dic_pos{3} = normVector(dic_pos{3});
   dic_neg{3} = normVector(dic_neg{3});




% dic_pos{1} = dic_pos{2};
% dic_neg{1} = dic_neg{3};

% A_pos_gist = dic_pos{2};
% A_neg_gist = dic_neg{2};
% 
% A_pos_hist = dic_pos{3};
% A_neg_hist = dic_neg{3};

% A_pos = dic_pos{1};
% A_neg = dic_neg{1};                                                     

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
recs = zeros(3,10,400);


% !!! PERSEMBE !!!
% 27/10/2014 kodalama kodlaam persembe gunune optimum degerler adaptive deger;ler
% icin. her optimum degerler icin tablo.

%  color histogram hesaplamadan once, piksel degerlerini 0-1 e cek. r g b histogramlarinin arka arkaya concat et. 
% sonuclarinin 0-255 araligina cekmeliyiz




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
    
    [wimgs Y param] = affineSample(double(img), sz, opt, param,0);    % draw N candidates with particle filter
    
    param2 = param;
   
    [wimgr Y2 param2, o_new] = affineSample(double(img_color(:,:,1)), sz, opt, param, 0);
    [wimgg Y2 param2, o_new] = affineSample(double(img_color(:,:,2)), sz, opt, param, o_new);
    [wimgb Y2 param2, o_new] = affineSample(double(img_color(:,:,3)), sz, opt, param, o_new);
    
    wimgRGB = reshape([wimgr wimgg wimgb], [size(wimgr),3]);
    wimgRGB = reshape([wimgs wimgs wimgs], [size(wimgr),3]);

    
    figure(7);
    imshow(uint8(mMergeIm(wimgRGB, [32,32,3])));
    
%     Bu hafta yapilanlar:
%       seyrek kodlamaya pozitif olma cons getirildi
%       seyrek kod ciktisi normalize edildi
%       debug cizimleri yapidldi.


% 
    
    for i = iterations% 2:size(imfunctions,2)
        [particleforms{i} param] = imfunctions{i}(wimgs, param);
    end
    
    [particleforms{3} param] = im2histcolor(wimgRGB, param);
%     particleforms{3}(:,1) = dic_pos{3}(:,1);
    
% %     Y = gists;
%     YYY_gist = particleforms{2};
%     YYY_hist = particleforms{3};
    
    YY = normVector(Y);                                             % normalization
    AA_pos = normVector(A_pos);
    AA_neg = normVector(A_neg);
    
%     P = selectFeature(AA_pos, AA_neg, paramSR);                     % feature selection
%     
%     YYY = P'*YY;                                                    % project the original feature space to the selected feature space
%     AAA_pos = P'*AA_pos;
%     AAA_neg = P'*AA_neg;
    
    YYY = YY;                                                    % project the original feature space to the selected feature space
    AAA_pos = AA_pos;
    AAA_neg = AA_neg;
    
%     dic_pos{2} = A_pos_gist;
%     dic_neg{2} = A_neg_gist;
%     
%     dic_pos{3} = A_pos_hist;
%     dic_neg{3} = A_neg_hist;
    
    paramSR.L = length(YYY(:,1));                                   % represent each candidate with training template set
    paramSR.lambda = 0.01;
%     beta = mexLasso(YYY, [AAA_pos AAA_neg], paramSR);
%     beta = full(beta);
%     
%     rec_f = sum((YYY - AAA_pos*beta(1:size(AAA_pos,2),:)).^2);      % the confidence value of each candidate
%     rec_b = sum((YYY - AAA_neg*beta(size(AAA_pos,2)+1:end,:)).^2);
%     con = exp(-rec_f/gamma)./exp(-rec_b/gamma);    
%     
% %     beta = mexLasso(particleforms{2}, [dic_pos{2} dic_neg{2}], paramSR);
% %     beta = full(beta);
% %     
% %     rec_f = sum((particleforms{2} - dic_pos{2}*beta(1:size(dic_pos{2},2),:)).^2);      % the confidence value of each candidate
% %     rec_b = sum((particleforms{2} - dic_neg{2}*beta(size(dic_pos{2},2)+1:end,:)).^2);
% %     con_gist = exp(-rec_f/gamma)./exp(-rec_b/gamma); 
% %     
% %     beta = mexLasso( particleforms{3}, [dic_pos{3} dic_neg{3}], paramSR);
% %     beta = full(beta);
% %     
% %     rec_f = sum(( particleforms{3} - dic_pos{3}*beta(1:size(dic_pos{3},2),:)).^2);      % the confidence value of each candidate
% %     rec_b = sum(( particleforms{3} - dic_neg{3}*beta(size(dic_pos{3},2)+1:end,:)).^2);
% %     con_hist = exp(-rec_f/gamma)./exp(-rec_b/gamma); 
    particleforms{1} = YYY;
    dic_pos{1} = AAA_pos;
    dic_neg{1} = AAA_neg;
    
    paramSR.numThreads=-1;    
%     paramSR.pos=true;

    for i=iterations%1:size(vecfunctions,2)
        beta = mexLasso( particleforms{i}, [dic_pos{i} dic_neg{i}], paramSR);
        beta = full(beta);
        beta = normc(beta);

        rec_f = sum(abs( particleforms{i} - dic_pos{i}*beta(1:size(dic_pos{i},2),:)).^2);      % the confidence value of each candidate
        rec_b = sum(abs( particleforms{i} - dic_neg{i}*beta(size(dic_pos{i},2)+1:end,:)).^2);
        if f<=10
            recs(i,f,:)=rec_f-rec_b;
        end
        mm = max(max(rec_f), rec_b);
% %         mm(1) = 6000;
% %         conn{i} = exp(-rec_f/mm(1))./exp(-rec_b/mm(1));
% % %          conn{i} = exp(-rec_f/gamma)./exp(-rec_b/gamma); 
% % color histogramin degerlerini normalize et hitogramda toplam piksel
% % sayisin aoranlayarak
% % gamma degerlerini max degerlerinin yarisini almayi dene
% % sonucu hacir olanlardan color histogram normalizasyonun fark yaratma
% % yuzdesine bak
% % o haftalik yapilan raporu, palnli ol artik son cagri
% % mail at color histogram karsilstirmasinin
%         gammabuf = gamma;
%         if i == 3
%            gammabuf = 6000 ;
%         end
% % %           conn{i} = exp(-abs(rec_f-rec_b)/gammabuf);%./exp(-rec_b/gammabuf);
% %           conn{i} = exp(-rec_f/gammabuf)./exp(-rec_b/gammabuf);


          gammabuf = 500 ;

%           conn{i} = exp(-rec_f/(mm(1)/2))./exp(-rec_b/(mm(1)/2));
          conn{i} = exp(-rec_f/gammabuf)./exp(-rec_b/gammabuf);


          
          conn{i} = conn{i}/max(conn{i});

          
          
%             colorstring = 'ymcrgbk';
%             h = figure(3); cla;
%             hold on
%             
%               plot(rec_f(:), 'Color', colorstring(1)); %yellow
%               plot(rec_b(:), 'Color', colorstring(2));  %pink
%               plot(conn{i}(:), 'Color', colorstring(3)); %green
%             
% %             saveas(h,sprintf('D:\\vot7\\rests\\rec\\result_%s_%d.jpg', tname,f))
%             saveas(h,sprintf('D:\\vot7\\rests\\rec\\result_%s_%s_frame%d.jpg',tname,strtrim(sprintf('%d',iterations)),f));
            
         
    end
% 
%     if f==11
%        
%         recs
%         
%     end
%     con_sum  = con;
    repeat = true;
    for i=iterations%1:size(vecfunctions,2)-1 % sondaki color histogram sonuclari devredisi -1 ile
%     con_sum  = con + con_gist;% + con_hist;
        if repeat==true
            con_sum  = wight(i)*conn{i};
            repeat = false;
        else
            con_sum = con_sum + wight(i)*conn{i};
        end
    end

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
%     likelihood = con;%.*sim;
%     [v_max,id_max] = max(likelihood);
    
% %     Toplam recons olacak ayari ayri alip ortalama yerine
% %     yeni partikellar da ortak olasilik uzerinden turetiledcek
% %     hog    vl-feat
% %     color histogram
% %     
% %     
% %     l1 a gore colora gore hoga gore ayri ayri hepsi berarber tabloda haftaya
% %     
% %     frame frame cizim yap 722 proje sonuclarini debug icin
    
    likelihood = con_sum;%.*sim;
    [v_max,id_max] = max(likelihood);
    
    
    param.est = affparam2mat(param.param(:,id_max));
    
    
%     s_mat = repmat(param.est,1,size(param.param,2));
%     
%     eq_dif = param.param - s_mat;
%     eq_dif_sq = eq_dif' * eq_dif;
%     eq_dif_sq_vec = diag(eq_dif_sq);
%     nrm = sum(eq_dif_sq_vec);
%     
    qlt = zeros(3,1);
%     
%     for i=iterations
%         qlt(i) = nrm/sum((conn{i}/max(conn{i}))' .* eq_dif_sq_vec);
%     end
%     
%     
%     
%     
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     dfff=0;
%     dfff_wgt = zeros(3,1);
%     
%     for i=iterations
%     conn{i} = (conn{i}/max(conn{i}));
%     end
%     
%     for ll = 1:size(param.param,2)
%        
%         dfff = dfff + norm(abs(param.param(:,ll) - param.est),lambd);
%         for i=iterations
%             dfff_wgt(i) = dfff_wgt(i) + conn{i}(ll).* norm(abs(param.param(:,ll) - param.est),lambd);
%             % 12 sekans
% %             latekteki tablolarla, avarage ranklar ile 
% % tek eksik kaliyorsa adaptive olsun, 
% % normalizasyondan sonra p ussunu al
% % kodu birebir vcevir
%         end
%         
%     end
%     
%     for i=iterations
%      qlt(i) = (dfff/dfff_wgt(i)) % sparse codingden donen degeri max degere gore normalize ettigim icin zaten max deger 1 olasilikli oluyor diye carpmadim
%     end
%      
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     
%     
%     
%     if f == 1
%        
%         old_qlt = qlt;
%         
%     else
%         
%         d_sum = sum(old_qlt - qlt);
%         for i=iterations
%            
%             wight(i) = wight(i) + (quality_cons)*qlt(i)/d_sum;
%             
%         end
%         
%     end
    
%     wight = [1/3 1/3 1/3];
    
qsum1 = 0;
qsum2 = 0;
qsum = 0;
difff = 0;

for b=iterations
%  for(b = 0; b < n; b++)     
%     {
        qsum1= 0;
        qsum2= 0;
        
        for i = 1:size(param.param,2)
%         for(i=1;i<nParticles;i++)
%         {
             
            difff = difff + norm(abs(param.param(:,i) - param.est),lambd);

            qsum1 = qsum1 + (1.0/size(param.param,2))*(difff^lambd);%pow(diff,4);
%             qsum2 += (exp(-1*scores[i][b]*scores[i][b]/0.02))*pow(diff,4);
            qsum2 = qsum2 + conn{b}(i)*(difff^lambd);%pow(difff,lambd);
            
%         }
        end
        
%         qsum1 
%         qsum2
        qlt(b) = qsum1/(qsum2);%exp(-1*score1[b]*score1[b]/0.02)*qsum1/(qsum2);
        qsum = qsum + qlt(b);
%     }
end


delta_t = 0.1;

for b=iterations
%     for(b = 0; b < n; b++)
%     {
        qsum
        
        if (qsum>0)
%         {
            qlt(b) = qlt(b) /qsum;
            wight(b) = (1-delta_t) * wight(b) + delta_t * qlt(b);
%         }
        end
%         else
%             newWts[Index2D(0,b,1,n)] = Wts[b];
%     }
end

% wight
        
    
    
%     param.est = (param.est + affparam2mat(param.param(:,id_max_gist)))/2;
    result(f,:) = param.est';
    displayResult_sf;                                               % display the tracking result in each frame
    
    results(f,:) =  corners(:);
%     rect= round(aff2image(param.est', sz_T));
%     inp	= reshape(rect,2,4);
%     
%     results(f,1)=round(mean(inp(2,:)));
%     results(f,2)=round(mean(inp(1,:)));
%     results(f,4)=inp(1,4)-inp(1,1);
%     results(f,3)=inp(2,4)-inp(2,1);
    
    
%     save('D:\vot7\rests\results.mat', 'results')
    save(sprintf('D:\\vot7\\rests\\result_%s_%s.mat',tname,strtrim(sprintf('%d',iterations))), 'results');
    
    %%----------------- Update Scheme ----------------%%
    upRate = 5;
    if rem(f, upRate)==0
%         [A_neg alpha_qq] = updateDic(dataPath, sz, opt, param, num_n, forMat, p0, f, abs(alpha_q), abs(alpha_p(:,:,id_max)), (b(id_max)/lambda_thr)/prod(patchnum));
%         [A_neg alpha_q] = updateDic(dataPath, sz, opt, param, num_n, forMat, p0, f, ones(50,196), abs(alpha_p(:,:,id_max)), (b(id_max)/lambda_thr)/prod(patchnum));

    end
end

end

% for t = 1:count
%     afnv	= tracking_res(:,t)';
% 
%     rect= round(aff2image(afnv', sz_T));
%     inp	= reshape(rect,2,4);
%     
% %     results(t,1)=inp(1,1)+H/2;
% %     results(t,2)=inp(2,1)+W/2;
% %     results(t,3)=inp(1,4)-inp(1,1);
% %     results(t,4)=inp(2,4)-inp(2,1);
%     
%     results(t,1)=round(mean(inp(2,:)));
%     results(t,2)=round(mean(inp(1,:)));
%     results(t,4)=inp(1,4)-inp(1,1);
%     results(t,3)=inp(2,4)-inp(2,1);
% 
%     results;
%  
% end
% end
% 
% % results=[1,2,3,4]
% 
% % save('d:\l1.mat', 'results');
% csvwrite(output_file, results);
% vot_deinitialize(results);

