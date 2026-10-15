%--------------------------------------------------------------------------
%% load data: cs-qalas on XA30
%--------------------------------------------------------------------------

addpath(genpath('utils'))

data_path_org = '/path_to_twix';
data_path = '/path_to_all_data';

filename = 'twix_file_name';

dt = mapVBVD2([data_path_org, filename, '.dat'], 'removeOS');

kspace = dt{end}.image();

% sqzDims: {'Col'  'Cha'  'Lin'  'Par'  'Set'}
kspace = sq(kspace);


%--------------------------------------------------------------------------
% patref scan
%--------------------------------------------------------------------------

ref = dt{end}.refscan();

% sqzDims: {'Col'  'Cha'  'Lin'  'Par'  'Set'}
ref = sq(ref);

ref = permute(ref, [1,3,4,2]);


img_ref = ifft3call(ref);

imagesc3d2(rsos(img_ref,4), s(img_ref)/2, 6, [0,0,0], [0,3e-3])


% %--------------------------------------------------------------------------
%% coil compression
%--------------------------------------------------------------------------

% num_chan = 24;  % num channels to compress to
% 
% [ref_svd, cmp_mtx] = svd_compress3d(ref, num_chan, 1);
% 
% rmse(rsos(ref_svd,4), rsos(ref,4))
% 
% N = size(sq(kspace(:,1,:,:,1)));
% num_eco = size(kspace,5);
% 
% kspace_svd = zeross([N,num_chan,num_eco]);
% 
% for t = 1:size(kspace,5)
%     disp(num2str(t))
% 
%     ksp = kspace(:,:,:,:,t);
%     ksp = permute(ksp, [1,3,4,2]);
% 
%     kspace_svd(:,:,:,:,t) = svd_apply3d(ksp, cmp_mtx);
% end
% 
% rmse(rsos(kspace_svd,4), rsos(kspace,2))
% 
% clear kspace ref ksp
% for  t = 1:size(kspace,5)
%     img_before(:,:,:,:,t) = ifft3call(permute(sq(kspace(:,:,:,:,t)),[1,3,4,2,5]));
% end
%% GCC
% 
num_chan = 20;  % num channels to compress to
dim = 1;            % readout axis


% estimate gcc matrix
gccmtx = calcGCCMtx( padarray( ref,[(size(kspace,1)-size(ref,1))/2 0 0 0]), dim);

gccmtx_aligned = alignCCMtx(gccmtx(:,1:num_chan,:));

clear CCDATA_aligned
for cc = 1:size(kspace,5)
% apply gcc matrix
    ksp = kspace(:,:,:,:,cc);
    ksp = permute(ksp, [1,3,4,2]);
    CCDATA_aligned(:,:,:,:,cc) = CC(sq(ksp), dim, gccmtx_aligned);
end

% apply the SAME aligned GCC matrix to calibration data
ref_svd  = CC(padarray( ref,[(size(kspace,1)-size(ref,1))/2 0 0 0]),  dim, gccmtx_aligned);
% for  t = 1:size(kspace,5)
%     img_gcc(:,:,:,:,t) = ifft3call(CCDATA_aligned(:,:,:,:,t));
% end

% 
% rmse_gcc = rmse(rsos(img_gcc,4), rsos(img_before,4))

% rmse(rsos(CCDATA_aligned,4), rsos(kspace,2))    

kspace_svd = CCDATA_aligned;
clear CCDATA_aligned
%--------------------------------------------------------------------------
%% permute k-space
%--------------------------------------------------------------------------

% zero pad to even size
size_data = size(kspace_svd(:,:,:,1,1));

kspace_svd = padarray( kspace_svd, rem(size_data,2), 'post' );

% make Nx=Ny
% kspace_svd = padarray( kspace_svd, [0,1,0] ); 


%--------------------------------------------------------------------------
%% interpolate patref by zero padding to the high res matrix size
%--------------------------------------------------------------------------

size_patref = size(ref_svd(:,:,:,1,1))
size_data = size(kspace_svd(:,:,:,1,1));


patref_pad = padarray( ref_svd, [size_data-size_patref, 0, 0, 0]/2 );

img_patref_pad = ifft3c(patref_pad);

imagesc3d2( rsos(img_patref_pad,4), s(img_patref_pad)/2, 7, [-90,-90,90], [0,1e-4]/2)

img_patref_pad = single(img_patref_pad);

clear patref_pad ref_svd img_ref


%--------------------------------------------------------------------------
%% calculate sens map using ESPIRiT: parfor
%--------------------------------------------------------------------------

num_acs = min(size_patref);
kernel_size = [6,6];
eigen_thresh = 0.7;

receive = zeross(size(kspace_svd(:,:,:,:,1)));


delete(gcp('nocreate'))
c = parcluster('local');    

total_cores = c.NumWorkers;  
% parpool(ceil(total_cores/2))
parpool(4)
    
tic
parfor slc_select = 1:s(img_patref_pad,1)     
    disp(num2str(slc_select))
    
    [maps, weights] = ecalib_soft( fft2c( sq(img_patref_pad(slc_select,:,:,:)) ), num_acs, kernel_size, eigen_thresh );

    receive(slc_select,:,:,:) = permute(dot_mult(maps, weights >= eigen_thresh ), [1,2,4,3]);
end 
toc

delete(gcp('nocreate'))

clear img_patref_pad 

rec = abs(receive) .* exp(1i * angle( receive .* repmat(conj(receive(:,:,:,1)), [1,1,1,num_chan]) ));

save([data_path, 'rec_svd_', filename, '.mat'], 'rec', '-v7.3')

clear receive

%--------------------------------------------------------------------------
%% load sensitivities and sense recon
%--------------------------------------------------------------------------

% tic
%     load([data_path, 'rec_svd_', filename, '.mat'])
%     load([data_path, 'img_sense_', filename, '.mat'])
% toc


%--------------------------------------------------------------------------
%% save data for zs-ssl
%--------------------------------------------------------------------------

kspace_svd_2d = ifftc(kspace_svd,1);

for nn = 1:size(kspace_svd,1)
    tic
    fprintf('slice: %d/%d... ',nn,size(kspace_svd,1));

    kspace      = sq(kspace_svd_2d(nn,:,:,:,:));
    sens_maps   = sq(rec(nn,:,:,:));
    mask        = single(sq(kspace(:,:,1,:))~=0);

    save([data_path,'data_20ch_GCC_fa4/','data_s',num2str(nn),'.mat'],'kspace','sens_maps','mask');
    toc
end


%--------------------------------------------------------------------------
%% lsqr 3d-Sense recon [without CS]
%--------------------------------------------------------------------------

lsqr_iter = 100;        % max num of iterations
lsqr_tol = 1e-3;        % tolerance to terminate iterations

m3d = kspace_svd~=0;
img_sense = zeros([size(kspace_svd,1), size(kspace_svd,2), size(kspace_svd,3), size(kspace_svd,5)]);
        

param = [];
param.N = size(m3d(:,:,:,1,1));
param.num_chan = size(m3d,4);
param.sens = rec;

param.lambda = 1e-2;        % L2 regularization amount

    
for t = 1:size(kspace_svd,5)
    disp(['eco: ', num2str(t)])

    param.m3d = m3d(:,:,:,:,t);         
    kspace_coils = kspace_svd(:,:,:,:,t);
    
    res = lsqr(@apply_sense_tikc3d, cat(1, kspace_coils(:), zeros(prod(param.N),1)), lsqr_tol, lsqr_iter, [], [], [], param);  

    Res = reshape(res, param.N);
   
    img_sense(:,:,:,t) = Res;
end


for t = 1:size(img_sense,4)
    imagesc3d2( img_sense(:,:,:,t), s(img_sense)/2, t, [-90,-90,90], [0,3e-4]), setGcf(.5)
    imagesc3d2( angle(img_sense(:,:,:,t)), s(img_sense)/2, t+10, [-90,-90,90], [-pi,pi]), setGcf(.5)
end

% save as .mat
save([data_path, 'img_sense_', filename, '.mat'], 'img_sense', '-v7.3')

 
%--------------------------------------------------------------------------
%% 3d-Sense recon with wavelet and TV regularization
%--------------------------------------------------------------------------

% check if wavelet toolbox exists -> use only TV if not present
try
    v = ver;
catch ME
    v = struct();
    v.Name = 'Wavelet Toolbox';
end
has_wavelet = any(strcmp(cellstr(char(v.Name)), 'Wavelet Toolbox'));

 
mtx_size = size(m3d(:,:,:,1,1));
size_powerof2 = nextpow2( mtx_size );

Nwav = 2.^size_powerof2;        % zero padded mtx size for wavelet transform to the next power of 2
 
param = init;

% 3d-wavelet parameters:
param.wav_scale = min(size_powerof2);
param.wav_type = 'db1';
param.wav_mode = 'per';


param.TV = TV3D;
 
param.TVWeight = 3e-7;                                  % 3D TV penalty 
param.WavWeight = has_wavelet * param.TVWeight;         % 3D Wavelet penalty: set to 0 if wavelet toolbox not present. use same lambda as TV


param.num_chan = size(kspace_svd,4);
param.wav_pad = (Nwav - mtx_size) / 2;

param.Receive = rec;
param.Ct = conj(param.Receive);


param.Itnlim = 10;          % num max inner iters
num_outer_iters = 5;        % num outer iters

param.tol = 1e-2;           % tolerance to terminate inner loop
param.pNorm = 1;            % use L1 constraint

img_cs = zeros([size(kspace_svd,1), size(kspace_svd,2), size(kspace_svd,3), size(kspace_svd,5)]);

tic

for t = 1:size(kspace_svd,5)
    disp(['contrast: ', num2str(t)])
    
    param.data = kspace_svd(:,:,:,:,t);
    param.M3d = m3d(:,:,:,:,t);         

    % sense initialization
    res = img_sense(:,:,:,t);

    for n = 1:num_outer_iters
        res = fnlCg_ics_sense(res, param);

        imagesc3d2(res, mtx_size/2, t+20, [0,0,0], [0,1e-4], 0, ['CS recon iter: ', num2str(n)]), setGcf(.5)
    end
    
    img_cs(:,:,:,t) = res;
end
toc
 


for t = 1:size(img_sense,4)
    imagesc3d2( img_sense(:,:,:,t), s(img_sense)/2, t, [180,-0,0], [0,3e-5]), setGcf(.5)
%     imagesc3d2( angle(img_sense(:,:,:,t)), s(img_sense)/2, t+10, [-0,-0,0], [-pi,pi]), setGcf(.5)

    imagesc3d2( img_cs(:,:,:,t), s(img_sense)/2, t+100, [180,-0,0], [0,5e-5]), setGcf(.5)
%     imagesc3d2( angle(img_cs(:,:,:,t)), s(img_sense)/2, t+110, [-0,-0,0], [-pi,pi]), setGcf(.5)
end

save([data_path, 'img_cs_', filename, '.mat'], 'img_cs', '-v7.3')

  
%--------------------------------------------------------------------------
%% save using nifti header info
%--------------------------------------------------------------------------

% data_path = '/autofs/space/marduk_001/users/berkin/2024_10_16_bay2_qalas_550um_invivo/';
filename = 'yuting_dicom_qalas_vnav_fa4_450um_r2x2_Vref190_Inv430_20260805181022_32';


dt_nifti = load_nifti([data_path, 'nifti/yuting_dicom_qalas_vnav_fa4_450um_r2x2_Vref190_Inv430_20260805181022_32.nii']);

imagesc3d2( dt_nifti.vol(:,:,:,1), s(dt_nifti.vol)/2, 1, [90,90,90], [0,500]), setGcf(.5)


% sense recon 
% load([data_path, 'img_sense_', filename, '.mat'])

% sag orientation
im_sense = circshift(flip(flip(permute(img_sense,[2,1,3,4]),2),1),[1,0,-1]);
% im_sense = circshift(flip(flip(permute(img_sense(:,:,17:end-16,:),[2,1,3,4]),3),1),[0,-1,1]);

target = size(dt_nifti.vol);
sz = size(im_sense);

im_sense2 = im_sense;

for d = 1:numel(target)
    sz = size(im_sense2);

    % center crop if larger
    if sz(d) > target(d)
        ncut = sz(d) - target(d);
        idx = repmat({':'}, 1, ndims(im_sense2));
        idx{d} = floor(ncut/2)+1 : sz(d)-ceil(ncut/2);
        im_sense2 = im_sense2(idx{:});
    end

    % center pad if smaller
    sz = size(im_sense2);
    if sz(d) < target(d)
        npad = target(d) - sz(d);
        pad_pre  = zeros(1, ndims(im_sense2));
        pad_post = zeros(1, ndims(im_sense2));
        pad_pre(d)  = floor(npad/2);
        pad_post(d) = ceil(npad/2);

        im_sense2 = padarray(im_sense2, pad_pre,  0, 'pre');
        im_sense2 = padarray(im_sense2, pad_post, 0, 'post');
    end
end

size(im_sense2)
size(dt_nifti.vol)

im_sense = im_sense2;
% axial orientation
% im_sense = circshift(flip(img_sense(:,:,9:end-8,:),3), [0,-1,1]);
% im_sense = circshift(flip(img_sense(:,:,17:end-16,:),3), [0,-1,1]);

imagesc3d2( im_sense(:,:,:,1), s(im_sense)/2, 2, [90,90,90], [0,2e-5]), setGcf(.5)


dt_sense = dt_nifti;
dt_sense.vol = abs(im_sense);
dt_sense.data_type = 16;
dt_sense.bitpix = 32;

dt_sense.vol = abs(im_sense) * max(dt_nifti.vol(:)) / max(abs(im_sense(:)));

save_nifti(dt_sense, [data_path, 'nifti/img_sense_correctHeader_', filename, '.nii'])


% cs recon

% sag orientation
im_cs = circshift(flip(flip(permute(img_cs,[2,1,3,4]),2),1),[1,0,-1]);
% im_cs = circshift(flip(flip(permute(img_cs(:,:,17:end-16,:),[2,1,3,4]),3),1),[0,-1,1]);
% im_cs = padarray(im_cs,(size(dt_nifti.vol)-size(im_cs))/2);

% axial orientation
% im_cs = circshift(flip(img_cs(:,:,9:end-8,:),3), [0,-1,1]);
target = size(dt_nifti.vol);
im_cs2 = im_cs;

for d = 1:numel(target)
    sz = size(im_cs2);

    % center crop if larger
    if sz(d) > target(d)
        ncut = sz(d) - target(d);
        idx = repmat({':'}, 1, ndims(im_cs2));
        idx{d} = floor(ncut/2)+1 : sz(d)-ceil(ncut/2);
        im_cs2 = im_cs2(idx{:});
    end

    % center pad if smaller
    sz = size(im_cs2);
    if sz(d) < target(d)
        npad = target(d) - sz(d);
        pad_pre  = zeros(1, ndims(im_cs2));
        pad_post = zeros(1, ndims(im_cs2));
        pad_pre(d)  = floor(npad/2);
        pad_post(d) = ceil(npad/2);

        im_cs2 = padarray(im_cs2, pad_pre,  0, 'pre');
        im_cs2 = padarray(im_cs2, pad_post, 0, 'post');
    end
end

size(im_cs2)
size(dt_nifti.vol)
im_cs = im_cs2;


imagesc3d2( im_cs(:,:,:,1), s(im_cs)/2, 3, [90,90,90], [0,2e-5]), setGcf(.5)

dt_cs = dt_nifti;
dt_cs.data_type = 16;
dt_cs.bitpix = 32;

dt_cs.vol = abs(im_cs) * max(dt_nifti.vol(:)) / max(abs(im_cs(:)));

save_nifti(dt_cs, [data_path, 'nifti/img_cs_correctHeader_', filename, '.nii'])


%--------------------------------------------------------------------------
%% load b1 map
%--------------------------------------------------------------------------

dt_b1_16 = load_nifti([data_path, 'nifti/dicom_gre_afi_b1_mgh_wtr_Ref220V_20251114134309_6.nii']);
dt_b1_18 = load_nifti([data_path, 'nifti/dicom_gre_afi_b1_mgh_wtr_Ref220V_20251114134309_8.nii']);

imagesc3d2( dt_b1_16.vol(:,:,:,1), s(dt_b1_16.vol)/2, 11, [90,90,90], [0,900]), setGcf(.5)
imagesc3d2( dt_b1_18.vol(:,:,:,1) / 58, s(dt_b1_16.vol)/2, 12, [90,90,90], [0,2]), setGcf(.5)


dt_b1_12 = load_nifti([data_path, 'nifti/dicom_gre_afi_b1_mgh_R3x1_FA58_Vref300V_20241016092457_12.nii']);
dt_b1_14 = load_nifti([data_path, 'nifti/dicom_gre_afi_b1_mgh_R3x1_FA58_Vref300V_20241016092457_14.nii']);

imagesc3d2( dt_b1_12.vol(:,:,:,1), s(dt_b1_12.vol)/2, 13, [90,90,90], [0,900]), setGcf(.5)
imagesc3d2( dt_b1_14.vol(:,:,:,1) / 58, s(dt_b1_12.vol)/2, 14, [90,90,90], [0,2]), setGcf(.5)




%--------------------------------------------------------------------------
%% load data create brain mask
%--------------------------------------------------------------------------

im_cs_rsos = rsos(dt_cs.vol,4);

thre = 0.25;

msk = BET(im_cs_rsos, s(im_cs_rsos), [1,1,1]*.39, thre);

imagesc3d2( dt_cs.vol(:,:,:,1) .* msk, s(msk)/2, 3, [90,90,90], [0,300]), setGcf(.5)
imagesc3d2( im_cs_rsos .* 1, s(msk)/2, 4, [90,90,90], [0,700]), setGcf(.5)


dt_cs_rsos_bet = dt_nifti;
dt_cs_rsos_bet.data_type = 16;
dt_cs_rsos_bet.bitpix = 32;

dt_cs_rsos_bet.vol = msk;

save_nifti(dt_cs_rsos_bet, [data_path, 'nifti/bet_mask_correctHeader_', filename, '.nii'])


%--------------------------------------------------------------------------
%% load data, test w/o opt shrink
%--------------------------------------------------------------------------

addpath /cluster/berkin/berkin/Matlab_Code_New/tmppca_qalas/Tensor-MP-PCA-main/

window = [6,6,6];
indices = {1:3 4};
opt_shrink = false;


tic
    [denoised_NOopt, Sigma2_NOopt, P_NOopt, SNR_gain_NOopt] = denoise_recursive_tensor(im_cs, ...
        window, 'indices', indices, 'mask', msk, 'opt_shrink', opt_shrink);
toc


imagesc3d2( dt_nifti.vol(:,:,:,1), s(dt_nifti.vol)/2, 1, [-180,-0,90], [0,300]), setGcf(.5)
imagesc3d2( denoised_NOopt(:,:,:,1), s(dt_nifti.vol)/2, 11, [-180,-0,90], [0,300]), setGcf(.5)


dt_mppca = dt_nifti;
dt_mppca.data_type = 16;
dt_mppca.bitpix = 32;

% normalize w/ same factor
dt_mppca.vol = abs(denoised_NOopt) * max(dt_nifti.vol(:)) / max(abs(im_cs(:)));

% save([data_path, 'denoised_NOopt_window6_', filename, '.mat'], 'denoised_NOopt', '-v7.3')
% save([data_path, 'SNR_gain_NOopt_window6_', filename, '.mat'], 'SNR_gain_NOopt', '-v7.3')

save_nifti(dt_mppca, [data_path, 'nifti/img_cs_denoised_NOopt_correctHeader_window6_', filename, '.nii'])


%--------------------------------------------------------------------------
%% load zs-ssl data, save nii file
%--------------------------------------------------------------------------

% zs-ssl recon
zs_ssl_path = '/autofs/space/daisy_002/users/berkin/2026_08_05_bay2_invivo_qalas_450um_cfa/';

img_zsssl = h5read([zs_ssl_path, 'qalas_450um_fa4_533_zsssl_tfv2_epoch9000.h5'], '/img_zsssl');
% img_zsssl = h5read([zs_ssl_path, 'qalas_390um_gcc_zsssl_tfv2_epoch200000.h5'], '/img_zsssl');

img_zsssl = img_zsssl.r + 1i * img_zsssl.i;
img_zsssl = permute(img_zsssl,[4 3 2 1]);
% imagesc3d2( img_zsssl(:,:,:,1), s(img_zsssl(:,:,:,1))/2, 11, [90,90,90], [0,1e-4]), setGcf(.5)


% axial orientation
% im_sense = circshift(flip(img_sense(:,:,9:end-8,:),3), [0,-1,1]);
% sag
im_zsssl = circshift(flip(flip(permute(img_zsssl,[2,1,3,4]),2),1),[1,0,-1]);
% im_zsssl = circshift(flip(img_zsssl(:,:,17:end-16,:),3), [0,-1,1]);
% im_zsssl = circshift(flip(flip(permute(img_zsssl(:,:,17:end-16,:),[2,1,3,4]),3),1),[0,-1,1]);
% im_zsssl = padarray(im_zsssl,(size(dt_nifti.vol)-size(im_zsssl))/2);

target = size(dt_nifti.vol);
im_zsssl2 = im_zsssl;

for d = 1:numel(target)
    sz = size(im_zsssl2);

    if sz(d) > target(d)
        ncut = sz(d) - target(d);
        idx = repmat({':'}, 1, ndims(im_zsssl2));
        idx{d} = floor(ncut/2)+1 : sz(d)-ceil(ncut/2);
        im_zsssl2 = im_zsssl2(idx{:});
    end

    sz = size(im_zsssl2);
    if sz(d) < target(d)
        npad = target(d) - sz(d);
        pad_pre = zeros(1, ndims(im_zsssl2));
        pad_post = zeros(1, ndims(im_zsssl2));

        pad_pre(d) = floor(npad/2);
        pad_post(d) = ceil(npad/2);

        im_zsssl2 = padarray(im_zsssl2, pad_pre, 0, 'pre');
        im_zsssl2 = padarray(im_zsssl2, pad_post, 0, 'post');
    end
end

size(im_zsssl2)
size(dt_nifti.vol)
im_zsssl = im_zsssl2;

imagesc3d2( im_zsssl(:,:,:,1), s(im_zsssl(:,:,:,1))/2, 11, [90,90,90], [0,1e-5]), setGcf(.5)


dt_zsssl = dt_nifti;
dt_zsssl.vol = abs(im_zsssl);
dt_zsssl.data_type = 16;
dt_zsssl.bitpix = 32;

dt_zsssl.vol = abs(im_zsssl) * max(dt_nifti.vol(:)) / max(abs(im_zsssl(:)));

save_nifti(dt_zsssl, [data_path, 'nifti/img_zsssl_correctHeader_', filename, '_gcc.nii'])

%--------------------------------------------------------------------------
%% load data create brain mask
%--------------------------------------------------------------------------

im_zsssl_rsos = rsos(dt_zsssl.vol,4);

thre = 0.15;

msk = BET(im_zsssl_rsos, s(im_zsssl_rsos), [1,1,1]*.45, thre);

imagesc3d2( dt_zsssl.vol(:,:,:,1) .* msk, s(msk)/2+[0 0 80], 3, [90,90,90], [0,300]), setGcf(.5)
imagesc3d2( im_zsssl_rsos .* 1, s(msk)/2, 4, [90,90,90], [0,700]), setGcf(.5)


dt_zsssl_rsos_bet = dt_nifti;
dt_zsssl_rsos_bet.data_type = 16;
dt_zsssl_rsos_bet.bitpix = 32;

dt_zsssl_rsos_bet.vol = msk;

save_nifti(dt_zsssl_rsos_bet, [data_path, 'nifti/bet_mask_correctHeader_', filename, '.nii'])






