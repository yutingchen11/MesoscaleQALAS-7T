%--------------------------------------------------------------------------
%% load data and prot 
%--------------------------------------------------------------------------

load prot
load img_gre_R1_fa04_1p25mm

imagesc3d2(rsos(img_gre_R1_fa04_1p25mm,4), [80,80,60], 1, [-90,-90,180], [0,2.5e-4], [], '32 chan')


%--------------------------------------------------------------------------
%% coil compression: SVD 
%--------------------------------------------------------------------------

[num_read, N(1), N(2), all_chan] = size(img_gre_R1_fa04_1p25mm);

kspace = fft3call(img_gre_R1_fa04_1p25mm);

num_chan_svd = 16;  % svd compression
num_acs = 24;       % calibration size in (ky,kz) to estimate gcc matrix 

% estimate svd compression matrix from center of k-space
calib = crop(kspace, [num_read, num_acs, num_acs, all_chan]);

[~, cmp_mtx] = svd_compress3d(ifft3call(calib), num_chan_svd);

% apply estimated compression matrix
img_svd = svd_apply3d(img_gre_R1_fa04_1p25mm, cmp_mtx);

rmse_svd = rmse(rsos(img_svd,4), rsos(img_gre_R1_fa04_1p25mm,4), 1)

imagesc3d2(rsos(img_svd,4), [80,80,60], 2, [-90,-90,180], [0, 2.5e-4], [], 'svd')
imagesc3d2(10 * abs(rsos(img_svd,4) - rsos(img_gre_R1_fa04_1p25mm,4)), [80,80,60], 12, [-90,-90,180], [0, 2.5e-4], [], 'svd error')


%--------------------------------------------------------------------------
%% coil compression: GCC 
%--------------------------------------------------------------------------

num_chan_gcc = 16;  % num channels to compress to
dim = 1;            % readout axis


% estimate gcc matrix
gccmtx = calcGCCMtx(calib, dim);
 
gccmtx_aligned = alignCCMtx(gccmtx(:,1:num_chan_gcc,:));

% apply gcc matrix
CCDATA_aligned = CC(kspace, dim, gccmtx_aligned);
    
img_gcc = ifft3call(CCDATA_aligned);

rmse_gcc = rmse(rsos(img_gcc,4), rsos(img_gre_R1_fa04_1p25mm,4), 1)

imagesc3d2(rsos(img_gcc,4), [80,80,60], 3, [-90,-90,180], [0, 2.5e-4], [], 'gcc')
imagesc3d2(10 * abs(rsos(img_gcc,4) - rsos(img_gre_R1_fa04_1p25mm,4)), [80,80,60], 13, [-90,-90,180], [0, 2.5e-4], [], 'gcc error')




    
   
