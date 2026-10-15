%--------------------------------------------------------------------------
%% load data: cs-qalas on XA30
%--------------------------------------------------------------------------

addpath(genpath('/cluster/berkin/berkin/Matlab_Code_New/LIBRARY'))
addpath(genpath('/autofs/cluster/berkin/berkin/Matlab_Code/TOOLBOXES/SENSE_LSQR_Toolbox'))
addpath(genpath('/autofs/cluster/berkin/berkin/Matlab_Code_New/TOOLBOXES/CS_Wave_Toolbox'))

addpath '/autofs/cluster/berkin/berkin/Matlab_Code_New/NEATR/NEATR_Wip_936_Toolbox/neatr_wip_936_sense'

data_path_org = '/autofs/space/daisy_002/users/berkin/2026_08_05_bay2_invivo_qalas_450um_cfa/twix/';
data_path = '//autofs/space/daisy_002/users/berkin/2026_08_05_bay2_invivo_qalas_450um_cfa/';

filename = 'meas_MID00034_FID53327_qalas_vnav_fa4_450um_r2x2_Vref190_Inv430';

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

size_patref = size(ref_svd(:,:,:,1,1));
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
%% save data for zs-ssl
%--------------------------------------------------------------------------
mkdir([data_path,'data_20ch_GCC']);
kspace_svd_2d = ifftc(kspace_svd,1);

for nn = 1:size(kspace_svd,1)
    tic
    fprintf('slice: %d/%d... ',nn,size(kspace_svd,1));

    kspace      = sq(kspace_svd_2d(nn,:,:,:,:));
    sens_maps   = sq(rec(nn,:,:,:));
    mask        = single(sq(kspace(:,:,1,:))~=0);

    save([data_path,'data_20ch_GCC/','data_s',num2str(nn),'.mat'],'kspace','sens_maps','mask');
    toc
end

%--------------------------------------------------------------------------
%% after MZS-SSL recon, save data using nifti header info
%--------------------------------------------------------------------------

filename = 'dicom_file_name';


dt_nifti = load_nifti([data_path, filename]);

imagesc3d2( dt_nifti.vol(:,:,:,1), s(dt_nifti.vol)/2, 1, [90,90,90], [0,500]), setGcf(.5)



%--------------------------------------------------------------------------
%% load zs-ssl data, save nii file
%--------------------------------------------------------------------------

% zs-ssl recon path
zs_ssl_path = '/path_to_saved_recon_data/';

img_zsssl = h5read([zs_ssl_path], '/img_zsssl');
% img_zsssl = h5read([zs_ssl_path, 'qalas_390um_gcc_zsssl_tfv2_epoch200000.h5'], '/img_zsssl');

img_zsssl = img_zsssl.r + 1i * img_zsssl.i;
img_zsssl = permute(img_zsssl,[4 3 2 1]);
% imagesc3d2( img_zsssl(:,:,:,1), s(img_zsssl(:,:,:,1))/2, 11, [90,90,90], [0,1e-4]), setGcf(.5)


% axial orientation
% sag
im_zsssl = circshift(flip(flip(permute(img_zsssl,[2,1,3,4]),2),1),[1,0,-1]);

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







