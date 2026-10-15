%%

clear; clc; close all;

set(0,'DefaultFigureWindowStyle','docked')
addpath(genpath('/autofs/cluster/berkin/yohan/matlab_code/qalas/utils'));


%%

savepath = 'h5_data/multicoil_train/';
savename = 'train_data.h5';

compare_ref_map     = 0;
% 1 (compare ssl-qalas with reference maps (e.g., dictionary matching) during training)
% 0 (no comparison > will use zeros)

load_b1_map         = 1;
% 1 (load pre-acquired b1 map)
% 0 (no b1 map)

b1_type             = 2;
% 1 (TFL-based)
% 2 (AFI-based)

load_IE_map         = 1;


%% LOAD DATA

dpath       = 'dicom_data/image/qalas';
b1path      = 'dicom_data/image/b1';
maskpath = 'dicom_data/image/mask';
iepath      = 'ie/';

if compare_ref_map == 1
    load('map_data/ref_map.mat');
end

fprintf('loading data ... ');
tic
% input_img   = single(dicomread_dir(dpath));
input_img   = single(niftiread([dpath]));
input_img   = flip(permute(input_img,[2,1,3,4]),1);
% input_img   = reshape(input_img,[size(input_img,1),size(input_img,2),size(input_img,3)/5,1,5]);
input_img   = reshape(input_img,[size(input_img,1),size(input_img,2),size(input_img,3),1,5]);

if load_b1_map == 1
    % B1_map = single(dicomread_dir(b1path));
    B1_map = single(niftiread([b1path]));
    B1_map = flip(permute(B1_map,[2,1,3]),1);
    % load([b1path,'b1_map_nonChemShift_medfilt3_2qalas_afi_scale_factor.mat']);
    if b1_type == 1
        B1_map = B1_map./800; % for TFL-based B1
    elseif b1_type == 2
        B1_map = B1_map./60; % for AFI-based B1
        % B1_map = B1_map./afi_scale_factor; % for AFI-based B1
    end
    B1_map = imresize3(B1_map,[size(input_img,1),size(input_img,2),size(input_img,3)]);
    B1_map(B1_map<0) = 0;
    B1_map_thre_high = prctile(B1_map(:),99.9);
    B1_map_thre_low = prctile(B1_map(:),0.1);
    % B1_map_thre_high = 1.35;
    % B1_map_thre_low = 0.65;
    B1_map(B1_map>B1_map_thre_high) = B1_map_thre_high;
    B1_map(B1_map<B1_map_thre_low) = B1_map_thre_low;
end
toc

% 1-slice selection
% input_img = repmat(permute(input_img(:,100,:,:,:),[3,1,2,4,5]),[1,1,100,1,1]);
% B1_map = repmat(permute(B1_map(:,100,:),[3,1,2]),[1,1,100]);
%


%% Brain Mask (simple thresholding mask) -> may not be accurate

% threshold = 100; % 50 / 100/ 250
% 
[Nx,Ny,Nz,~,~]  = size(input_img);
% bmask           = ones(Nx,Ny,Nz,'single');

% load([dpath,'bmask.mat']);
% 
% for slc = 1:size(input_img,3)
%    bmask(:,:,slc) = imfill(squeeze(rsos(input_img(:,:,slc,1,:),5)) > threshold, 'holes');
% end
% bmask(:,:,129:end) = repmat(bmask(:,:,128),[1,1,(size(bmask,3)-129+1)]);

bmask = single(niftiread([maskpath]));
bmask = flip(permute(bmask,[2,1,3]),1);


%%

input_img       = input_img./max(input_img(:));
% input_img       = input_img./prctile(input_img(:),99)./2.5;

sens            = ones(Nx,Ny,Nz,1,'single');
mask            = ones(Nx,Ny,'single');
if compare_ref_map == 0
    T1_map = ones(Nx,Ny,Nz,'single').*5;
    T2_map = ones(Nx,Ny,Nz,'single').*2.5;
    PD_map = ones(Nx,Ny,Nz,'single');
    IE_map = ones(Nx,Ny,Nz,'single');
end
if load_b1_map == 0
    B1_map = ones(Nx,Ny,Nz,'single');
end
if load_IE_map == 1
    % load([iepath,'ielookup_toolbox_v3_500deg.mat']);
    load([iepath,'ielookup_toolbox_v3_FA750.mat']); % 7T
    % load([iepath,'ielookup_toolbox_v3.mat']);
    IE_map = single(cat(2,ielookup.ies_mtx,cat(1,ielookup.T1',zeros(length(ielookup.T2)-length(ielookup.T1),1)),ielookup.T2));
end

input_img   = permute(input_img,[2,1,4,3,5]);
sens        = permute(sens,[2,1,4,3]);

T1_map      = permute(T1_map,[2,1,3]);
T2_map      = permute(T2_map,[2,1,3]);
PD_map      = permute(PD_map,[2,1,3]);
IE_map      = permute(IE_map,[2,1,3]);
B1_map      = permute(B1_map,[2,1,3]);

bmask       = permute(bmask,[2,1,3]);
mask        = permute(mask,[2,1]);

kspace_acq1 = single(input_img(:,:,:,:,1));
kspace_acq2 = single(input_img(:,:,:,:,2));
kspace_acq3 = single(input_img(:,:,:,:,3));
kspace_acq4 = single(input_img(:,:,:,:,4));
kspace_acq5 = single(input_img(:,:,:,:,5));


%% SAVE DATA

fprintf('save h5 data ... ');

tic

file_name   = strcat(savepath,savename);

att_patient = '0000';
att_seq     = 'QALAS';

kspace_acq1     = permute(kspace_acq1,[4,3,2,1]);
kspace_acq2     = permute(kspace_acq2,[4,3,2,1]);
kspace_acq3     = permute(kspace_acq3,[4,3,2,1]);
kspace_acq4     = permute(kspace_acq4,[4,3,2,1]);
kspace_acq5     = permute(kspace_acq5,[4,3,2,1]);
coil_sens       = permute(sens,[4,3,2,1]);

saveh5(struct('kspace_acq1', kspace_acq1, 'kspace_acq2', kspace_acq2, ...
              'kspace_acq3', kspace_acq3, 'kspace_acq4', kspace_acq4, ...
              'kspace_acq5', kspace_acq5), ...
              file_name, 'ComplexFormat',{'r','i'});

h5create(file_name,'/reconstruction_t1',[Ny,Nx,Nz],'Datatype','single');
h5write(file_name, '/reconstruction_t1', T1_map);
h5create(file_name,'/reconstruction_t2',[Ny,Nx,Nz],'Datatype','single');
h5write(file_name, '/reconstruction_t2', T2_map);
h5create(file_name,'/reconstruction_pd',[Ny,Nx,Nz],'Datatype','single');
h5write(file_name, '/reconstruction_pd', PD_map);
if load_IE_map == 0
    h5create(file_name,'/reconstruction_ie',[Ny,Nx,Nz],'Datatype','single');
    h5write(file_name, '/reconstruction_ie', IE_map);
else
    h5create(file_name,'/reconstruction_ie',[size(IE_map,1),size(IE_map,2),size(IE_map,3)],'Datatype','single');
    h5write(file_name, '/reconstruction_ie', IE_map);
end
h5create(file_name,'/reconstruction_b1',[Ny,Nx,Nz],'Datatype','single');
h5write(file_name, '/reconstruction_b1', B1_map);

% h5create(file_name,'/mask_acq1',[Ny,1],'Datatype','single');
% h5write(file_name, '/mask_acq1', mask(:,1));
% h5create(file_name,'/mask_acq2',[Ny,1],'Datatype','single');
% h5write(file_name, '/mask_acq2', mask(:,1));
% h5create(file_name,'/mask_acq3',[Ny,1],'Datatype','single');
% h5write(file_name, '/mask_acq3', mask(:,1));
% h5create(file_name,'/mask_acq4',[Ny,1],'Datatype','single');
% h5write(file_name, '/mask_acq4', mask(:,1));
% h5create(file_name,'/mask_acq5',[Ny,1],'Datatype','single');
% h5write(file_name, '/mask_acq5', mask(:,1));
h5create(file_name,'/mask_acq1',[Nx,1],'Datatype','single');
h5write(file_name, '/mask_acq1', mask(1,:)');
h5create(file_name,'/mask_acq2',[Nx,1],'Datatype','single');
h5write(file_name, '/mask_acq2', mask(1,:)');
h5create(file_name,'/mask_acq3',[Nx,1],'Datatype','single');
h5write(file_name, '/mask_acq3', mask(1,:)');
h5create(file_name,'/mask_acq4',[Nx,1],'Datatype','single');
h5write(file_name, '/mask_acq4', mask(1,:)');
h5create(file_name,'/mask_acq5',[Nx,1],'Datatype','single');
h5write(file_name, '/mask_acq5', mask(1,:)');

h5create(file_name,'/mask_brain',[Ny,Nx,Nz],'Datatype','single');
h5write(file_name, '/mask_brain', single(bmask));

att_norm_t1 = norm(T1_map(:));
att_max_t1  = max(T1_map(:));
att_norm_t2 = norm(T2_map(:));
att_max_t2  = max(T2_map(:));
att_norm_pd = norm(PD_map(:));
att_max_pd  = max(PD_map(:));
att_norm_ie = norm(IE_map(:));
att_max_ie  = max(IE_map(:));
att_norm_b1 = norm(B1_map(:));
att_max_b1  = max(B1_map(:));

h5writeatt(file_name,'/','norm_t1',att_norm_t1);
h5writeatt(file_name,'/','max_t1',att_max_t1);
h5writeatt(file_name,'/','norm_t2',att_norm_t2);
h5writeatt(file_name,'/','max_t2',att_max_t2);
h5writeatt(file_name,'/','norm_pd',att_norm_pd);
h5writeatt(file_name,'/','max_pd',att_max_pd);
h5writeatt(file_name,'/','norm_ie',att_norm_ie);
h5writeatt(file_name,'/','max_ie',att_max_ie);
h5writeatt(file_name,'/','norm_b1',att_norm_b1);
h5writeatt(file_name,'/','max_b1',att_max_b1);
h5writeatt(file_name,'/','patient_id',att_patient);
h5writeatt(file_name,'/','acquisition',att_seq);


%%

dset = ismrmrd.Dataset(file_name);

header = [];

% Experimental Conditions (Required)
header.experimentalConditions.H1resonanceFrequency_Hz   = 128000000; % 3T

% Acquisition System Information (Optional)
header.acquisitionSystemInformation.receiverChannels    = 32;

% The Encoding (Required)
header.encoding.trajectory = 'cartesian';
header.encoding.encodedSpace.fieldOfView_mm.x   = Nx;
header.encoding.encodedSpace.fieldOfView_mm.y   = Ny;
header.encoding.encodedSpace.fieldOfView_mm.z   = Nz;
header.encoding.encodedSpace.matrixSize.x       = Nx*2;
header.encoding.encodedSpace.matrixSize.y       = Ny;
header.encoding.encodedSpace.matrixSize.z       = Nz;

% Recon Space
header.encoding.reconSpace.fieldOfView_mm.x     = Nx;
header.encoding.reconSpace.fieldOfView_mm.y     = Ny;
header.encoding.reconSpace.fieldOfView_mm.z     = Nz;
header.encoding.reconSpace.matrixSize.x         = Nx;
header.encoding.reconSpace.matrixSize.y         = Ny;
header.encoding.reconSpace.matrixSize.z         = Nz;

% Encoding Limits
header.encoding.encodingLimits.kspace_encoding_step_1.minimum   = 0;
header.encoding.encodingLimits.kspace_encoding_step_1.maximum   = Nx-1;
header.encoding.encodingLimits.kspace_encoding_step_1.center    = Nx/2;
header.encoding.encodingLimits.kspace_encoding_step_2.minimum   = 0;
header.encoding.encodingLimits.kspace_encoding_step_2.maximum   = 0;
header.encoding.encodingLimits.kspace_encoding_step_2.center    = 0;

% Serialize and write to the data set
xmlstring = ismrmrd.xml.serialize(header);
dset.writexml(xmlstring);

% Write the dataset
dset.close();
toc
