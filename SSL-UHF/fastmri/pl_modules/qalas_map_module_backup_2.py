"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser

import fastmri
import torch
from fastmri.data import transforms_qalas
from fastmri.models import QALAS_MAP

from .mri_module_qalas_map import MriModuleQALAS_MAP
import numpy as np
import scipy.io as sio

class MapSampledImageLoss(torch.nn.Module):
    def __init__(self, map_min, map_max, num_bins, samples_per_bin):
        super().__init__()
        self.map_bins = torch.linspace(map_min, map_max, num_bins + 1)
        self.num_bins = num_bins
        self.samples_per_bin = samples_per_bin
        self.loss_fn = torch.nn.MSELoss(reduction='sum') # Use sum to average later

    def forward(self, pred_map, recon_image, target_image):
        total_loss = torch.tensor(0.0).to(pred_map.device)
        
        # Ensure bins are on the same device as the data
        self.map_bins = self.map_bins.to(pred_map.device)

        # Flatten the tensors for easier indexing
        pred_map_flat = pred_map.view(-1)
        recon_image_flat = recon_image.view(-1)
        target_image_flat = target_image.view(-1)
        
        # Calculate loss for each map bin
        for i in range(self.num_bins):
            # 1. Find the indices of pixels in the current map bin
            bin_indices = torch.nonzero(
                (pred_map_flat >= self.map_bins[i]) & (pred_map_flat < self.map_bins[i+1]),
                as_tuple=False
            ).squeeze()

            if bin_indices.numel() == 1:
                continue

            # 2. Sample a fixed number of pixels from the bin
            num_to_sample = min(self.samples_per_bin, bin_indices.numel())
            
            # Randomly select indices
            sampled_indices = bin_indices[torch.randperm(bin_indices.numel())[:num_to_sample]]

            # 3. Calculate the loss for the sampled pixels
            sampled_recon = recon_image_flat[sampled_indices]
            sampled_target = target_image_flat[sampled_indices]

            bin_loss = self.loss_fn(sampled_recon, sampled_target)
            total_loss += bin_loss

        # Average the total loss by the number of bins
        # This provides an equal contribution from each bin
        return total_loss / self.num_bins

class QALAS_MAPModule(MriModuleQALAS_MAP):
    """
    QALAS training module.

    This can be used to train variational networks from the paper:

    A. Sriram et al. End-to-end variational networks for accelerated MRI
    reconstruction. In International Conference on Medical Image Computing and
    Computer-Assisted Intervention, 2020.

    which was inspired by the earlier paper:

    K. Hammernik et al. Learning a variational network for reconstruction of
    accelerated MRI data. Magnetic Resonance inMedicine, 79(6):3055–3071, 2018.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        pools: int = 4,
        chans: int = 18,
        maps_chans: int = 32,
        maps_layers: int = 5,
        lr: float = 0.0003,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            chans: Number of channels for cascade U-Net.
            lr: Learning rate.
            lr_step_size: Learning rate step size.
            lr_gamma: Learning rate gamma decay.
            weight_decay: Parameter for penalizing weights norm.
            num_sense_lines: Number of low-frequency lines to use for sensitivity map
                computation, must be even or `None`. Default `None` will automatically
                compute the number from masks. Default behaviour may cause some slices to
                use more low-frequency lines than others, when used in conjunction with
                e.g. the EquispacedMaskFunc defaults. To prevent this, either set
                `num_sense_lines`, or set `skip_low_freqs` and `skip_around_low_freqs`
                to `True` in the EquispacedMaskFunc. Note that setting this value may
                lead to undesired behaviour when training on multiple accelerations
                simultaneously.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.num_cascades = num_cascades
        self.pools = pools
        self.chans = chans
        self.maps_chans = maps_chans
        self.maps_layers = maps_layers
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.qalas = QALAS_MAP(
            num_cascades=self.num_cascades,
            maps_chans=self.maps_chans,
            maps_layers=self.maps_layers,
            chans=self.chans,
            pools=self.pools,
        )

        # self.loss_t1_sampled = MapSampledImageLoss(map_min=0.0, map_max=5000.0, num_bins=50, samples_per_bin=100)
        # self.loss_t2_sampled = MapSampledImageLoss(map_min=0.0, map_max=2500.0, num_bins=100, samples_per_bin=50)
        self.loss_t1_sampled = MapSampledImageLoss(map_min=0.0, map_max=5.0, num_bins=50, samples_per_bin=100)
        self.loss_t2_sampled = MapSampledImageLoss(map_min=0.0, map_max=2.5, num_bins=100, samples_per_bin=50)
        self.loss_l2_t1 = torch.nn.MSELoss()
        self.loss_l2_t2 = torch.nn.MSELoss()
        self.loss_l2_pd = torch.nn.MSELoss()
        self.loss_l2_ie = torch.nn.MSELoss()
        self.loss_l2_img1 = torch.nn.MSELoss()
        self.loss_l2_img2 = torch.nn.MSELoss()
        self.loss_l2_img3 = torch.nn.MSELoss()
        self.loss_l2_img4 = torch.nn.MSELoss()
        self.loss_l2_img5 = torch.nn.MSELoss()
        # self.loss_l2_t1 = fastmri.SSIMLoss()
        # self.loss_l2_t2 = fastmri.SSIMLoss()
        # self.loss_l2_pd = fastmri.SSIMLoss()
        # self.loss_l2_img1 = fastmri.SSIMLoss()
        # self.loss_l2_img2 = fastmri.SSIMLoss()
        # self.loss_l2_img3 = fastmri.SSIMLoss()
        # self.loss_l2_img4 = fastmri.SSIMLoss()
        # self.loss_l2_img5 = fastmri.SSIMLoss()

    def forward(self, masked_kspace_acq1, masked_kspace_acq2, masked_kspace_acq3, masked_kspace_acq4, masked_kspace_acq5, \
                mask_acq1, mask_acq2, mask_acq3, mask_acq4, mask_acq5, mask_brain, \
                b1, ie, max_value_t1, max_value_t2, max_value_pd, num_low_frequencies):
        return self.qalas(masked_kspace_acq1, masked_kspace_acq2, masked_kspace_acq3, masked_kspace_acq4, masked_kspace_acq5, \
                        mask_acq1, mask_acq2, mask_acq3, mask_acq4, mask_acq5, mask_brain, \
                        b1, ie, max_value_t1, max_value_t2, max_value_pd, num_low_frequencies)

    def training_step(self, batch, batch_idx):
        # ie_mat = sio.loadmat('ie_tmp/ie_s' + str(batch_idx+1) + '.mat') # for dict_v4 IE
        # ie = torch.from_numpy(ie_mat['ie']) # for dict_v4 IE
        
        output_t1, output_t2, output_pd, output_ie, output_b1, \
        output_img1, output_img2, output_img3, output_img4, output_img5 = \
            self(batch.masked_kspace_acq1, batch.masked_kspace_acq2, batch.masked_kspace_acq3, batch.masked_kspace_acq4, batch.masked_kspace_acq5, \
                batch.mask_acq1, batch.mask_acq2, batch.mask_acq3, batch.mask_acq4, batch.mask_acq5, batch.mask_brain, \
                # batch.b1, ie, batch.max_value_t1, batch.max_value_t2, batch.max_value_pd, batch.num_low_frequencies) # for previous dict_v4 IE
                batch.b1, batch.ie, batch.max_value_t1, batch.max_value_t2, batch.max_value_pd, batch.num_low_frequencies)

        # If DICOM data were used, use following 5 lines
        img_acq1 = batch.masked_kspace_acq1
        img_acq2 = batch.masked_kspace_acq2
        img_acq3 = batch.masked_kspace_acq3
        img_acq4 = batch.masked_kspace_acq4
        img_acq5 = batch.masked_kspace_acq5

        target_t1, output_t1 = transforms_qalas.center_crop_to_smallest(batch.target_t1, output_t1)
        target_t2, output_t2 = transforms_qalas.center_crop_to_smallest(batch.target_t2, output_t2)
        target_pd, output_pd = transforms_qalas.center_crop_to_smallest(batch.target_pd, output_pd)
        target_pd, output_ie = transforms_qalas.center_crop_to_smallest(batch.target_pd, output_ie)
        target_pd, output_b1 = transforms_qalas.center_crop_to_smallest(batch.target_pd, output_b1)

        target_t1, img_acq1 = transforms_qalas.center_crop_to_smallest(batch.target_t1, img_acq1.squeeze(1))
        target_t1, img_acq2 = transforms_qalas.center_crop_to_smallest(batch.target_t1, img_acq2.squeeze(1))
        target_t1, img_acq3 = transforms_qalas.center_crop_to_smallest(batch.target_t1, img_acq3.squeeze(1))
        target_t1, img_acq4 = transforms_qalas.center_crop_to_smallest(batch.target_t1, img_acq4.squeeze(1))
        target_t1, img_acq5 = transforms_qalas.center_crop_to_smallest(batch.target_t1, img_acq5.squeeze(1))

        target_t1, output_img1 = transforms_qalas.center_crop_to_smallest(batch.target_t1, output_img1.squeeze(1))
        target_t1, output_img2 = transforms_qalas.center_crop_to_smallest(batch.target_t1, output_img2.squeeze(1))
        target_t1, output_img3 = transforms_qalas.center_crop_to_smallest(batch.target_t1, output_img3.squeeze(1))
        target_t1, output_img4 = transforms_qalas.center_crop_to_smallest(batch.target_t1, output_img4.squeeze(1))
        target_t1, output_img5 = transforms_qalas.center_crop_to_smallest(batch.target_t1, output_img5.squeeze(1))

        target_t1 = target_t1 * batch.mask_brain
        target_t2 = target_t2 * batch.mask_brain
        target_pd = target_pd * batch.mask_brain

        output_t1 = output_t1 * batch.mask_brain
        output_t2 = output_t2 * batch.mask_brain
        output_pd = output_pd * batch.mask_brain
        output_ie = output_ie * batch.mask_brain
        output_b1 = output_b1 * batch.mask_brain

        ### for dict_v4 IE
        b, nx, ny = output_ie.shape
        target_ie = torch.ones_like(output_ie)
        with torch.no_grad():
            t1_dict = batch.ie[:,0:batch.ie.shape[2]-2,-2].squeeze(0).to(output_t1.device)
            t2_dict = batch.ie[:,:,-1].squeeze(0).to(output_t2.device)
            ie_ = batch.ie.squeeze(0)[:,0:-2].to(output_ie.device)
            for x in range(nx):
                t1_idx = torch.argmin(torch.absolute(t1_dict.unsqueeze(-1).repeat(1,ny)-output_t1[:,x,:].repeat(t1_dict.size(0),1)), dim=0)
                t2_idx = torch.argmin(torch.absolute(t2_dict.unsqueeze(-1).repeat(1,ny)-output_t2[:,x,:].repeat(t2_dict.size(0),1)), dim=0)
                target_ie[:,x,:] = ie_[t2_idx,t1_idx].unsqueeze(0)
        target_ie = target_ie * batch.mask_brain
        ###

        img_acq1 = img_acq1 * batch.mask_brain
        img_acq2 = img_acq2 * batch.mask_brain
        img_acq3 = img_acq3 * batch.mask_brain
        img_acq4 = img_acq4 * batch.mask_brain
        img_acq5 = img_acq5 * batch.mask_brain

        output_img1 = output_img1 * batch.mask_brain
        output_img2 = output_img2 * batch.mask_brain
        output_img3 = output_img3 * batch.mask_brain
        output_img4 = output_img4 * batch.mask_brain
        output_img5 = output_img5 * batch.mask_brain

        if batch.mask_brain.sum() == 0:
            target_t1 = target_t1 + 1e-5
            target_t2 = target_t2 + 1e-5
            target_pd = target_pd + 1e-5
            target_ie = target_ie + 1e-5
            output_t1 = output_t1 + 1e-5
            output_t2 = output_t2 + 1e-5
            output_pd = output_pd + 1e-5
            output_ie = output_ie + 1e-5
            output_b1 = output_b1 + 1e-5
            img_acq1 = img_acq1 + 1e-5
            img_acq2 = img_acq2 + 1e-5
            img_acq3 = img_acq3 + 1e-5
            img_acq4 = img_acq4 + 1e-5
            img_acq5 = img_acq5 + 1e-5
            output_img1 = output_img1 + 1e-5
            output_img2 = output_img2 + 1e-5
            output_img3 = output_img3 + 1e-5
            output_img4 = output_img4 + 1e-5
            output_img5 = output_img5 + 1e-5

        loss_t1 = self.loss_l2_t1(output_t1.unsqueeze(1), target_t1.unsqueeze(1))
        loss_t2 = self.loss_l2_t2(output_t2.unsqueeze(1), target_t2.unsqueeze(1))
        loss_pd = self.loss_l2_pd(output_pd.unsqueeze(1) / output_pd.max(), target_pd.unsqueeze(1) / target_pd.max())
        # loss_pd = self.loss_l2_pd(output_ie.unsqueeze(1), target_pd.unsqueeze(1))
        loss_ie = self.loss_l2_ie(output_ie.unsqueeze(1), target_ie.unsqueeze(1))
        # loss_ie = self.loss_t2_sampled(output_t2.unsqueeze(1), output_ie.unsqueeze(1), target_ie.unsqueeze(1))

        # loss_img1 = self.loss_l2_img1(output_img1.unsqueeze(1), img_acq1.unsqueeze(1))
        # loss_img2 = self.loss_l2_img2(output_img2.unsqueeze(1), img_acq2.unsqueeze(1))
        # loss_img3 = self.loss_l2_img3(output_img3.unsqueeze(1), img_acq3.unsqueeze(1))
        # loss_img4 = self.loss_l2_img4(output_img4.unsqueeze(1), img_acq4.unsqueeze(1))
        # loss_img5 = self.loss_l2_img5(output_img5.unsqueeze(1), img_acq5.unsqueeze(1))
        loss_img1 = self.loss_t2_sampled(output_t2.unsqueeze(1), output_img1.unsqueeze(1), img_acq1.unsqueeze(1))
        loss_img2 = self.loss_t1_sampled(output_t1.unsqueeze(1), output_img2.unsqueeze(1), img_acq2.unsqueeze(1))
        loss_img3 = self.loss_t1_sampled(output_t1.unsqueeze(1), output_img3.unsqueeze(1), img_acq3.unsqueeze(1))
        loss_img4 = self.loss_t1_sampled(output_t1.unsqueeze(1), output_img4.unsqueeze(1), img_acq4.unsqueeze(1))
        loss_img5 = self.loss_t1_sampled(output_t1.unsqueeze(1), output_img5.unsqueeze(1), img_acq5.unsqueeze(1))
        def loss_tv(img):
            pixel_dif1 = img[..., 1:,:] - img[..., :-1,:]
            pixel_dif2 = img[..., :,1:] - img[..., :,:-1]
            reduce_axes = (-2,-1)
            res1 = pixel_dif1.abs().sum(dim=reduce_axes)
            res2 = pixel_dif2.abs().sum(dim=reduce_axes)
            return (res1 + res2) / (img.shape[-1] * img.shape[-2])
        loss_img1_tv = loss_tv(output_img1)
        loss_img2_tv = loss_tv(output_img2)
        loss_img3_tv = loss_tv(output_img3)
        loss_img4_tv = loss_tv(output_img4)
        loss_img5_tv = loss_tv(output_img5)
        loss_t1_tv = loss_tv(output_t1)
        loss_t2_tv = loss_tv(output_t2)
        loss_pd_tv = loss_tv(output_pd)
        loss_ie_tv = loss_tv(output_ie)

        loss_weight_t1 = 0
        loss_weight_t2 = 0
        loss_weight_pd = 0
        loss_weight_ie = 0.1 # 0.1 (best)
        loss_weight_img1 = 1
        loss_weight_img2 = 1
        loss_weight_img3 = 1
        loss_weight_img4 = 1
        loss_weight_img5 = 1
        loss_weight_img1_tv = 0
        loss_weight_img2_tv = 0
        loss_weight_img3_tv = 0
        loss_weight_img4_tv = 0
        loss_weight_img5_tv = 0
        loss_weight_t1_tv = 0
        loss_weight_t2_tv = 0
        loss_weight_pd_tv = 0
        loss_weight_ie_tv = 0
        # loss = (loss_t1 * loss_weight_t1 + loss_t2 * loss_weight_t2 + loss_pd * loss_weight_pd + \
        #         loss_img1 * loss_weight_img1 + loss_img2 * loss_weight_img2 + loss_img3 * loss_weight_img3 + loss_img4 * loss_weight_img4 + loss_img5 * loss_weight_img5) \
        #          / (loss_weight_t1 + loss_weight_t2 + loss_weight_pd + \
        #             loss_weight_img1 + loss_weight_img2 + loss_weight_img3 + loss_weight_img4 + loss_weight_img5)
        # loss = (loss_t1 * loss_weight_t1 + loss_t2 * loss_weight_t2 + loss_pd * loss_weight_pd + \
        #         loss_img1 * loss_weight_img1 + loss_img2 * loss_weight_img2 + loss_img3 * loss_weight_img3 + loss_img4 * loss_weight_img4 + loss_img5 * loss_weight_img5 + \
        #         loss_img1_tv * loss_weight_img1_tv + loss_img2_tv * loss_weight_img2_tv + loss_img3_tv * loss_weight_img3_tv + loss_img4_tv * loss_weight_img4_tv + loss_img5_tv * loss_weight_img5_tv) \
        #          / (loss_weight_t1 + loss_weight_t2 + loss_weight_pd + \
        #             loss_weight_img1 + loss_weight_img2 + loss_weight_img3 + loss_weight_img4 + loss_weight_img5 + \
        #             loss_weight_img1_tv + loss_weight_img2_tv + loss_weight_img3_tv + loss_weight_img4_tv + loss_weight_img5_tv)
        loss = (loss_t1 * loss_weight_t1 + loss_t2 * loss_weight_t2 + loss_pd * loss_weight_pd + loss_ie * loss_weight_ie + \
                loss_img1 * loss_weight_img1 + loss_img2 * loss_weight_img2 + loss_img3 * loss_weight_img3 + loss_img4 * loss_weight_img4 + loss_img5 * loss_weight_img5 + \
                loss_t1_tv * loss_weight_t1_tv + loss_t2_tv * loss_weight_t2_tv + loss_pd_tv * loss_weight_pd_tv + loss_ie_tv * loss_weight_ie_tv) \
                 / (loss_weight_t1 + loss_weight_t2 + loss_weight_pd + \
                    loss_weight_img1 + loss_weight_img2 + loss_weight_img3 + loss_weight_img4 + loss_weight_img5 + \
                    loss_weight_t1_tv + loss_weight_t2_tv + loss_weight_pd_tv + loss_weight_ie_tv)

        self.log("train_loss_t1", loss_t1)
        self.log("train_loss_t2", loss_t2)
        self.log("train_loss_pd", loss_pd)
        self.log("train_loss_ie", loss_ie)
        self.log("train_loss_img1", loss_img1)
        self.log("train_loss_img2", loss_img2)
        self.log("train_loss_img3", loss_img3)
        self.log("train_loss_img4", loss_img4)
        self.log("train_loss_img5", loss_img5)

        return loss


    def validation_step(self, batch, batch_idx):
        # ie_mat = sio.loadmat('ie_tmp/ie_s' + str(batch_idx+1) + '.mat') # for previous dict_v4 IE
        # ie = torch.from_numpy(ie_mat['ie']) # for previous dict_v4 IE

        output_t1, output_t2, output_pd, output_ie, output_b1, \
        output_img1, output_img2, output_img3, output_img4, output_img5 = \
            self.forward(batch.masked_kspace_acq1, batch.masked_kspace_acq2, batch.masked_kspace_acq3, batch.masked_kspace_acq4, batch.masked_kspace_acq5, \
                        batch.mask_acq1, batch.mask_acq2, batch.mask_acq3, batch.mask_acq4, batch.mask_acq5, batch.mask_brain, \
                        # batch.b1, ie, batch.max_value_t1, batch.max_value_t2, batch.max_value_pd, batch.num_low_frequencies) # for previous dict_v4 IE
                        batch.b1, batch.ie, batch.max_value_t1, batch.max_value_t2, batch.max_value_pd, batch.num_low_frequencies)

        # If DICOM data were used, use following 5 lines
        img_acq1 = batch.masked_kspace_acq1
        img_acq2 = batch.masked_kspace_acq2
        img_acq3 = batch.masked_kspace_acq3
        img_acq4 = batch.masked_kspace_acq4
        img_acq5 = batch.masked_kspace_acq5

        target_t1, output_t1 = transforms_qalas.center_crop_to_smallest(batch.target_t1, output_t1)
        target_t2, output_t2 = transforms_qalas.center_crop_to_smallest(batch.target_t2, output_t2)
        target_pd, output_pd = transforms_qalas.center_crop_to_smallest(batch.target_pd, output_pd)
        target_pd, output_ie = transforms_qalas.center_crop_to_smallest(batch.target_pd, output_ie)
        target_pd, output_b1 = transforms_qalas.center_crop_to_smallest(batch.target_pd, output_b1)

        target_t1, img_acq1 = transforms_qalas.center_crop_to_smallest(batch.target_t1, img_acq1.squeeze(1))
        target_t1, img_acq2 = transforms_qalas.center_crop_to_smallest(batch.target_t1, img_acq2.squeeze(1))
        target_t1, img_acq3 = transforms_qalas.center_crop_to_smallest(batch.target_t1, img_acq3.squeeze(1))
        target_t1, img_acq4 = transforms_qalas.center_crop_to_smallest(batch.target_t1, img_acq4.squeeze(1))
        target_t1, img_acq5 = transforms_qalas.center_crop_to_smallest(batch.target_t1, img_acq5.squeeze(1))

        target_t1, output_img1 = transforms_qalas.center_crop_to_smallest(batch.target_t1, output_img1.squeeze(1))
        target_t1, output_img2 = transforms_qalas.center_crop_to_smallest(batch.target_t1, output_img2.squeeze(1))
        target_t1, output_img3 = transforms_qalas.center_crop_to_smallest(batch.target_t1, output_img3.squeeze(1))
        target_t1, output_img4 = transforms_qalas.center_crop_to_smallest(batch.target_t1, output_img4.squeeze(1))
        target_t1, output_img5 = transforms_qalas.center_crop_to_smallest(batch.target_t1, output_img5.squeeze(1))

        target_t1 = target_t1 * batch.mask_brain
        target_t2 = target_t2 * batch.mask_brain
        target_pd = target_pd * batch.mask_brain

        output_t1 = output_t1 * batch.mask_brain
        output_t2 = output_t2 * batch.mask_brain
        output_pd = output_pd * batch.mask_brain
        output_ie = output_ie * batch.mask_brain
        output_b1 = output_b1 * batch.mask_brain

        ### for dict_v4 IE
        b, nx, ny = output_ie.shape
        target_ie = torch.ones_like(output_ie)
        with torch.no_grad():
            t1_dict = batch.ie[:,0:batch.ie.shape[2]-2,-2].squeeze(0).to(output_t1.device)
            t2_dict = batch.ie[:,:,-1].squeeze(0).to(output_t2.device)
            ie_ = batch.ie.squeeze(0)[:,0:-2].to(output_ie.device)
            for x in range(nx):
                t1_idx = torch.argmin(torch.absolute(t1_dict.unsqueeze(-1).repeat(1,ny)-output_t1[:,x,:].repeat(t1_dict.size(0),1)), dim=0)
                t2_idx = torch.argmin(torch.absolute(t2_dict.unsqueeze(-1).repeat(1,ny)-output_t2[:,x,:].repeat(t2_dict.size(0),1)), dim=0)
                # output_ie[:,x,:] = ie_[t2_idx,t1_idx].unsqueeze(0)
                target_ie[:,x,:] = ie_[t2_idx,t1_idx].unsqueeze(0)
        # output_ie = output_ie * batch.mask_brain
        target_ie = target_ie * batch.mask_brain
        ###

        img_acq1 = img_acq1 * batch.mask_brain
        img_acq2 = img_acq2 * batch.mask_brain
        img_acq3 = img_acq3 * batch.mask_brain
        img_acq4 = img_acq4 * batch.mask_brain
        img_acq5 = img_acq5 * batch.mask_brain

        output_img1 = output_img1 * batch.mask_brain
        output_img2 = output_img2 * batch.mask_brain
        output_img3 = output_img3 * batch.mask_brain
        output_img4 = output_img4 * batch.mask_brain
        output_img5 = output_img5 * batch.mask_brain

        if batch.mask_brain.sum() == 0:
            target_t1 = target_t1 + 1e-5
            target_t2 = target_t2 + 1e-5
            target_pd = target_pd + 1e-5
            target_ie = target_ie + 1e-5
            output_t1 = output_t1 + 1e-5
            output_t2 = output_t2 + 1e-5
            output_pd = output_pd + 1e-5
            output_ie = output_ie + 1e-5
            output_b1 = output_b1 + 1e-5
            img_acq1 = img_acq1 + 1e-5
            img_acq2 = img_acq2 + 1e-5
            img_acq3 = img_acq3 + 1e-5
            img_acq4 = img_acq4 + 1e-5
            img_acq5 = img_acq5 + 1e-5
            output_img1 = output_img1 + 1e-5
            output_img2 = output_img2 + 1e-5
            output_img3 = output_img3 + 1e-5
            output_img4 = output_img4 + 1e-5
            output_img5 = output_img5 + 1e-5
        
        # ie = output_ie.squeeze(0).detach().cpu().numpy() # for previous dict_v4 IE
        # sio.savemat('ie_tmp/ie_s' + str(batch_idx+1) + '.mat',{'ie':ie}) # for previous dict_v4 IE

        def loss_tv(img):
            pixel_dif1 = img[..., 1:,:] - img[..., :-1,:]
            pixel_dif2 = img[..., :,1:] - img[..., :,:-1]
            reduce_axes = (-2,-1)
            res1 = pixel_dif1.abs().sum(dim=reduce_axes)
            res2 = pixel_dif2.abs().sum(dim=reduce_axes)
            return (res1 + res2) / (img.shape[-1] * img.shape[-2])
        loss_img1_tv = loss_tv(output_img1)
        loss_img2_tv = loss_tv(output_img2)
        loss_img3_tv = loss_tv(output_img3)
        loss_img4_tv = loss_tv(output_img4)
        loss_img5_tv = loss_tv(output_img5)

        loss_weight_t1 = 0
        loss_weight_t2 = 0
        loss_weight_pd = 0
        loss_weight_ie = 0
        loss_weight_img1 = 1
        loss_weight_img2 = 1
        loss_weight_img3 = 1
        loss_weight_img4 = 1
        loss_weight_img5 = 1
        loss_weight_img1_tv = 0
        loss_weight_img2_tv = 0
        loss_weight_img3_tv = 0
        loss_weight_img4_tv = 0
        loss_weight_img5_tv = 0

        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value_t1": batch.max_value_t1,
            "max_value_t2": batch.max_value_t2,
            "max_value_pd": batch.max_value_pd,
            "max_value_ie": torch.ones_like(batch.max_value_t1),
            "output_t1": output_t1,
            "output_t2": output_t2,
            "output_pd": output_pd,
            "output_ie": output_ie,
            "output_b1": output_b1,
            "output_img1": torch.abs(output_img1),
            "output_img2": torch.abs(output_img2),
            "output_img3": torch.abs(output_img3),
            "output_img4": torch.abs(output_img4),
            "output_img5": torch.abs(output_img5),
            "target_t1": target_t1,
            "target_t2": target_t2,
            "target_pd": target_pd,
            "target_ie": target_ie,
            "target_img1": torch.abs(img_acq1),
            "target_img2": torch.abs(img_acq2),
            "target_img3": torch.abs(img_acq3),
            "target_img4": torch.abs(img_acq4),
            "target_img5": torch.abs(img_acq5),
            "val_loss_t1": self.loss_l2_t1(output_t1.unsqueeze(1), target_t1.unsqueeze(1)),
            "val_loss_t2": self.loss_l2_t2(output_t2.unsqueeze(1), target_t2.unsqueeze(1)),
            "val_loss_pd": self.loss_l2_pd(output_pd.unsqueeze(1) / output_pd.max(), target_pd.unsqueeze(1) / target_pd.max()),
            "val_loss_ie": self.loss_l2_ie(output_ie.unsqueeze(1), target_ie.unsqueeze(1)),
            "val_loss_img1": self.loss_l2_img1(output_img1.unsqueeze(1), img_acq1.unsqueeze(1)),
            "val_loss_img2": self.loss_l2_img2(output_img2.unsqueeze(1), img_acq2.unsqueeze(1)),
            "val_loss_img3": self.loss_l2_img3(output_img3.unsqueeze(1), img_acq3.unsqueeze(1)),
            "val_loss_img4": self.loss_l2_img4(output_img4.unsqueeze(1), img_acq4.unsqueeze(1)),
            "val_loss_img5": self.loss_l2_img5(output_img5.unsqueeze(1), img_acq5.unsqueeze(1)),
            "loss_weight_t1": loss_weight_t1,
            "loss_weight_t2": loss_weight_t2,
            "loss_weight_pd": loss_weight_pd,
            "loss_weight_ie": loss_weight_ie,
            "loss_weight_img1": loss_weight_img1,
            "loss_weight_img2": loss_weight_img2,
            "loss_weight_img3": loss_weight_img3,
            "loss_weight_img4": loss_weight_img4,
            "loss_weight_img5": loss_weight_img5,
        }


    def test_step(self, batch, batch_idx):
        # ie_mat = sio.loadmat('ie_tmp/ie_s' + str(batch_idx+1) + '.mat') # for previous dict_v4 IE
        # ie = torch.from_numpy(ie_mat['ie']) # for previous dict_v4 IE

        output_t1, output_t2, output_pd, output_ie, \
        output_img1, output_img2, output_img3, output_img4, output_img5 = \
            self(batch.masked_kspace_acq1, batch.masked_kspace_acq2, batch.masked_kspace_acq3, batch.masked_kspace_acq4, batch.masked_kspace_acq5, \
                batch.mask_acq1, batch.mask_acq2, batch.mask_acq3, batch.mask_acq4, batch.mask_acq5, batch.mask_brain, \
                # batch.b1, ie, batch.max_value_t1, batch.max_value_t2, batch.max_value_pd, batch.num_low_frequencies) # for previous dict_v4 IE
                batch.b1, batch.ie, batch.max_value_t1, batch.max_value_t2, batch.max_value_pd, batch.num_low_frequencies)

        # check for FLAIR 203
        if output_t1.shape[-1] < batch.crop_size[1]:
            crop_size = (output_t1.shape[-1], output_t1.shape[-1])
        else:
            crop_size = batch.crop_size

        output_t1 = transforms_qalas.center_crop(output_t1, crop_size)
        output_t2 = transforms_qalas.center_crop(output_t2, crop_size)
        output_pd = transforms_qalas.center_crop(output_pd, crop_size)
        output_ie = transforms_qalas.center_crop(output_ie, crop_size)

        output_t1 = output_t1 * batch.mask_brain
        output_t2 = output_t2 * batch.mask_brain
        output_pd = output_pd * batch.mask_brain
        output_ie = output_ie * batch.mask_brain

        return {
            "fname": batch.fname,
            "slice": batch.slice_num,
            "output_t1": output_t1.cpu().numpy(),
            "output_t2": output_t2.cpu().numpy(),
            "output_pd": output_pd.cpu().numpy(),
            "output_ie": output_ie.cpu().numpy(),
        }

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModuleQALAS_MAP.add_model_specific_args(parser)

        # param overwrites

        # network params
        parser.add_argument(
            "--num_cascades",
            default=12,
            type=int,
            help="Number of QALAS cascades",
        )
        parser.add_argument(
            "--pools",
            default=4,
            type=int,
            help="Number of U-Net pooling layers in QALAS blocks",
        )
        parser.add_argument(
            "--chans",
            default=18,
            type=int,
            help="Number of channels for U-Net in QALAS blocks",
        )
        parser.add_argument(
            "--maps_chans",
            default=32,
            type=int,
            help="Number of channels for mapping CNN in QALAS",
        )
        parser.add_argument(
            "--maps_layers",
            default=5,
            type=float,
            help="Number of layers for mapping CNN in QALAS",
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.0003, type=float, help="Adam learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma",
            default=0.1,
            type=float,
            help="Extent to which step size should be decreased",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )

        return parser
