# MesoscaleQALAS-7T
## Reconstrcution: MZS-SSL
The baseline zero-shot reconstrcution code is forked from https://github.com/byaman14/ZS-SSL & https://github.com/yohan-jun/Zero-DeepSub
### Installation
Dependencies are given in ` MZS-SSL-QALAS/environment_tf2.yml` and can be installed with``conda env create -f environment_tf2.yml``.
### Reconstruction Pipeline

#### 1. Preprocessing

Run the following MATLAB script to prepare the raw data for zero-shot self-supervised reconstruction:

```matlab
prepare_data_for_QALAS_MZS_SSL_recon.m
```

#### 2. Training

Run one of the following scripts to perform multi-contrast and multi-slice zero-shot self-supervised learning training:

```bash
python zs_ssl_train_muti_mask_batch_v10_ms.py
```

Before training, reconstruction hyperparameters can be adjusted in `parser_ops.py` under the corresponding reconstruction folder.
#### 3. Inference

Run one of the following notebooks to load the checkpoints saved during training and perform inference:

```bash
zs_ssl_inference_ms_qalas.ipynb
```

## Parameter Estimation: SSL-UHF
### Method Overview

The mapping CNN takes the five QALAS contrast images as input and predicts:

- **T1:** longitudinal relaxation time
- **T2:** transverse relaxation time
- **PD:** proton density
- **Cprep:** a correction factor for nonideal T2-preparation signal behavior
- **IE:** inversion efficiency

The predicted parameters are passed through the QALAS signal model to synthesize the five contrasts. The training objective includes acquired-versus-synthesized image consistency and spatial regularization. The current training module also assigns a nonzero weight to an IE reference loss; the T1, T2, and PD reference-map loss weights are set to zero.

`Cprep` accounts for effective signal contributions associated with imperfect T2 preparation, including residual longitudinal magnetization and other nonideal signal pathways. Together with B1+ modeling, this correction is intended to reduce B1+-related T2 shading and improve quantitative-map uniformity at 7T. **The current implementation names this learned quantity `t2eff`** (and saves it as `reconstruction_t2eff

### Installation

The project is derived from the original SSL-QALAS/fastMRI codebase. From the `SSL-UHF` directory:

```bash
conda env create -f environment.yml
conda activate ssl_qalas_test
pip install -e .
```

The provided environment file specifies Python 3.9 and the project's pinned dependencies. A CUDA-capable GPU is expected by the training script.

### Input Data

The mapping workflow uses:

1. Five co-registered 3D-QALAS contrast images.
2. A corresponding B1+ map, spatially aligned with the QALAS data.
3. Brain mask.
4. Sequence timing and preparation parameters matching the acquisition.

The data loader uses HDF5 (`.h5`) files with a layout such as:

```text
matlab/h5_data/
├── multicoil_train/
│   └── train_data.h5
└── multicoil_val/
    └── train_data.h5
```

The included `matlab/ssl_qalas_save_h5_from_dicom.m` prepares the HDF5 data. The loader in `fastmri/data/mri_data_qalas.py` reads five acquisitions (`kspace_acq1` through `kspace_acq5`), `reconstruction_b1`, `reconstruction_ie`, `mask_brain`, and reference T1/T2/PD maps. Acquisition masks (`mask_acq1` through `mask_acq5`) can also be stored in the HDF5 file.

### Training

From the `SSL-UHF` directory, enter `qalas/` and launch:

```bash
cd qalas
PYTHONPATH=/SSL-UHF:$PYTHONPATH \
python train_qalas.py \
    --data_path ../matlab/h5_data \
    --check_val_every_n_epoch 4 \
    --num_sanity_val_steps 0 \
    --use_dataset_cache_file False
```

Adjust PYTHONPATH`` and `--data_path` to the project and HDF5 parent directory. 

Sequence-specific settings (including turbo factor, echo spacing, TR, and T2-preparation timing) must match the actual 7T protocol. The current forward-model settings are located in `fastmri/models/qalas_map.py`.

### Inference

From `SSL-UHF/qalas`, run:

```bash
PYTHONPATH=/SSL-UHF:$PYTHONPATH \
python inference_qalas_map.py \
    --data_path ../matlab/h5_data/multicoil_val \
    --state_dict_file qalas_log/checkpoints/YOUR_CHECKPOINT.ckpt \
    --output_path qalas_log
```

Inference writes HDF5 files under `qalas_log/reconstructions/`. Each output contains `reconstruction_t1`, `reconstruction_t2`, `reconstruction_t2eff` (Cprep), `reconstruction_pd`, `reconstruction_ie`, and `reconstruction_b1`. The B1+ output is the supplied map, not an independently predicted parameter.

### Acknowledgment and Citation

This work builds on the official [SSL-QALAS repository](https://github.com/yohan-jun/SSL-QALAS). Please cite the original paper when using this code:

> Jun Y, Cho J, Wang X, Gee M, Grant PE, Bilgic B, Gagoski B. SSL-QALAS: Self-supervised learning for rapid multiparameter estimation in quantitative MRI using 3D-QALAS. *Magnetic Resonance in Medicine*. 2023;90(5):2019–2032.

Please retain the original project's copyright and license notices in any redistributed derivative code.
