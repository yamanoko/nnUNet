# nnU-Net Copilot Instructions

## Project Overview
nnU-Net is a self-configuring deep learning framework for medical image segmentation. It automatically adapts U-Net architectures to new datasets based on dataset fingerprinting—no manual tuning required.

## Architecture

### Core Pipeline Flow
1. **Dataset Preparation** → `nnunetv2/dataset_conversion/` - Convert data to nnU-Net format
2. **Fingerprinting & Planning** → `nnunetv2/experiment_planning/` - Analyze dataset properties, generate plans
3. **Preprocessing** → `nnunetv2/preprocessing/` - Resample, normalize based on plans
4. **Training** → `nnunetv2/training/nnUNetTrainer/` - Train with auto-configured hyperparameters
5. **Inference** → `nnunetv2/inference/predict_from_raw_data.py` - Sliding window prediction with TTA
6. **Postprocessing** → `nnunetv2/postprocessing/` - Connected component analysis

### Key Directories
- `nnunetv2/training/nnUNetTrainer/variants/` - All trainer customizations (loss, augmentation, architecture)
- `nnunetv2/imageio/` - Pluggable readers/writers for different formats (NIfTI, TIFF, PNG)
- `nnunetv2/utilities/plans_handling/` - Plan file parsing and configuration management

## Environment Variables (Required)
```bash
nnUNet_raw=/path/to/raw          # Raw datasets (DatasetXXX_Name format)
nnUNet_preprocessed=/path/to/preprocessed  # Use SSD for speed
nnUNet_results=/path/to/results  # Trained models
```

## CLI Commands
All commands prefixed with `nnUNetv2_`. Key workflows:
```bash
# Full pipeline
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
nnUNetv2_train DATASET_ID CONFIG FOLD  # CONFIG: 2d, 3d_fullres, 3d_lowres, 3d_cascade_fullres
nnUNetv2_predict -i INPUT -o OUTPUT -d DATASET_ID -c CONFIG

# Evaluation & ensemble
nnUNetv2_find_best_configuration DATASET_ID  # Requires --npz during training
nnUNetv2_ensemble -i FOLDER1 FOLDER2 -o OUTPUT
```

## Dataset Format Convention
```
nnUNet_raw/DatasetXXX_Name/
├── dataset.json           # Metadata: channel_names, labels, file_ending
├── imagesTr/             # {case_id}_{XXXX}.{ext} where XXXX=channel (0000, 0001...)
├── labelsTr/             # {case_id}.{ext}
└── imagesTs/             # Optional test images
```
- Use `generate_dataset_json()` from `nnunetv2/dataset_conversion/generate_dataset_json.py`
- Labels must be consecutive integers starting from 0 (background)

## Extending nnU-Net

### Custom Trainer Pattern
Inherit from `nnUNetTrainer` and override specific methods. Place in `nnunetv2/training/nnUNetTrainer/variants/`:
```python
# Example: Custom loss (see variants/loss/nnUNetTrainerDiceLoss.py)
class nnUNetTrainerCustomLoss(nnUNetTrainer):
    def _build_loss(self):
        # Return your loss wrapped in DeepSupervisionWrapper if needed
```

Use custom trainer: `nnUNetv2_train DATASET CONFIG FOLD -tr nnUNetTrainerCustomLoss`

### Key Override Points in nnUNetTrainer
- `_build_loss()` - Loss function
- `configure_rotation_dummyDA_mirroring_and_inital_patch_size()` - Data augmentation
- `build_network_architecture()` - Network topology
- `configure_optimizers()` - Optimizer/scheduler

### Custom Image I/O
Inherit from `BaseReaderWriter` in `nnunetv2/imageio/`, register in `reader_writer_registry.py`

## Code Conventions
- Plans files (`.json`) control all configuration—edit these rather than code when possible
- Network outputs raw logits (no softmax/sigmoid)—nnU-Net applies activation during inference
- Deep supervision is standard; wrap losses with `DeepSupervisionWrapper`
- Use `batchgenerators`/`batchgeneratorsv2` for data augmentation transforms

## Testing
Integration tests in `nnunetv2/tests/integration_tests/` - test full pipeline on dummy datasets. Run preprocessing + training + inference to validate changes.

## Common Pitfalls
- Always use `--verify_dataset_integrity` on first preprocessing run
- For `nnUNetv2_find_best_configuration`, training must use `--npz` flag
- GPU selection via `CUDA_VISIBLE_DEVICES=X`, not command flags
- Data augmentation workers: set `nnUNet_n_proc_DA` env var based on CPU/GPU ratio
