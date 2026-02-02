# Multi-Task Segmentation Training

## Overview

nnU-Net now supports multi-task segmentation, where a single model can learn multiple independent segmentation tasks simultaneously. Each task has its own segmentation head, while sharing the encoder and decoder features.

This is useful when you have data with multiple types of annotations that are **independent** of each other. For example:
- Organ segmentation (liver, spleen, kidney) + Lesion segmentation (tumor, cyst)
- Cell type classification + Cell state classification
- Anatomical structure segmentation + Pathology segmentation

**Note**: This is different from [region-based training](region_based_training.md), which handles **hierarchical/overlapping** labels (e.g., whole tumor ⊃ tumor core ⊃ enhancing tumor).

## Architecture

```
                    ┌─→ Head A (softmax) → Task A output
Input → Encoder → Decoder ─┤
                    └─→ Head B (softmax) → Task B output
                           ⋮
                    └─→ Head N (softmax) → Task N output
```

The encoder and decoder are shared across all tasks, enabling efficient learning of shared representations. Each task has an independent segmentation head that produces task-specific outputs.

## Dataset Format

### Directory Structure

```
nnUNet_raw/DatasetXXX_MultiTaskExample/
├── dataset.json           # Multi-task format (see below)
├── imagesTr/             # {case_id}_{XXXX}.nii.gz (same as standard nnU-Net)
├── labelsTr/             # {case_id}.nii.gz (multi-channel: one channel per task)
└── imagesTs/             # Optional test images
```

### dataset.json Format

```json
{
    "channel_names": {
        "0": "CT"
    },
    "file_ending": ".nii.gz",
    
    "tasks": {
        "organ": {
            "labels": {
                "background": 0,
                "liver": 1,
                "spleen": 2,
                "kidney": 3
            }
        },
        "lesion": {
            "labels": {
                "background": 0,
                "tumor": 1,
                "cyst": 2
            }
        }
    },
    
    "numTraining": 100
}
```

**Key differences from standard dataset.json:**
- Uses `"tasks"` instead of `"labels"`
- Each task has its own `"labels"` dictionary
- Each task can have a different number of classes
- Each task can independently use region-based training (by specifying labels as lists)

### Label Files Format

Label files are **multi-channel**, where each channel contains the labels for one task:

```python
# Shape: (num_tasks, X, Y, Z)
# Example for 2 tasks:
# labels[0, :, :, :] = organ labels (0=background, 1=liver, 2=spleen, 3=kidney)
# labels[1, :, :, :] = lesion labels (0=background, 1=tumor, 2=cyst)
```

The order of channels must match the order of tasks in `dataset.json`.

## Usage

### Training

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainerMultiTask
```

### Custom Task Loss Weights

You can weight tasks differently by creating a custom trainer:

```python
class nnUNetTrainerMultiTaskCustomWeights(nnUNetTrainerMultiTask):
    def __init__(self, plans, configuration, fold, dataset_json, device):
        super().__init__(
            plans, configuration, fold, dataset_json, device,
            task_loss_weights={"organ": 1.0, "lesion": 2.0}  # Weight lesion task higher
        )
```

## Finetuning

### From Multi-Task to Single-Task

A key feature is that multi-task pretrained models can be finetuned to single-task models. The encoder and decoder weights are transferred, while segmentation heads are reinitialized:

```bash
# Pretrain with multi-task
nnUNetv2_train MULTITASK_DATASET 3d_fullres 0 -tr nnUNetTrainerMultiTask

# Finetune on single-task dataset
nnUNetv2_train SINGLETASK_DATASET 3d_fullres 0 \
    -pretrained_weights /path/to/multitask/checkpoint_final.pth
```

### From Single-Task to Multi-Task

Similarly, single-task pretrained weights can be loaded into multi-task models:

```bash
nnUNetv2_train MULTITASK_DATASET 3d_fullres 0 -tr nnUNetTrainerMultiTask \
    -pretrained_weights /path/to/singletask/checkpoint_final.pth
```

## Creating Multi-Channel Label Files

Here's an example script to create multi-channel label files:

```python
import nibabel as nib
import numpy as np

def create_multi_task_labels(organ_labels_path, lesion_labels_path, output_path):
    """
    Combine separate label files into multi-channel format.
    """
    organ_nii = nib.load(organ_labels_path)
    lesion_nii = nib.load(lesion_labels_path)
    
    organ_data = organ_nii.get_fdata()
    lesion_data = lesion_nii.get_fdata()
    
    # Stack into multi-channel (num_tasks, X, Y, Z)
    multi_task_labels = np.stack([organ_data, lesion_data], axis=0)
    
    # Save with same affine
    multi_task_nii = nib.Nifti1Image(multi_task_labels, organ_nii.affine)
    nib.save(multi_task_nii, output_path)

# Usage
create_multi_task_labels(
    'organ_labels/case_001.nii.gz',
    'lesion_labels/case_001.nii.gz',
    'labelsTr/case_001.nii.gz'
)
```

## Evaluation

During training, nnU-Net logs:
- Per-task Dice scores (e.g., `organ_dice`, `lesion_dice`)
- Combined mean foreground Dice across all tasks

Model selection is based on the combined Dice score.

## Inference

**Note**: Multi-task inference is not yet fully implemented. Current workaround is to:

1. Run inference to get multi-task outputs
2. Post-process to separate task outputs

```python
# Coming soon: nnUNetv2_predict with multi-task support
```

## Best Practices

1. **Balanced Tasks**: Try to balance the number of classes across tasks, or use task loss weights
2. **Related Tasks**: Multi-task learning works best when tasks are related and can benefit from shared features
3. **Sufficient Data**: Multi-task learning requires more data since the model must learn multiple tasks
4. **Task Order**: The order of tasks in `dataset.json` must match the channel order in label files

## Supported Architectures

Multi-task training supports **all nnU-Net architectures** through the `MultiHeadSegmentationWrapper`:

| Architecture | Support | Notes |
|-------------|---------|-------|
| `PlainConvUNet` | ✅ Fully supported | Standard nnU-Net architecture |
| `ResidualEncoderUNet` | ✅ Fully supported | ResEnc presets |
| `Primus` | ✅ Fully supported | Mamba-based architecture |
| Custom architectures | ✅ Supported | Any architecture with `seg_layers` attribute |

The wrapper approach automatically detects and replaces segmentation heads for any architecture, so you can use:

```bash
# With ResEnc preset
nnUNetv2_train DATASET 3d_fullres_resenc FOLD -tr nnUNetTrainerMultiTask

# With Primus (if available)
nnUNetv2_train DATASET 3d_fullres_primus FOLD -tr nnUNetTrainerMultiTask
```

## Limitations

- Inference pipeline is not fully integrated yet
- Region-based training per task is supported but not extensively tested
- Cascaded training with multi-task is not supported

## Technical Details

### Network Architecture

The `MultiHeadSegmentationWrapper` class wraps **any** nnU-Net architecture and adds multi-head functionality:

1. **Base network creation**: Standard network is created via `get_network_from_plans`
2. **Head replacement**: Original `seg_layers` are replaced with task-specific heads stored in `task_heads`
3. **Feature interception**: Decoder features are captured via forward hooks and passed to each task head

This approach has several advantages:
- Works with any architecture (PlainConvUNet, ResidualEncoderUNet, Primus, custom)
- No need to reimplement or modify the base architecture
- Automatically benefits from improvements to base architectures
- `multi_head_seg_layers`: Contains task-specific 1×1 convolution heads
- Deep supervision is supported with multiple outputs per task at different resolutions

### Loss Function

`MultiTaskLoss` computes a weighted combination of per-task losses:

```python
total_loss = Σ(weight_i × loss_i(output_i, target_i))
```

Each task uses either DC+CE loss (standard) or DC+BCE loss (region-based) depending on its configuration.

### Weight Transfer

When loading pretrained weights, the following patterns are skipped:
- `.seg_layers.` (standard nnU-Net heads)
- `.multi_head_seg_layers.` (multi-task heads)
- `.task_heads.` (task-specific heads)

This ensures encoder/decoder weights transfer while heads are reinitialized.
