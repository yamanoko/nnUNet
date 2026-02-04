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
    
    "labels": {
        "background": 0,
        "foreground": 1
    },
    
    "numTraining": 100
}
```

**Key differences from standard dataset.json:**
- Uses `"tasks"` instead of `"labels"` for defining task-specific classes
- Each task has its own `"labels"` dictionary
- Each task can have a different number of classes
- Each task can independently use region-based training (by specifying labels as lists)

**Important:** The `"labels"` key (with `background` and `foreground`) is **required** for compatibility with nnU-Net's experiment planning and preprocessing pipeline. This allows `nnUNetv2_plan_and_preprocess` to run without errors. The actual task-specific labels in `"tasks"` are used during training.

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

### From Multi-Task to Single-Task (Complete Workflow)

A key feature is that multi-task pretrained models can be finetuned to single-task models. The encoder and decoder weights are transferred, while segmentation heads are reinitialized.

Below is the complete command sequence for pretraining with multi-task and finetuning on a single-task dataset:

#### Step 1: Plan and Preprocess the Finetuning Dataset

First, run experiment planning and preprocessing on your target (single-task) dataset:

```bash
nnUNetv2_plan_and_preprocess -d FINETUNING_DATASET --verify_dataset_integrity
```

#### Step 2: Extract Fingerprint of the Pretraining Dataset

Extract the dataset fingerprint of your multi-task pretraining dataset (if not yet available):

```bash
nnUNetv2_extract_fingerprint -d MULTITASK_DATASET
```

#### Step 3: Transfer Plans from Finetuning to Pretraining Dataset

Transfer the plans from the finetuning dataset to the pretraining dataset to ensure matching network topologies:

```bash
nnUNetv2_move_plans_between_datasets -s FINETUNING_DATASET -t MULTITASK_DATASET -sp nnUNetPlans -tp nnUNetPlans_pretrain
```

This ensures that the network architecture (patch size, topology, batch size) matches between pretraining and finetuning.

#### Step 4: Preprocess the Multi-Task Pretraining Dataset

Run preprocessing on the pretraining dataset with the transferred plans:

```bash
nnUNetv2_preprocess -d MULTITASK_DATASET -plans_name nnUNetPlans_pretrain
```

#### Step 5: Pretrain with Multi-Task

Train the multi-task model using all available data (fold `all`):

```bash
nnUNetv2_train MULTITASK_DATASET 3d_fullres all -tr nnUNetTrainerMultiTask -p nnUNetPlans_pretrain
```

The trained model will be saved at:
```
nnUNet_results/DatasetXXX_MultiTask/nnUNetTrainerMultiTask__nnUNetPlans_pretrain__3d_fullres/fold_all/checkpoint_final.pth
```

#### Step 6: Finetune on Single-Task Dataset

Finally, finetune on your target single-task dataset using the pretrained weights:

```bash
nnUNetv2_train FINETUNING_DATASET 3d_fullres FOLD \
    -pretrained_weights /path/to/nnUNet_results/DatasetXXX_MultiTask/nnUNetTrainerMultiTask__nnUNetPlans_pretrain__3d_fullres/fold_all/checkpoint_final.pth
```

Replace `FOLD` with the desired fold number (0-4) or `all`.

**Note**: When loading multi-task pretrained weights, only the encoder and decoder weights are transferred. The multi-task segmentation heads (`.seg_layers.`, `.multi_head_seg_layers.`, `.task_heads.`) are automatically skipped, and the single-task segmentation head is initialized randomly.

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
| `Primus` | ✅ Fully supported | Vision Transformer architecture (see dedicated section below) |
| Custom architectures | ✅ Supported | Any architecture with `seg_layers` attribute |

The wrapper approach automatically detects and replaces segmentation heads for any architecture, so you can use:

```bash
# With ResEnc preset
nnUNetv2_train DATASET 3d_fullres_resenc FOLD -tr nnUNetTrainerMultiTask

# With Primus (dedicated trainers)
nnUNetv2_train DATASET 3d_fullres FOLD -tr nnUNet_Primus_M_MultiTask_Trainer
```

## Primus Multi-Task Training

Primus is a Vision Transformer (ViT) based architecture for 3D medical image segmentation. Due to its unique architecture (EVA encoder + PatchDecode), Primus has **dedicated multi-task trainers** that are optimized for its structure.

### Architecture Overview

```
                         ┌─→ PatchDecode A → Task A output
Input → PatchEmbed → EVA ─┤
                         └─→ PatchDecode B → Task B output
                                ⋮
                         └─→ PatchDecode N → Task N output
```

Unlike standard nnU-Net architectures that use `seg_layers`, Primus uses `up_projection` (PatchDecode) for the segmentation head. The `PrimusMultiHeadWrapper` handles this by creating task-specific PatchDecode instances while sharing the EVA encoder.

### Available Trainers

| Trainer | Model Size | embed_dim | depth | heads | Memory |
|---------|-----------|-----------|-------|-------|--------|
| `nnUNet_Primus_S_MultiTask_Trainer` | Small | 396 | 12 | 6 | ~8GB |
| `nnUNet_Primus_B_MultiTask_Trainer` | Base | 792 | 12 | 12 | ~16GB |
| `nnUNet_Primus_M_MultiTask_Trainer` | Medium | 864 | 16 | 12 | ~24GB |
| `nnUNet_Primus_L_MultiTask_Trainer` | Large | 1056 | 24 | 16 | ~40GB |

### Usage

```bash
# Plan and preprocess
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity

# Train with Primus-M (recommended starting point)
nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNet_Primus_M_MultiTask_Trainer

# Train with Primus-S (for limited GPU memory)
nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNet_Primus_S_MultiTask_Trainer
```

### Important Constraints

1. **No Deep Supervision**: Primus does not support deep supervision due to its single-resolution output architecture. The `enable_deep_supervision` flag is automatically set to `False`.

2. **Patch Size Requirements**: Input patch size must be divisible by 8 (the default patch embed size).

3. **3D Only**: Primus is designed for 3D volumetric data only.

### Pretraining and Finetuning with Primus

A key advantage of multi-task Primus training is that the pretrained EVA encoder can be transferred to single-task Primus models.

#### Complete Workflow: Multi-Task Pretraining → Single-Task Finetuning

**Step 1: Plan and preprocess the finetuning (single-task) dataset**
```bash
nnUNetv2_plan_and_preprocess -d FINETUNING_DATASET --verify_dataset_integrity
```

**Step 2: Extract fingerprint of the pretraining (multi-task) dataset**
```bash
nnUNetv2_extract_fingerprint -d MULTITASK_DATASET
```

**Step 3: Transfer plans from finetuning to pretraining dataset**
```bash
nnUNetv2_move_plans_between_datasets \
    -s FINETUNING_DATASET \
    -t MULTITASK_DATASET \
    -sp nnUNetPlans \
    -tp nnUNetPlans_pretrain
```

**Step 4: Preprocess the multi-task dataset with transferred plans**
```bash
nnUNetv2_preprocess -d MULTITASK_DATASET -plans_name nnUNetPlans_pretrain
```

**Step 5: Pretrain with multi-task Primus**
```bash
nnUNetv2_train MULTITASK_DATASET 3d_fullres all \
    -tr nnUNet_Primus_M_MultiTask_Trainer \
    -p nnUNetPlans_pretrain
```

**Step 6: Finetune on single-task dataset**
```bash
nnUNetv2_train FINETUNING_DATASET 3d_fullres FOLD \
    -tr nnUNet_Primus_M_Trainer \
    -pretrained_weights /path/to/checkpoint_final.pth
```

#### What Gets Transferred

| Component | Transferred | Notes |
|-----------|-------------|-------|
| `down_projection` (PatchEmbed) | ✅ Yes | Input channels auto-adjusted if different |
| `eva` (ViT Encoder) | ✅ Yes | All transformer weights |
| `register_tokens` | ✅ Yes | If present |
| `mask_token` | ✅ Yes | Buffer |
| `task_up_projections` | ❌ No | Multi-task decoders skipped |
| `up_projection` | ❌ No | Randomly initialized |

The EVA encoder contains the majority of the model parameters, so pretraining provides significant benefit.

### Custom Task Weights

Create a custom trainer with specific task weights:

```python
from nnunetv2.training.nnUNetTrainer.primus.primus_trainers import AbstractPrimusMultiTask, PRIMUS_CONFIGS

class nnUNet_Primus_M_MultiTask_CustomWeights(AbstractPrimusMultiTask):
    def __init__(self, plans, configuration, fold, dataset_json, device):
        super().__init__(
            plans, configuration, fold, dataset_json, device,
            task_loss_weights={"organ": 1.0, "lesion": 2.0}
        )
    
    @property
    def primus_config(self):
        return PRIMUS_CONFIGS["M"]
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
- Works with any architecture (PlainConvUNet, ResidualEncoderUNet, custom)
- No need to reimplement or modify the base architecture
- Automatically benefits from improvements to base architectures
- `multi_head_seg_layers`: Contains task-specific 1×1 convolution heads
- Deep supervision is supported with multiple outputs per task at different resolutions

#### Primus Multi-Head Architecture

For Primus, the `PrimusMultiHeadWrapper` is used instead, as Primus has a fundamentally different structure:

1. **Shared encoding**: `down_projection` (PatchEmbed) + `eva` (ViT encoder) are shared
2. **Task-specific decoding**: Each task gets its own `PatchDecode` instance stored in `task_up_projections`
3. **No deep supervision**: Primus outputs at a single resolution only

```python
# PrimusMultiHeadWrapper structure
self.base_network        # Original Primus (down_projection + eva)
self.task_up_projections # ModuleDict: {task_name: PatchDecode}
```

### Loss Function

`MultiTaskLoss` computes a weighted combination of per-task losses:

```python
total_loss = Σ(weight_i × loss_i(output_i, target_i))
```

Each task uses either DC+CE loss (standard) or DC+BCE loss (region-based) depending on its configuration.

### Weight Transfer

When loading pretrained weights, the following patterns are skipped:
- `.seg_layers.` (standard nnU-Net heads)
- `.multi_head_seg_layers.` (multi-task heads in legacy MultiHeadUNet)
- `.task_heads.` (task-specific heads in legacy MultiHeadUNet)
- `.task_seg_layers.` (task-specific heads in MultiHeadSegmentationWrapper)
- `.task_up_projections.` (task-specific PatchDecode heads in PrimusMultiHeadWrapper)

This ensures encoder/decoder weights transfer while heads are reinitialized.
