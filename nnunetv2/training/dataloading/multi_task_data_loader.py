"""
Multi-Task Data Loader for nnU-Net

This module provides a data loader that handles multiple segmentation targets (tasks).
Each task has its own label map channel, enabling multi-task learning where a single
pixel can have different labels for different tasks.

Data format:
- Images: Same as standard nnU-Net (C, X, Y, Z)
- Labels: Multi-channel (num_tasks, X, Y, Z) where each channel contains labels for one task

The labels should be stored in the preprocessed folder as:
- {case_id}_seg.npy: Shape (num_tasks, X, Y, Z) instead of (1, X, Y, Z)
"""

import os
import warnings
from typing import Union, Tuple, List, Dict

import numpy as np
import torch
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile
from threadpoolctl import threadpool_limits

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetBaseDataset
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.utilities.label_handling.multi_task_label_handling import MultiTaskLabelManager
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd


class nnUNetDataLoaderMultiTask(nnUNetDataLoader):
    """
    Data loader for multi-task segmentation.
    
    Handles loading of multi-channel label maps where each channel corresponds
    to a different segmentation task.
    
    The key difference from standard nnUNetDataLoader:
    - seg has shape (num_tasks, X, Y, Z) instead of (1, X, Y, Z)
    - Each task's labels are processed independently for foreground sampling
    """
    
    def __init__(
        self,
        data: nnUNetBaseDataset,
        batch_size: int,
        patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
        final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
        label_manager: MultiTaskLabelManager,
        oversample_foreground_percent: float = 0.0,
        sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
        pad_sides: Union[List[int], Tuple[int, ...]] = None,
        probabilistic_oversampling: bool = False,
        transforms=None,
        foreground_sampling_strategy: str = "any_task"
    ):
        """
        Args:
            data: Dataset object
            batch_size: Batch size
            patch_size: Patch size for sampling
            final_patch_size: Final patch size after augmentation
            label_manager: MultiTaskLabelManager instance
            oversample_foreground_percent: Percentage of samples to oversample foreground
            sampling_probabilities: Per-sample sampling probabilities
            pad_sides: Padding on each side
            probabilistic_oversampling: Use probabilistic oversampling
            transforms: Data augmentation transforms
            foreground_sampling_strategy: Strategy for foreground sampling
                - "any_task": Sample foreground from any task (union)
                - "all_tasks": Sample foreground present in all tasks (intersection)
                - "primary": Use first task for foreground sampling
        """
        # Initialize base class
        # We pass a dummy LabelManager-compatible object to parent __init__
        # then override with our multi-task label manager
        
        # Store multi-task specific attributes
        self.multi_task_label_manager = label_manager
        self.foreground_sampling_strategy = foreground_sampling_strategy
        self.num_tasks = label_manager.num_tasks
        self.task_names = label_manager.task_names
        
        # Create a wrapper that provides necessary interface for parent class
        self._init_as_base(
            data, batch_size, patch_size, final_patch_size, 
            oversample_foreground_percent, sampling_probabilities,
            pad_sides, probabilistic_oversampling, transforms
        )
    
    def _init_as_base(
        self,
        data: nnUNetBaseDataset,
        batch_size: int,
        patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
        final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
        oversample_foreground_percent: float,
        sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray],
        pad_sides: Union[List[int], Tuple[int, ...]],
        probabilistic_oversampling: bool,
        transforms
    ):
        """Initialize base DataLoader attributes without calling parent __init__."""
        DataLoader.__init__(
            self, data, batch_size, 1, None, True, False, True, sampling_probabilities
        )
        
        if len(patch_size) == 2:
            final_patch_size = (1, *patch_size)
            patch_size = (1, *patch_size)
            self.patch_size_was_2d = True
        else:
            self.patch_size_was_2d = False
        
        self.indices = data.identifiers
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        
        self.need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)
        if pad_sides is not None:
            if self.patch_size_was_2d:
                pad_sides = (0, *pad_sides)
            for d in range(len(self.need_to_pad)):
                self.need_to_pad[d] += pad_sides[d]
        
        self.num_channels = None
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.sampling_probabilities = sampling_probabilities
        
        # For multi-task, we need to handle class_locations differently
        self.annotated_classes_key = self._build_annotated_classes_key()
        self.has_ignore = self.multi_task_label_manager.has_any_ignore_label()
        
        self.get_do_oversample = (
            self._oversample_last_XX_percent 
            if not probabilistic_oversampling 
            else self._probabilistic_oversampling
        )
        self.transforms = transforms
    
    def _build_annotated_classes_key(self) -> tuple:
        """Build annotated classes key for foreground sampling."""
        # For multi-task, we combine all foreground labels across tasks
        all_labels = set()
        for task_name in self.task_names:
            lm = self.multi_task_label_manager.get_task_label_manager(task_name)
            all_labels.update(lm.all_labels)
        return tuple([-1] + sorted(all_labels))
    
    def determine_shapes(self):
        """Determine data and seg shapes."""
        data, seg, seg_prev, properties = self._data.load_case(self._data.identifiers[0])
        num_color_channels = data.shape[0]
        
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        
        # For multi-task, seg has shape (num_tasks, X, Y, Z)
        num_seg_channels = seg.shape[0]  # Should be num_tasks
        if seg_prev is not None:
            num_seg_channels += 1
        seg_shape = (self.batch_size, num_seg_channels, *self.patch_size)
        
        return data_shape, seg_shape
    
    def get_bbox(
        self, 
        data_shape: np.ndarray, 
        force_fg: bool, 
        class_locations: Union[dict, None],
        overwrite_class: Union[int, Tuple[int, ...]] = None, 
        verbose: bool = False
    ):
        """
        Get bounding box for patch sampling.
        
        For multi-task, class_locations should contain foreground locations
        considering all tasks according to the sampling strategy.
        """
        # Most of the logic is the same as parent, but we may need to handle
        # class_locations differently for multi-task
        return super().get_bbox(data_shape, force_fg, class_locations, overwrite_class, verbose)
    
    def generate_train_batch(self):
        """Generate a training batch with multi-task labels."""
        selected_keys = self.get_indices()
        
        # Preallocate memory
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        
        for j, i in enumerate(selected_keys):
            force_fg = self.get_do_oversample(j)
            
            data, seg, seg_prev, properties = self._data.load_case(i)
            shape = data.shape[1:]
            
            # Get class locations (may need adaptation for multi-task)
            class_locations = properties.get('class_locations', None)
            
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, class_locations)
            bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]
            
            data_all[j] = crop_and_pad_nd(data, bbox, 0)
            
            # Crop seg - handles multi-channel (num_tasks, X, Y, Z)
            seg_cropped = crop_and_pad_nd(seg, bbox, -1)
            if seg_prev is not None:
                seg_cropped = np.vstack((seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)[None]))
            seg_all[j] = seg_cropped
        
        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]
        
        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    
                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(
                            **{'image': data_all[b], 'segmentation': seg_all[b]}
                        )
                        images.append(tmp['image'])
                        segs.append(tmp['segmentation'])
                    
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        # Deep supervision case
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images
            
            return {'data': data_all, 'target': seg_all, 'keys': selected_keys}
        
        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}


class MultiTaskSegmentationTransform:
    """
    Wrapper transform that applies transforms to multi-task segmentation labels.
    
    This ensures that spatial transforms are applied consistently across all task channels.
    """
    
    def __init__(self, base_transforms, task_names: List[str]):
        """
        Args:
            base_transforms: Base transforms to apply
            task_names: List of task names (for reference)
        """
        self.base_transforms = base_transforms
        self.task_names = task_names
    
    def __call__(self, **data_dict):
        """
        Apply transforms.
        
        The segmentation tensor has shape (num_tasks, X, Y, Z).
        Spatial transforms should be applied to all channels together.
        """
        return self.base_transforms(**data_dict)


def convert_multi_task_seg_to_per_task_targets(
    seg: torch.Tensor,
    label_manager: MultiTaskLabelManager,
    deep_supervision_scales: List[List[float]] = None
) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
    """
    Convert multi-channel segmentation to per-task targets.
    
    Args:
        seg: Segmentation tensor of shape (B, num_tasks, *spatial) or 
             list of such tensors (for deep supervision)
        label_manager: MultiTaskLabelManager instance
        deep_supervision_scales: Scales for deep supervision (if applicable)
    
    Returns:
        Dict mapping task names to target tensors.
        If deep supervision, each value is a list of tensors at different scales.
    """
    task_names = label_manager.task_names
    
    if isinstance(seg, list):
        # Deep supervision case - seg is a list of tensors at different scales
        result = {task_name: [] for task_name in task_names}
        
        for scale_idx, seg_at_scale in enumerate(seg):
            for task_idx, task_name in enumerate(task_names):
                task_seg = seg_at_scale[:, task_idx:task_idx+1]  # (B, 1, *spatial)
                task_lm = label_manager.get_task_label_manager(task_name)
                
                # Convert to one-hot if using regions
                if task_lm.has_regions:
                    task_target = _convert_to_regions(task_seg, task_lm)
                else:
                    task_target = task_seg
                
                result[task_name].append(task_target)
        
        return result
    else:
        # Standard case - single segmentation tensor
        result = {}
        
        for task_idx, task_name in enumerate(task_names):
            task_seg = seg[:, task_idx:task_idx+1]  # (B, 1, *spatial)
            task_lm = label_manager.get_task_label_manager(task_name)
            
            # Convert to one-hot if using regions
            if task_lm.has_regions:
                task_target = _convert_to_regions(task_seg, task_lm)
            else:
                task_target = task_seg
            
            result[task_name] = task_target
        
        return result


def _convert_to_regions(
    seg: torch.Tensor, 
    task_lm
) -> torch.Tensor:
    """
    Convert integer segmentation to region-based one-hot encoding.
    
    Args:
        seg: Integer segmentation tensor (B, 1, *spatial)
        task_lm: SingleTaskLabelManager with region information
    
    Returns:
        One-hot tensor (B, num_regions, *spatial)
    """
    regions = task_lm.all_regions
    if regions is None:
        return seg
    
    b, _, *spatial = seg.shape
    num_regions = len(regions)
    
    result = torch.zeros((b, num_regions, *spatial), dtype=torch.float32, device=seg.device)
    
    for region_idx, region in enumerate(regions):
        if isinstance(region, (tuple, list)):
            mask = torch.zeros_like(seg, dtype=torch.bool)
            for label in region:
                mask |= (seg == label)
        else:
            mask = (seg == region)
        result[:, region_idx] = mask.float().squeeze(1)
    
    return result
