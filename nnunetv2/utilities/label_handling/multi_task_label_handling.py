"""
Multi-Task Label Manager for nnU-Net

Manages label information for multiple segmentation tasks, where each task has its own
set of labels/classes. This enables multi-task learning where a single pixel can have
different labels for different tasks (e.g., organ type and lesion type).

Dataset JSON format for multi-task:
{
    "channel_names": {"0": "CT"},
    "file_ending": ".nii.gz",
    "tasks": {
        "task_A": {
            "labels": {
                "background": 0,
                "class_A1": 1,
                "class_A2": 2
            }
        },
        "task_B": {
            "labels": {
                "background": 0,
                "class_B1": 1,
                "class_B2": 2,
                "class_B3": 3
            }
        }
    },
    "numTraining": 100
}

Labels are stored in multi-channel format: each channel corresponds to one task.
E.g., for 2 tasks: seg.shape = (2, X, Y, Z) where seg[0] is task_A labels, seg[1] is task_B labels.
"""

from typing import Dict, List, Tuple, Union, Optional, Type
import numpy as np
import torch

from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.helpers import softmax_helper_dim0


class SingleTaskLabelManager:
    """
    Manages labels for a single task within a multi-task setup.
    Similar to the standard LabelManager but simplified for single task use.
    """
    def __init__(
        self,
        task_name: str,
        label_dict: Dict[str, int],
        regions_class_order: Optional[List[int]] = None
    ):
        self.task_name = task_name
        self.label_dict = label_dict
        self.regions_class_order = regions_class_order
        
        # Determine if this task uses region-based training
        self._has_regions = any(
            isinstance(v, (tuple, list)) and len(v) > 1 
            for v in label_dict.values()
        )
        
        # Get all labels
        self._all_labels = self._get_all_labels()
        self._regions = self._get_regions() if self._has_regions else None
        
        # Determine ignore label
        self._ignore_label = label_dict.get('ignore', None)
        
        # Set inference nonlinearity
        self.inference_nonlin = torch.sigmoid if self._has_regions else softmax_helper_dim0
    
    def _get_all_labels(self) -> List[int]:
        """Get all unique integer labels."""
        all_labels = set()
        for k, v in self.label_dict.items():
            if k == 'ignore':
                continue
            if isinstance(v, (tuple, list)):
                all_labels.update(int(x) for x in v)
            else:
                all_labels.add(int(v))
        return sorted(all_labels)
    
    def _get_regions(self) -> Optional[List[Union[int, Tuple[int, ...]]]]:
        """Get regions for region-based training."""
        if not self._has_regions:
            return None
        
        regions = []
        for k, v in self.label_dict.items():
            if k == 'ignore':
                continue
            # Skip background-only regions
            if isinstance(v, (tuple, list)):
                if len(set(v)) == 1 and v[0] == 0:
                    continue
                regions.append(tuple(v))
            elif v == 0:
                continue
            else:
                regions.append(v)
        return regions
    
    @property
    def has_regions(self) -> bool:
        return self._has_regions
    
    @property
    def has_ignore_label(self) -> bool:
        return self._ignore_label is not None
    
    @property
    def ignore_label(self) -> Optional[int]:
        return self._ignore_label
    
    @property
    def all_labels(self) -> List[int]:
        return self._all_labels
    
    @property
    def all_regions(self) -> Optional[List[Union[int, Tuple[int, ...]]]]:
        return self._regions
    
    @property
    def foreground_labels(self) -> List[int]:
        return [l for l in self._all_labels if l != 0]
    
    @property
    def foreground_regions(self) -> Optional[List[Union[int, Tuple[int, ...]]]]:
        if self._regions is None:
            return None
        return [r for r in self._regions if not (isinstance(r, int) and r == 0)]
    
    @property
    def num_segmentation_heads(self) -> int:
        """Number of output channels needed for this task."""
        if self.has_regions:
            return len(self.foreground_regions)
        else:
            return len(self.all_labels)


class MultiTaskLabelManager:
    """
    Manages labels for multiple segmentation tasks.
    
    Each task has its own label set and can independently use standard or region-based training.
    """
    
    def __init__(
        self,
        tasks_config: Dict[str, dict],
        inference_nonlin: Optional[Dict[str, callable]] = None
    ):
        """
        Args:
            tasks_config: Dict mapping task names to their configuration.
                         Each config should have 'labels' and optionally 'regions_class_order'.
                         Example:
                         {
                             "organ": {"labels": {"background": 0, "liver": 1, "spleen": 2}},
                             "lesion": {"labels": {"background": 0, "tumor": 1, "cyst": 2}}
                         }
            inference_nonlin: Optional dict mapping task names to inference nonlinearities
        """
        self.tasks_config = tasks_config
        self.task_names = list(tasks_config.keys())
        
        # Create label manager for each task
        self.task_label_managers: Dict[str, SingleTaskLabelManager] = {}
        for task_name, config in tasks_config.items():
            self.task_label_managers[task_name] = SingleTaskLabelManager(
                task_name=task_name,
                label_dict=config['labels'],
                regions_class_order=config.get('regions_class_order', None)
            )
            
            # Override inference nonlin if provided
            if inference_nonlin is not None and task_name in inference_nonlin:
                self.task_label_managers[task_name].inference_nonlin = inference_nonlin[task_name]
    
    @property
    def num_tasks(self) -> int:
        return len(self.task_names)
    
    def get_task_label_manager(self, task_name: str) -> SingleTaskLabelManager:
        """Get the label manager for a specific task."""
        return self.task_label_managers[task_name]
    
    @property
    def num_segmentation_heads_per_task(self) -> Dict[str, int]:
        """Get the number of segmentation heads needed for each task."""
        return {
            task_name: lm.num_segmentation_heads 
            for task_name, lm in self.task_label_managers.items()
        }
    
    @property
    def total_segmentation_heads(self) -> int:
        """Total number of segmentation heads across all tasks."""
        return sum(self.num_segmentation_heads_per_task.values())
    
    def has_any_regions(self) -> bool:
        """Check if any task uses region-based training."""
        return any(lm.has_regions for lm in self.task_label_managers.values())
    
    def has_any_ignore_label(self) -> bool:
        """Check if any task has an ignore label."""
        return any(lm.has_ignore_label for lm in self.task_label_managers.values())
    
    def apply_inference_nonlin(
        self, 
        logits: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply task-specific inference nonlinearity to logits.
        
        Args:
            logits: Dict mapping task names to logit tensors
        
        Returns:
            Dict mapping task names to probability tensors
        """
        result = {}
        for task_name, task_logits in logits.items():
            lm = self.task_label_managers[task_name]
            with torch.no_grad():
                task_logits = task_logits.float()
                result[task_name] = lm.inference_nonlin(task_logits)
        return result
    
    def convert_logits_to_segmentation(
        self, 
        logits: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Convert logits to segmentation maps for all tasks.
        
        Args:
            logits: Dict mapping task names to logit tensors (c, x, y(, z))
        
        Returns:
            Dict mapping task names to segmentation tensors (x, y(, z))
        """
        result = {}
        for task_name, task_logits in logits.items():
            lm = self.task_label_managers[task_name]
            
            if lm.has_regions:
                # Apply sigmoid and threshold
                probs = torch.sigmoid(task_logits.float())
                segmentation = torch.zeros(
                    probs.shape[1:], dtype=torch.int16, device=probs.device
                )
                for i, c in enumerate(lm.regions_class_order):
                    segmentation[probs[i] > 0.5] = c
            else:
                # Argmax for standard classification
                segmentation = task_logits.argmax(0)
            
            result[task_name] = segmentation
        
        return result
    
    @classmethod
    def from_dataset_json(cls, dataset_json: dict) -> 'MultiTaskLabelManager':
        """
        Create MultiTaskLabelManager from dataset.json format.
        
        Expected format:
        {
            "tasks": {
                "task_name1": {
                    "labels": {...},
                    "regions_class_order": [...]  # optional
                },
                "task_name2": {...}
            }
        }
        """
        if 'tasks' not in dataset_json:
            raise ValueError(
                "dataset.json must contain 'tasks' key for multi-task training. "
                "Format: {'tasks': {'task_name': {'labels': {...}}, ...}}"
            )
        
        return cls(tasks_config=dataset_json['tasks'])


def get_multi_task_label_manager_from_plans(
    plans: dict,
    dataset_json: dict
) -> MultiTaskLabelManager:
    """
    Create MultiTaskLabelManager from plans and dataset.json.
    
    This is the entry point for creating a multi-task label manager,
    similar to how PlansManager.get_label_manager() works for single-task.
    """
    return MultiTaskLabelManager.from_dataset_json(dataset_json)
