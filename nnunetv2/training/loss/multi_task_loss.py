"""
Multi-Task Loss Functions for nnU-Net

This module provides loss functions for multi-task segmentation, where multiple
segmentation heads produce outputs for different tasks simultaneously.

The multi-task loss combines individual task losses with configurable weighting.
Each task can use either standard DC+CE loss (for classification) or DC+BCE loss
(for region-based training).
"""

from typing import Dict, List, Optional, Union
import torch
from torch import nn
import numpy as np

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.label_handling.multi_task_label_handling import MultiTaskLabelManager


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss that combines losses from multiple segmentation tasks.
    
    Each task has its own loss function (DC+CE or DC+BCE depending on whether
    region-based training is used), and the final loss is a weighted combination.
    """
    
    def __init__(
        self,
        label_manager: MultiTaskLabelManager,
        task_loss_weights: Optional[Dict[str, float]] = None,
        batch_dice: bool = True,
        smooth: float = 1e-5,
        ddp: bool = False,
        weight_ce: float = 1.0,
        weight_dice: float = 1.0
    ):
        """
        Args:
            label_manager: MultiTaskLabelManager containing task configurations
            task_loss_weights: Optional dict mapping task names to loss weights.
                              If None, all tasks are weighted equally (1.0).
            batch_dice: Whether to compute batch dice (True) or sample dice (False)
            smooth: Smoothing factor for dice loss
            ddp: Whether using distributed data parallel
            weight_ce: Weight for cross-entropy/BCE loss component
            weight_dice: Weight for dice loss component
        """
        super().__init__()
        
        self.label_manager = label_manager
        self.task_names = label_manager.task_names
        self.num_tasks = label_manager.num_tasks
        
        # Set task weights
        if task_loss_weights is None:
            self.task_loss_weights = {name: 1.0 for name in self.task_names}
        else:
            self.task_loss_weights = task_loss_weights
            # Fill in missing weights with 1.0
            for name in self.task_names:
                if name not in self.task_loss_weights:
                    self.task_loss_weights[name] = 1.0
        
        # Normalize weights so they sum to 1
        total_weight = sum(self.task_loss_weights.values())
        self.task_loss_weights = {
            k: v / total_weight for k, v in self.task_loss_weights.items()
        }
        
        # Create loss function for each task
        self.task_losses = nn.ModuleDict()
        for task_name in self.task_names:
            task_lm = label_manager.get_task_label_manager(task_name)
            
            if task_lm.has_regions:
                # Region-based training: use BCE + Dice
                loss = DC_and_BCE_loss(
                    bce_kwargs={},
                    soft_dice_kwargs={
                        'batch_dice': batch_dice,
                        'do_bg': True,
                        'smooth': smooth,
                        'ddp': ddp
                    },
                    weight_ce=weight_ce,
                    weight_dice=weight_dice,
                    use_ignore_label=task_lm.has_ignore_label,
                    dice_class=MemoryEfficientSoftDiceLoss
                )
            else:
                # Standard training: use CE + Dice
                loss = DC_and_CE_loss(
                    soft_dice_kwargs={
                        'batch_dice': batch_dice,
                        'smooth': smooth,
                        'do_bg': False,
                        'ddp': ddp
                    },
                    ce_kwargs={},
                    weight_ce=weight_ce,
                    weight_dice=weight_dice,
                    ignore_label=task_lm.ignore_label,
                    dice_class=MemoryEfficientSoftDiceLoss
                )
            
            self.task_losses[task_name] = loss
    
    def forward(
        self,
        net_outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute multi-task loss.
        
        Args:
            net_outputs: Dict mapping task names to network outputs (B, C, *spatial)
            targets: Dict mapping task names to target tensors
                    - For standard training: (B, 1, *spatial) integer labels
                    - For region-based: (B, num_regions, *spatial) one-hot
        
        Returns:
            Combined weighted loss
        """
        total_loss = 0.0
        
        for task_name in self.task_names:
            task_output = net_outputs[task_name]
            task_target = targets[task_name]
            
            task_loss = self.task_losses[task_name](task_output, task_target)
            total_loss = total_loss + self.task_loss_weights[task_name] * task_loss
        
        return total_loss


class MultiTaskDeepSupervisionWrapper(nn.Module):
    """
    Deep supervision wrapper for multi-task loss.
    
    Applies multi-task loss at multiple resolution levels with decreasing weights.
    """
    
    def __init__(
        self,
        loss: MultiTaskLoss,
        weight_factors: List[float]
    ):
        """
        Args:
            loss: MultiTaskLoss instance
            weight_factors: Weights for each resolution level (highest res first)
        """
        super().__init__()
        
        assert any([x != 0 for x in weight_factors]), \
            "At least one weight factor should be != 0.0"
        
        self.weight_factors = tuple(weight_factors)
        self.loss = loss
        self.task_names = loss.task_names
    
    def forward(
        self,
        net_outputs: Dict[str, List[torch.Tensor]],
        targets: Dict[str, List[torch.Tensor]]
    ) -> torch.Tensor:
        """
        Compute deep supervision multi-task loss.
        
        Args:
            net_outputs: Dict mapping task names to lists of outputs at each scale
            targets: Dict mapping task names to lists of targets at each scale
        
        Returns:
            Combined weighted loss across all scales and tasks
        """
        total_loss = 0.0
        
        for scale_idx, weight in enumerate(self.weight_factors):
            if weight == 0.0:
                continue
            
            # Gather outputs and targets at this scale
            scale_outputs = {
                task_name: outputs[scale_idx] 
                for task_name, outputs in net_outputs.items()
            }
            scale_targets = {
                task_name: targets_list[scale_idx]
                for task_name, targets_list in targets.items()
            }
            
            # Compute loss at this scale
            scale_loss = self.loss(scale_outputs, scale_targets)
            total_loss = total_loss + weight * scale_loss
        
        return total_loss


def build_multi_task_loss(
    label_manager: MultiTaskLabelManager,
    task_loss_weights: Optional[Dict[str, float]] = None,
    batch_dice: bool = True,
    smooth: float = 1e-5,
    ddp: bool = False,
    enable_deep_supervision: bool = True,
    deep_supervision_scales: Optional[List[List[float]]] = None,
    is_ddp_with_compile: bool = False
) -> nn.Module:
    """
    Build a multi-task loss function with optional deep supervision.
    
    Args:
        label_manager: MultiTaskLabelManager
        task_loss_weights: Optional weights for each task's loss
        batch_dice: Whether to use batch dice
        smooth: Smoothing factor for dice
        ddp: Whether using DDP
        enable_deep_supervision: Whether to enable deep supervision
        deep_supervision_scales: Scales for deep supervision
        is_ddp_with_compile: Whether using DDP with torch.compile
    
    Returns:
        MultiTaskLoss or MultiTaskDeepSupervisionWrapper
    """
    # Create base multi-task loss
    loss = MultiTaskLoss(
        label_manager=label_manager,
        task_loss_weights=task_loss_weights,
        batch_dice=batch_dice,
        smooth=smooth,
        ddp=ddp
    )
    
    # Wrap with deep supervision if needed
    if enable_deep_supervision and deep_supervision_scales is not None:
        # Compute weights for each scale
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        
        if ddp and not is_ddp_with_compile:
            # Avoid DDP issues with zero weights
            weights[-1] = 1e-6
        else:
            weights[-1] = 0
        
        # Normalize weights
        weights = weights / weights.sum()
        
        loss = MultiTaskDeepSupervisionWrapper(loss, weights.tolist())
    
    return loss
