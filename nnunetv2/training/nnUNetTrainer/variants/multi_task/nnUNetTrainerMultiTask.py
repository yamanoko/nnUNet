"""
nnUNetTrainerMultiTask: Multi-Task Segmentation Trainer

This trainer enables simultaneous training of multiple segmentation tasks using a 
shared encoder-decoder architecture with task-specific segmentation heads.

Features:
- Arbitrary number of tasks (configurable via dataset.json)
- Each task can use standard or region-based training independently
- Shared encoder/decoder for efficient multi-task learning
- Compatible with finetuning to single-task models (encoder/decoder weights transfer)
- Supports deep supervision

Dataset JSON format:
{
    "channel_names": {"0": "CT"},
    "file_ending": ".nii.gz",
    "tasks": {
        "task_A": {
            "labels": {"background": 0, "class_A1": 1, "class_A2": 2}
        },
        "task_B": {
            "labels": {"background": 0, "class_B1": 1, "class_B2": 2, "class_B3": 3}
        }
    },
    "numTraining": 100
}

Label format: Multi-channel segmentation where each channel is a task's labels.
seg.shape = (num_tasks, X, Y, Z)

Usage:
    nnUNetv2_train DATASET 3d_fullres FOLD -tr nnUNetTrainerMultiTask
"""

import os
import inspect
from copy import deepcopy
from time import time
from datetime import datetime
from typing import Union, Tuple, List, Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import dummy_context, empty_cache
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.training.dataloading.utils import get_allowed_n_proc_DA
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

# Multi-task specific imports
from nnunetv2.architectures.multi_head_wrapper import (
    MultiHeadSegmentationWrapper,
    get_multi_head_network_from_plans_v2
)
from nnunetv2.utilities.label_handling.multi_task_label_handling import (
    MultiTaskLabelManager,
    get_multi_task_label_manager_from_plans
)
from nnunetv2.training.dataloading.multi_task_data_loader import (
    nnUNetDataLoaderMultiTask,
    convert_multi_task_seg_to_per_task_targets
)
from nnunetv2.training.loss.multi_task_loss import (
    MultiTaskLoss,
    MultiTaskDeepSupervisionWrapper,
    build_multi_task_loss
)


class nnUNetTrainerMultiTask(nnUNetTrainer):
    """
    Multi-Task Segmentation Trainer for nnU-Net.
    
    Extends nnUNetTrainer to support multiple segmentation tasks with a shared
    encoder-decoder and task-specific segmentation heads.
    """
    
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device('cuda'),
        task_loss_weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            plans: nnU-Net plans dictionary
            configuration: Configuration name (e.g., '3d_fullres')
            fold: Fold number for cross-validation
            dataset_json: Dataset JSON with multi-task format
            device: Device to train on
            task_loss_weights: Optional weights for each task's loss
        """
        # Validate dataset.json format
        if 'tasks' not in dataset_json:
            raise ValueError(
                "Multi-task training requires 'tasks' key in dataset.json. "
                "Format: {'tasks': {'task_name': {'labels': {...}}, ...}}"
            )
        
        self.task_loss_weights = task_loss_weights
        
        # Call parent __init__ (this will set up most things)
        # We need to override label_manager initialization
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # Override label_manager with multi-task version
        self.label_manager: MultiTaskLabelManager = get_multi_task_label_manager_from_plans(
            plans, dataset_json
        )
        self.task_names = self.label_manager.task_names
        self.num_tasks = self.label_manager.num_tasks
        
        self.print_to_log_file(f"Multi-task training with {self.num_tasks} tasks: {self.task_names}")
        for task_name in self.task_names:
            task_lm = self.label_manager.get_task_label_manager(task_name)
            self.print_to_log_file(
                f"  {task_name}: {task_lm.num_segmentation_heads} heads, "
                f"regions={task_lm.has_regions}"
            )
    
    def initialize(self):
        """Initialize the trainer, building multi-head network and loss."""
        if not self.was_initialized:
            # Set batch size and oversample
            self._set_batch_size_and_oversample()
            
            # Determine number of input channels
            from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
            self.num_input_channels = determine_num_input_channels(
                self.plans_manager, self.configuration_manager, self.dataset_json
            )
            
            # Build multi-head network
            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads_per_task,  # Dict[str, int]
                self.enable_deep_supervision
            ).to(self.device)
            
            # Compile network if applicable
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)
            
            # Configure optimizer and scheduler
            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            
            # DDP setup
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])
            
            # Build multi-task loss
            self.loss = self._build_loss()
            
            # Set dataset class
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
            
            self.was_initialized = True
        else:
            raise RuntimeError(
                "You have called self.initialize even though the trainer was already initialized."
            )
    
    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: Union[int, Dict[str, int]],
        enable_deep_supervision: bool = True
    ) -> nn.Module:
        """
        Build the multi-head network architecture.
        
        This method uses the wrapper approach to support ANY nnU-Net architecture
        (PlainConvUNet, ResidualEncoderUNet, Primus, etc.).
        
        Args:
            architecture_class_name: Base architecture class name
            arch_init_kwargs: Architecture init kwargs
            arch_init_kwargs_req_import: Keys needing import
            num_input_channels: Number of input channels
            num_output_channels: Dict mapping task names to num classes
            enable_deep_supervision: Whether to enable deep supervision
        
        Returns:
            MultiHeadSegmentationWrapper instance (supports any base architecture)
        """
        if isinstance(num_output_channels, dict):
            # Multi-task mode - use wrapper approach for any architecture
            return get_multi_head_network_from_plans_v2(
                arch_class_name=architecture_class_name,
                arch_kwargs=arch_init_kwargs,
                arch_kwargs_req_import=arch_init_kwargs_req_import,
                input_channels=num_input_channels,
                task_num_classes=num_output_channels,
                allow_init=True,
                deep_supervision=enable_deep_supervision
            )
        else:
            # Standard single-task mode (for compatibility)
            return get_network_from_plans(
                architecture_class_name,
                arch_init_kwargs,
                arch_init_kwargs_req_import,
                num_input_channels,
                num_output_channels,
                allow_init=True,
                deep_supervision=enable_deep_supervision
            )
    
    def _build_loss(self) -> nn.Module:
        """Build the multi-task loss function."""
        deep_supervision_scales = self._get_deep_supervision_scales() if self.enable_deep_supervision else None
        
        return build_multi_task_loss(
            label_manager=self.label_manager,
            task_loss_weights=self.task_loss_weights,
            batch_dice=self.configuration_manager.batch_dice,
            smooth=1e-5,
            ddp=self.is_ddp,
            enable_deep_supervision=self.enable_deep_supervision,
            deep_supervision_scales=deep_supervision_scales,
            is_ddp_with_compile=self.is_ddp and self._do_i_compile()
        )
    
    def get_dataloaders(self):
        """Get multi-task data loaders."""
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        
        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        
        # Get transforms - use multi-task compatible transforms
        tr_transforms = self._get_multi_task_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug
        )
        val_transforms = self._get_multi_task_validation_transforms(deep_supervision_scales)
        
        # Get datasets
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        
        # Create multi-task data loaders
        dl_tr = nnUNetDataLoaderMultiTask(
            dataset_tr,
            self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling
        )
        dl_val = nnUNetDataLoaderMultiTask(
            dataset_val,
            self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling
        )
        
        # Set up multi-threaded augmenters
        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr,
                transform=None,
                num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2),
                seeds=None,
                pin_memory=self.device.type == 'cuda',
                wait_time=0.002
            )
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val,
                transform=None,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4),
                seeds=None,
                pin_memory=self.device.type == 'cuda',
                wait_time=0.002
            )
        
        # Warm up
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        
        return mt_gen_train, mt_gen_val
    
    def _get_multi_task_training_transforms(
        self,
        patch_size,
        rotation_for_DA,
        deep_supervision_scales,
        mirror_axes,
        do_dummy_2d_data_aug
    ):
        """Get training transforms for multi-task learning."""
        # For now, use the standard transforms
        # Multi-channel segmentation is handled automatically by spatial transforms
        return self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self._get_combined_foreground_labels(),
            regions=None,  # We handle regions per-task in loss computation
            ignore_label=None  # We handle ignore per-task
        )
    
    def _get_multi_task_validation_transforms(self, deep_supervision_scales):
        """Get validation transforms for multi-task learning."""
        return self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self._get_combined_foreground_labels(),
            regions=None,
            ignore_label=None
        )
    
    def _get_combined_foreground_labels(self) -> List[int]:
        """Get combined foreground labels from all tasks."""
        all_labels = set()
        for task_name in self.task_names:
            task_lm = self.label_manager.get_task_label_manager(task_name)
            all_labels.update(task_lm.foreground_labels)
        return sorted(all_labels)
    
    def train_step(self, batch: dict) -> dict:
        """
        Perform a single training step.
        
        Args:
            batch: Dict with 'data' and 'target' keys
        
        Returns:
            Dict with 'loss' key
        """
        data = batch['data']
        target = batch['target']
        
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)
        
        self.optimizer.zero_grad(set_to_none=True)
        
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            # Forward pass - returns Dict[str, List[Tensor]] for deep supervision
            output = self.network(data)
            del data
            
            # Convert targets to per-task format
            targets_per_task = self._convert_targets_to_per_task(target)
            
            # Compute loss
            l = self.loss(output, targets_per_task)
        
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        
        return {'loss': l.detach().cpu().numpy()}
    
    def validation_step(self, batch: dict) -> dict:
        """
        Perform a single validation step.
        
        Args:
            batch: Dict with 'data' and 'target' keys
        
        Returns:
            Dict with loss and per-task metrics
        """
        data = batch['data']
        target = batch['target']
        
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)
        
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            
            targets_per_task = self._convert_targets_to_per_task(target)
            l = self.loss(output, targets_per_task)
        
        # Get highest resolution outputs for evaluation
        if self.enable_deep_supervision:
            output_hr = {task: out[0] for task, out in output.items()}
            target_hr = target[0] if isinstance(target, list) else target
        else:
            output_hr = output
            target_hr = target
        
        # Compute per-task metrics
        result = {'loss': l.detach().cpu().numpy()}
        
        for task_idx, task_name in enumerate(self.task_names):
            task_output = output_hr[task_name]
            task_target = target_hr[:, task_idx:task_idx+1]  # (B, 1, *spatial)
            task_lm = self.label_manager.get_task_label_manager(task_name)
            
            # Compute Dice-like metrics
            axes = [0] + list(range(2, task_output.ndim))
            
            if task_lm.has_regions:
                pred_onehot = (torch.sigmoid(task_output) > 0.5).long()
                # Convert target to one-hot for comparison
                target_onehot = self._convert_to_task_onehot(task_target, task_lm)
            else:
                output_seg = task_output.argmax(1)[:, None]
                pred_onehot = torch.zeros(task_output.shape, device=task_output.device, dtype=torch.float32)
                pred_onehot.scatter_(1, output_seg, 1)
                # Create one-hot from target
                target_onehot = torch.zeros(task_output.shape, device=task_output.device, dtype=torch.float32)
                target_onehot.scatter_(1, task_target.long(), 1)
            
            from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
            tp, fp, fn, _ = get_tp_fp_fn_tn(pred_onehot, target_onehot, axes=axes)
            
            tp_hard = tp.detach().cpu().numpy()
            fp_hard = fp.detach().cpu().numpy()
            fn_hard = fn.detach().cpu().numpy()
            
            if not task_lm.has_regions:
                tp_hard = tp_hard[1:]
                fp_hard = fp_hard[1:]
                fn_hard = fn_hard[1:]
            
            result[f'{task_name}_tp'] = tp_hard
            result[f'{task_name}_fp'] = fp_hard
            result[f'{task_name}_fn'] = fn_hard
        
        return result
    
    def _convert_targets_to_per_task(
        self,
        target: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Convert multi-channel target to per-task targets.
        
        Args:
            target: Tensor (B, num_tasks, *spatial) or list of such tensors
        
        Returns:
            Dict mapping task names to target tensors
        """
        return convert_multi_task_seg_to_per_task_targets(
            target, self.label_manager
        )
    
    def _convert_to_task_onehot(
        self,
        target: torch.Tensor,
        task_lm
    ) -> torch.Tensor:
        """Convert integer target to one-hot for region-based task."""
        regions = task_lm.all_regions
        if regions is None:
            return target
        
        b, _, *spatial = target.shape
        result = torch.zeros((b, len(regions), *spatial), dtype=torch.float32, device=target.device)
        
        for region_idx, region in enumerate(regions):
            if isinstance(region, (tuple, list)):
                mask = torch.zeros_like(target, dtype=torch.bool)
                for label in region:
                    mask |= (target == label)
            else:
                mask = (target == region)
            result[:, region_idx] = mask.float().squeeze(1)
        
        return result
    
    def on_validation_epoch_end(self, val_outputs: List[dict]):
        """Handle end of validation epoch with per-task logging."""
        outputs_collated = collate_outputs(val_outputs)
        
        # Log per-task Dice
        for task_name in self.task_names:
            tp_key = f'{task_name}_tp'
            fp_key = f'{task_name}_fp'
            fn_key = f'{task_name}_fn'
            
            if tp_key in outputs_collated:
                tp = np.sum(outputs_collated[tp_key], 0)
                fp = np.sum(outputs_collated[fp_key], 0)
                fn = np.sum(outputs_collated[fn_key], 0)
                
                if self.is_ddp:
                    world_size = dist.get_world_size()
                    
                    tps = [None for _ in range(world_size)]
                    dist.all_gather_object(tps, tp)
                    tp = np.vstack([i[None] for i in tps]).sum(0)
                    
                    fps = [None for _ in range(world_size)]
                    dist.all_gather_object(fps, fp)
                    fp = np.vstack([i[None] for i in fps]).sum(0)
                    
                    fns = [None for _ in range(world_size)]
                    dist.all_gather_object(fns, fn)
                    fn = np.vstack([i[None] for i in fns]).sum(0)
                
                dice_per_class = 2 * tp / (2 * tp + fp + fn + 1e-8)
                mean_dice = np.mean(dice_per_class)
                
                self.logger.log(f'{task_name}_dice', mean_dice, self.current_epoch)
                self.print_to_log_file(f"  {task_name} mean Dice: {mean_dice:.4f}")
        
        # Call parent for standard logging
        # We need to aggregate tp/fp/fn across all tasks for model selection
        all_tp = []
        all_fp = []
        all_fn = []
        
        for task_name in self.task_names:
            tp_key = f'{task_name}_tp'
            if tp_key in outputs_collated:
                all_tp.append(np.sum(outputs_collated[tp_key], 0))
                all_fp.append(np.sum(outputs_collated[f'{task_name}_fp'], 0))
                all_fn.append(np.sum(outputs_collated[f'{task_name}_fn'], 0))
        
        if all_tp:
            tp = np.concatenate(all_tp)
            fp = np.concatenate(all_fp)
            fn = np.concatenate(all_fn)
            
            global_dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
            mean_fg_dice = np.mean(global_dice)
            
            self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
            self.logger.log('dice_per_class_or_region', global_dice, self.current_epoch)
            
            # Log loss
            loss_here = np.mean(outputs_collated['loss'])
            if self.is_ddp:
                losses = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(losses, loss_here)
                loss_here = np.mean(losses)
            
            self.logger.log('val_losses', loss_here, self.current_epoch)
            
            # Update EMA and check for best model
            self.ema_validation(mean_fg_dice)
    
    def save_checkpoint(self, filename: str) -> None:
        """Save checkpoint with multi-task information."""
        if self.local_rank == 0:
            # Get network weights
            if isinstance(self.network, DDP):
                mod = self.network.module
            else:
                mod = self.network
            
            if hasattr(mod, '_orig_mod'):
                mod = mod._orig_mod
            
            checkpoint = {
                'network_weights': mod.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                'logging': self.logger.get_checkpoint(),
                '_best_ema': self._best_ema,
                'current_epoch': self.current_epoch + 1,
                'init_args': self.my_init_kwargs,
                'trainer_name': self.__class__.__name__,
                'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                # Multi-task specific
                'task_names': self.task_names,
                'num_tasks': self.num_tasks,
            }
            
            torch.save(checkpoint, filename)
    
    def load_checkpoint(self, filename_or_checkpoint: Union[str, dict]) -> None:
        """Load checkpoint."""
        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
        else:
            checkpoint = filename_or_checkpoint
        
        # Load network weights
        if isinstance(self.network, DDP):
            mod = self.network.module
        else:
            mod = self.network
        
        if hasattr(mod, '_orig_mod'):
            mod = mod._orig_mod
        
        mod.load_state_dict(checkpoint['network_weights'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # Load grad scaler state
        if self.grad_scaler is not None and checkpoint['grad_scaler_state'] is not None:
            self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])
        
        # Load logging state
        self.logger.load_checkpoint(checkpoint['logging'])
        
        # Load other states
        self._best_ema = checkpoint['_best_ema']
        self.current_epoch = checkpoint['current_epoch']
        
        if 'inference_allowed_mirroring_axes' in checkpoint:
            self.inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes']
