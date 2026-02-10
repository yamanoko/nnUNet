"""
Primus-M Warmup Fine-tuning Learning Rate Schedulers

This module implements warmup schedulers specifically designed for Primus-M fine-tuning,
based on the paper findings where the Warm-Up schedule was identified as the best
fine-tuning strategy for Primus-M (transformer) architectures.

Learning Rate Schedule:
    - Stage 1 (Warm-Up): Linear increase from 0 to max_lr over N epochs
    - Stage 2 (Default): Polynomial decay (PolyLR) for the remaining epochs

Configuration (Primus-M fine-tuning):
    - Maximum learning rate: 3e-5
    - Warm-Up duration: 15 epochs (150 epoch training) or 50 epochs (1000 epoch training)
    - Optimizer: AdamW with weight_decay=5e-2, betas=(0.9, 0.98)

Usage:
    nnUNetv2_train DATASET CONFIG FOLD -tr nnUNet_Primus_M_WarmupFinetuning -pretrained_weights PATH_TO_CHECKPOINT
    nnUNetv2_train DATASET CONFIG FOLD -tr nnUNet_Primus_M_WarmupFinetuning_150ep -pretrained_weights PATH_TO_CHECKPOINT

Reference:
    Wald*, T., Roy*, S., Isensee*, F., et al. (2025). "Primus: Enforcing attention 
    usage for 3d medical image segmentation." arXiv preprint arXiv:2503.01835.
    * equal contribution
"""

from abc import abstractmethod
from typing import List, Tuple, Union

import numpy as np
import torch
from torch import nn, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from dynamic_network_architectures.architectures.primus import Primus
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.lr_scheduler.warmup import Lin_incr_LRScheduler, PolyLRScheduler_offset
from nnunetv2.utilities.helpers import empty_cache, dummy_context


class AbstractPrimusWarmupFinetuning(nnUNetTrainer):
    """
    Abstract base class for Primus warmup fine-tuning trainers.
    
    Implements the two-stage learning rate schedule:
    1. Warm-Up: Linear increase to max learning rate (3e-5)
    2. Training: Polynomial decay (PolyLR) for the remaining epochs
    
    This schedule is specifically designed for fine-tuning Primus models,
    where the Warm-Up strategy was found to be optimal.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Primus-M fine-tuning hyperparameters
        self.initial_lr = 3e-5  # Maximum learning rate for fine-tuning
        self.weight_decay = 5e-2
        self.warmup_duration_whole_net = 50  # Warm-up epochs for 1000 epoch training
        self.num_epochs = 1000
        self.enable_deep_supervision = False
        self.training_stage = None  # 'warmup_all' or 'train'

    @abstractmethod
    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        raise NotImplementedError()

    def get_stage(self) -> str:
        """Determine current training stage based on epoch."""
        if self.current_epoch < self.warmup_duration_whole_net:
            return 'warmup_all'
        else:
            return 'train'

    def on_train_epoch_start(self):
        """Handle learning rate schedule transitions at epoch boundaries."""
        if self.current_epoch == 0:
            self.optimizer, self.lr_scheduler = self.configure_optimizers('warmup_all')
        elif self.current_epoch == self.warmup_duration_whole_net:
            self.optimizer, self.lr_scheduler = self.configure_optimizers('train')

        self.network.train()
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}"
        )
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

    def configure_optimizers(self, stage: str = 'warmup_all'):
        """
        Configure optimizer and learning rate scheduler for the specified stage.
        
        Args:
            stage: Training stage ('warmup_all' or 'train')
            
        Returns:
            tuple: (optimizer, lr_scheduler)
        """
        assert stage in ['warmup_all', 'train'], f"Invalid stage: {stage}"

        if self.training_stage == stage:
            return self.optimizer, self.lr_scheduler

        if isinstance(self.network, DDP):
            params = self.network.module.parameters()
        else:
            params = self.network.parameters()

        if stage == 'warmup_all':
            self.print_to_log_file("Warm-Up stage: training whole network with linear warmup")
            self.print_to_log_file(f"  Max learning rate: {self.initial_lr}")
            self.print_to_log_file(f"  Warm-up duration: {self.warmup_duration_whole_net} epochs")
            optimizer = torch.optim.AdamW(
                params,
                self.initial_lr,
                weight_decay=self.weight_decay,
                amsgrad=False,
                betas=(0.9, 0.98),
                fused=True
            )
            lr_scheduler = Lin_incr_LRScheduler(
                optimizer, self.initial_lr, self.warmup_duration_whole_net
            )
            self.print_to_log_file(
                f"Initialized warmup_all optimizer and lr_scheduler at epoch {self.current_epoch}"
            )
        else:  # stage == 'train'
            self.print_to_log_file("Training stage: polynomial learning rate decay (Default schedule)")
            if self.training_stage == 'warmup_all':
                # Keep existing optimizer to maintain momentum terms
                optimizer = self.optimizer
            else:
                optimizer = torch.optim.AdamW(
                    params,
                    self.initial_lr,
                    weight_decay=self.weight_decay,
                    amsgrad=False,
                    betas=(0.9, 0.98),
                    fused=True
                )
            lr_scheduler = PolyLRScheduler_offset(
                optimizer, self.initial_lr, self.num_epochs, self.warmup_duration_whole_net
            )
            self.print_to_log_file(
                f"Initialized train optimizer and lr_scheduler at epoch {self.current_epoch}"
            )

        self.training_stage = stage
        empty_cache(self.device)
        return optimizer, lr_scheduler

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast handling for different device types
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}

    def set_deep_supervision_enabled(self, enabled: bool):
        pass


######################################################
# Primus-M Warmup Fine-tuning Trainers
######################################################

class nnUNet_Primus_M_WarmupFinetuning(AbstractPrimusWarmupFinetuning):
    """
    Primus-M Trainer with Warmup Fine-tuning Schedule (1000 epochs).
    
    Learning Rate Schedule:
        - Stage 1: Linear warm-up for 50 epochs (0 -> 3e-5)
        - Stage 2: Polynomial decay for 950 epochs (3e-5 -> 0)
    
    Usage:
        nnUNetv2_train DATASET CONFIG FOLD -tr nnUNet_Primus_M_WarmupFinetuning \\
            -pretrained_weights PATH_TO_CHECKPOINT
    """

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        # Primus-M architecture configuration
        # Note: Patch size must be divisible by 8
        model = Primus(
            num_input_channels,
            864,  # Primus-M hidden dimension
            (8, 8, 8),
            num_output_channels,
            16,  # Primus-M depth
            12,  # Primus-M num_heads
            self.configuration_manager.patch_size,
            drop_path_rate=0.2,
            scale_attn_inner=True,
            init_values=0.1,
        )
        return model


class nnUNet_Primus_M_WarmupFinetuning_150ep(nnUNet_Primus_M_WarmupFinetuning):
    """
    Primus-M Trainer with Warmup Fine-tuning Schedule (150 epochs).
    
    Learning Rate Schedule:
        - Stage 1: Linear warm-up for 15 epochs (0 -> 3e-5)
        - Stage 2: Polynomial decay for 135 epochs (3e-5 -> 0)
    
    Usage:
        nnUNetv2_train DATASET CONFIG FOLD -tr nnUNet_Primus_M_WarmupFinetuning_150ep \\
            -pretrained_weights PATH_TO_CHECKPOINT
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Scaled parameters for 150 epoch training
        self.warmup_duration_whole_net = 15  # Warm-up epochs for 150 epoch training
        self.num_epochs = 150


######################################################
# Primus-S Warmup Fine-tuning Trainers
######################################################

class nnUNet_Primus_S_WarmupFinetuning(AbstractPrimusWarmupFinetuning):
    """
    Primus-S Trainer with Warmup Fine-tuning Schedule (1000 epochs).
    """

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        model = Primus(
            num_input_channels,
            396,  # Primus-S hidden dimension
            (8, 8, 8),
            num_output_channels,
            12,  # Primus-S depth
            6,   # Primus-S num_heads
            self.configuration_manager.patch_size,
            drop_path_rate=0.2,
            scale_attn_inner=True,
            init_values=0.1,
        )
        return model


class nnUNet_Primus_S_WarmupFinetuning_150ep(nnUNet_Primus_S_WarmupFinetuning):
    """
    Primus-S Trainer with Warmup Fine-tuning Schedule (150 epochs).
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.warmup_duration_whole_net = 15
        self.num_epochs = 150


######################################################
# Primus-B Warmup Fine-tuning Trainers
######################################################

class nnUNet_Primus_B_WarmupFinetuning(AbstractPrimusWarmupFinetuning):
    """
    Primus-B Trainer with Warmup Fine-tuning Schedule (1000 epochs).
    """

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        model = Primus(
            num_input_channels,
            792,  # Primus-B hidden dimension
            (8, 8, 8),
            num_output_channels,
            12,  # Primus-B depth
            12,  # Primus-B num_heads
            self.configuration_manager.patch_size,
            drop_path_rate=0.2,
            scale_attn_inner=True,
            init_values=0.1,
        )
        return model


class nnUNet_Primus_B_WarmupFinetuning_150ep(nnUNet_Primus_B_WarmupFinetuning):
    """
    Primus-B Trainer with Warmup Fine-tuning Schedule (150 epochs).
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.warmup_duration_whole_net = 15
        self.num_epochs = 150


######################################################
# Primus-L Warmup Fine-tuning Trainers
######################################################

class nnUNet_Primus_L_WarmupFinetuning(AbstractPrimusWarmupFinetuning):
    """
    Primus-L Trainer with Warmup Fine-tuning Schedule (1000 epochs).
    """

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        model = Primus(
            num_input_channels,
            1056,  # Primus-L hidden dimension
            (8, 8, 8),
            num_output_channels,
            24,  # Primus-L depth
            16,  # Primus-L num_heads
            self.configuration_manager.patch_size,
            drop_path_rate=0.2,
            scale_attn_inner=True,
            init_values=0.1,
        )
        return model


class nnUNet_Primus_L_WarmupFinetuning_150ep(nnUNet_Primus_L_WarmupFinetuning):
    """
    Primus-L Trainer with Warmup Fine-tuning Schedule (150 epochs).
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.warmup_duration_whole_net = 15
        self.num_epochs = 150


######################################################
# Additional Epoch Variants for Primus-M Warmup Fine-tuning
######################################################

class nnUNet_Primus_M_WarmupFinetuning_250ep(nnUNet_Primus_M_WarmupFinetuning):
    """Primus-M Warmup Fine-tuning with 250 epochs."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Proportionally scaled warmup: 50 * (250/1000) = 12.5 -> 15
        self.warmup_duration_whole_net = 15
        self.num_epochs = 250


class nnUNet_Primus_M_WarmupFinetuning_500ep(nnUNet_Primus_M_WarmupFinetuning):
    """Primus-M Warmup Fine-tuning with 500 epochs."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Proportionally scaled warmup: 50 * (500/1000) = 25
        self.warmup_duration_whole_net = 25
        self.num_epochs = 500


class nnUNet_Primus_M_WarmupFinetuning_2000ep(nnUNet_Primus_M_WarmupFinetuning):
    """Primus-M Warmup Fine-tuning with 2000 epochs (extended training)."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Extended warmup for longer training
        self.warmup_duration_whole_net = 100
        self.num_epochs = 2000
