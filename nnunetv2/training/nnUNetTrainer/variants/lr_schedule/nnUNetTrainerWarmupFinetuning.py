"""
nnUNet Trainers with Advanced Learning Rate Schedules for Supervised Pretraining Fine-tuning

These trainers extend the standard nnUNetTrainer to support advanced learning rate scheduling strategies
for fine-tuning models pretrained with supervised learning. They work with standard nnU-Net pretrained
weights loaded via the -pretrained_weights flag.

Based on the paper: "An OpenMind for 3D medical vision self-supervised learning"
https://arxiv.org/html/2412.17041v2

Usage:
    nnUNetv2_train DATASET CONFIG FOLD -tr nnUNetTrainerWarmupFinetuning -pretrained_weights PATH_TO_CHECKPOINT

Available Trainers:
    ResEnc (CNN) Trainers - Use SGD optimizer, initial_lr=1e-3:
        - nnUNetTrainerWarmupFinetuning: Warm-Up schedule (1000 epochs)
        - nnUNetTrainerWarmupFinetuning_150ep: Warm-Up schedule (150 epochs)
        - nnUNetTrainerValleyFinetuning: Valley schedule (1000 epochs)
        - nnUNetTrainerValleyFinetuning_150ep: Valley schedule (150 epochs)
        - nnUNetTrainerSawtoothFinetuning: Sawtooth schedule (1000 epochs)
        - nnUNetTrainerSawtoothFinetuning_150ep: Sawtooth schedule (150 epochs)
    
    Primus (Transformer) Trainers - Use AdamW optimizer, initial_lr=1e-4:
        - nnUNetTrainerWarmupFinetuning_Primus: Warm-Up schedule (1000 epochs)
        - nnUNetTrainerWarmupFinetuning_Primus_150ep: Warm-Up schedule (150 epochs)
        - nnUNetTrainerValleyFinetuning_Primus: Valley schedule (1000 epochs)
        - nnUNetTrainerValleyFinetuning_Primus_150ep: Valley schedule (150 epochs)
        - nnUNetTrainerSawtoothFinetuning_Primus: Sawtooth schedule (1000 epochs)
        - nnUNetTrainerSawtoothFinetuning_Primus_150ep: Sawtooth schedule (150 epochs)

Scheduling Strategies (from the paper):
    Warm-Up:
        - Stage 1: Linear warmup of entire network (N epochs)
        - Stage 2: Polynomial decay of entire network (remaining epochs)
    
    Valley:
        - Stage 1: Train decoder only with linear decreasing lr (N epochs)
        - Stage 2: Linear warmup of entire network (M epochs)
        - Stage 3: Polynomial decay of entire network (remaining epochs)
    
    Sawtooth:
        - Stage 1: Warmup decoder only with linear increasing lr (N/2 epochs)
        - Stage 2: Train decoder only with polynomial decay (N/2 epochs)
        - Stage 3: Warmup entire network with linear increasing lr (M epochs)
        - Stage 4: Polynomial decay of entire network (remaining epochs)

Architecture-Specific Notes:
    ResEnc (CNN): Uses network.decoder.parameters() for decoder-only training
    Primus (Transformer): Uses network.up_projection.parameters() for decoder-only training
"""

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.lr_scheduler.warmup import (
    Lin_incr_LRScheduler,
    PolyLRScheduler_offset,
    Lin_incr_offset_LRScheduler
)
from nnunetv2.utilities.helpers import empty_cache


class nnUNetTrainerWarmupFinetuning(nnUNetTrainer):
    """
    nnUNet Trainer with warmup learning rate schedule for fine-tuning.
    
    This trainer implements a two-stage learning rate schedule:
    1. Warmup: Linear increase to max learning rate
    2. Training: Standard polynomial decay
    
    Designed for fine-tuning models pretrained with supervised learning using
    standard nnU-Net pretraining framework.
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
        # Fine-tuning hyperparameters (aligned with sawtooth_trainer.py)
        self.initial_lr = 1e-3
        self.warmup_duration_whole_net = 50  # Linear warmup duration
        self.num_epochs = 1000
        self.training_stage = None

    def get_stage(self):
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
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
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
            self.print_to_log_file("Warmup stage: training whole network with linear warmup")
            optimizer = torch.optim.SGD(
                params, self.initial_lr, 
                weight_decay=self.weight_decay, 
                momentum=0.99, 
                nesterov=True
            )
            lr_scheduler = Lin_incr_LRScheduler(optimizer, self.initial_lr, self.warmup_duration_whole_net)
        else:  # stage == 'train'
            self.print_to_log_file("Training stage: polynomial learning rate decay")
            if self.training_stage == 'warmup_all':
                # Keep existing optimizer to maintain momentum
                optimizer = self.optimizer
            else:
                optimizer = torch.optim.SGD(
                    params, self.initial_lr,
                    weight_decay=self.weight_decay,
                    momentum=0.99, 
                    nesterov=True
                )
            lr_scheduler = PolyLRScheduler_offset(
                optimizer, self.initial_lr, self.num_epochs, self.warmup_duration_whole_net
            )

        self.training_stage = stage
        empty_cache(self.device)
        return optimizer, lr_scheduler


class nnUNetTrainerWarmupFinetuning_150ep(nnUNetTrainerWarmupFinetuning):
    """
    Warmup fine-tuning trainer with 150 epochs instead of 1000.
    
    Suitable for smaller datasets or faster experimentation.
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
        # Proportionally scaled (aligned with sawtooth_trainer_150ep.py)
        self.warmup_duration_whole_net = 15
        self.num_epochs = 150


class nnUNetTrainerValleyFinetuning(nnUNetTrainer):
    """
    nnUNet Trainer with valley learning rate schedule for fine-tuning.
    
    This trainer implements a three-stage learning rate schedule (Valley):
    1. Train decoder: Linear decrease for decoder only
    2. Warmup all: Linear warmup for entire network
    3. Train: Standard polynomial decay for entire network
    
    This schedule prioritizes decoder adaptation by first training it with 
    a linear decreasing lr, followed by a linear warm-up phase for the full 
    network, and then the default schedule.
    
    Designed for fine-tuning models pretrained with supervised learning using
    standard nnU-Net pretraining framework.
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
        # Fine-tuning hyperparameters (aligned with sawtooth_trainer.py)
        self.initial_lr = 1e-3
        self.valley_decoder_duration = 50  # Linear decay for decoder
        self.warmup_duration_whole_net = 50  # Linear warmup for entire network
        self.num_epochs = 1000
        self.training_stage = None

    def get_stage(self):
        """Determine current training stage based on epoch."""
        if self.current_epoch < self.valley_decoder_duration:
            return 'train_decoder'
        elif self.current_epoch < self.valley_decoder_duration + \
                self.warmup_duration_whole_net:
            return 'warmup_all'
        else:
            return 'train'

    def on_train_epoch_start(self):
        """Handle learning rate schedule transitions at epoch boundaries."""
        if self.current_epoch == 0:
            self.optimizer, self.lr_scheduler = \
                self.configure_optimizers('train_decoder')
        elif self.current_epoch == self.valley_decoder_duration:
            self.optimizer, self.lr_scheduler = \
                self.configure_optimizers('warmup_all')
        elif self.current_epoch == self.valley_decoder_duration + \
                self.warmup_duration_whole_net:
            self.optimizer, self.lr_scheduler = \
                self.configure_optimizers('train')

        self.network.train()
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: "
            f"{np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}"
        )
        # lrs are the same for all workers
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'],
                       self.current_epoch)

    def configure_optimizers(self, stage: str = 'train_decoder'):
        """
        Configure optimizer and learning rate scheduler.
        
        Args:
            stage: Training stage ('train_decoder', 'warmup_all', or 'train')
        
        Returns:
            tuple: (optimizer, lr_scheduler)
        """
        assert stage in ['train_decoder', 'warmup_all', 'train'], \
            f"Invalid stage: {stage}"

        if self.training_stage == stage:
            return self.optimizer, self.lr_scheduler

        if isinstance(self.network, DDP):
            params = self.network.module.parameters()
            # Get decoder parameters for ResEnc and Primus architectures
            heads = self.network.module.decoder.parameters()
        else:
            params = self.network.parameters()
            heads = self.network.decoder.parameters()

        if stage == 'train_decoder':
            self.print_to_log_file(
                "Stage 1 (Valley): Train decoder with linear decreasing lr"
            )
            optimizer = torch.optim.SGD(
                heads, self.initial_lr,
                weight_decay=self.weight_decay,
                momentum=0.99, 
                nesterov=True
            )
            # Linear decrease: start from initial_lr and go to 0
            lr_scheduler = PolyLRScheduler_offset(
                optimizer, self.initial_lr, self.valley_decoder_duration, 0
            )
        elif stage == 'warmup_all':
            self.print_to_log_file(
                "Stage 2 (Valley): Warmup entire network with linear warmup"
            )
            optimizer = torch.optim.SGD(
                params, self.initial_lr,
                weight_decay=self.weight_decay,
                momentum=0.99, 
                nesterov=True
            )
            lr_scheduler = Lin_incr_offset_LRScheduler(
                optimizer, self.initial_lr,
                self.valley_decoder_duration + self.warmup_duration_whole_net,
                self.valley_decoder_duration
            )
        else:  # stage == 'train'
            self.print_to_log_file(
                "Stage 3 (Valley): Train whole network with polynomial decay"
            )
            if self.training_stage == 'warmup_all':
                # Keep existing optimizer to maintain momentum
                optimizer = self.optimizer
            else:
                optimizer = torch.optim.SGD(
                    params, self.initial_lr,
                    weight_decay=self.weight_decay,
                    momentum=0.99, 
                    nesterov=True
                )
            lr_scheduler = PolyLRScheduler_offset(
                optimizer, self.initial_lr, self.num_epochs,
                self.valley_decoder_duration + self.warmup_duration_whole_net
            )

        self.training_stage = stage
        empty_cache(self.device)
        return optimizer, lr_scheduler


class nnUNetTrainerValleyFinetuning_150ep(nnUNetTrainerValleyFinetuning):
    """
    Valley fine-tuning trainer with 150 epochs instead of 1000.
    
    Suitable for smaller datasets or faster experimentation.
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
        # Proportionally scaled (aligned with sawtooth_trainer_150ep.py)
        self.valley_decoder_duration = 15
        self.warmup_duration_whole_net = 15
        self.num_epochs = 150


class nnUNetTrainerSawtoothFinetuning(nnUNetTrainer):
    """
    nnUNet Trainer with sawtooth learning rate schedule for fine-tuning.
    
    This trainer implements a four-stage learning rate schedule:
    1. Warmup Decoder: Linear warmup of segmentation head only
    2. Train Decoder: Polynomial decay of segmentation head only
    3. Warmup All: Linear warmup of entire network
    4. Train: Polynomial decay of entire network
    
    The sawtooth pattern (decoder warmup -> decoder train -> all warmup -> all train)
    allows the segmentation head to adapt to the new task before fine-tuning the
    entire network, which can lead to better performance.
    
    Designed for fine-tuning models pretrained with supervised learning using
    standard nnU-Net pretraining framework.
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
        # Fine-tuning hyperparameters (aligned with sawtooth_trainer.py)
        self.warmup_duration_decoder = 50  # Decoder warmup+training duration
        self.initial_lr = 1e-3
        self.warmup_duration_whole_net = 50  # Full network warmup duration
        self.num_epochs = 1000
        self.training_stage = None

    def get_stage(self):
        """Determine current training stage based on epoch."""
        if self.current_epoch < self.warmup_duration_decoder // 2:
            return 'warmup_decoder'
        elif self.current_epoch < self.warmup_duration_decoder:
            return 'train_decoder'
        elif self.current_epoch < self.warmup_duration_decoder + self.warmup_duration_whole_net:
            return 'warmup_all'
        else:
            return 'train'

    def on_train_epoch_start(self):
        """Handle learning rate schedule transitions at epoch boundaries."""
        if self.current_epoch == 0:
            self.optimizer, self.lr_scheduler = self.configure_optimizers('warmup_decoder')
        elif self.current_epoch == self.warmup_duration_decoder // 2:
            self.optimizer, self.lr_scheduler = self.configure_optimizers('train_decoder')
        elif self.current_epoch == self.warmup_duration_decoder:
            self.optimizer, self.lr_scheduler = self.configure_optimizers('warmup_all')
        elif self.current_epoch == self.warmup_duration_decoder + self.warmup_duration_whole_net:
            self.optimizer, self.lr_scheduler = self.configure_optimizers('train')

        self.network.train()
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}"
        )
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

    def configure_optimizers(self, stage: str = 'warmup_all'):
        """
        Configure optimizer and learning rate scheduler for the specified stage.
        
        Args:
            stage: Training stage ('warmup_decoder', 'train_decoder', 'warmup_all', or 'train')
        
        Returns:
            tuple: (optimizer, lr_scheduler)
        """
        assert stage in ['warmup_all', 'train', 'warmup_decoder', 'train_decoder'], \
            f"Invalid stage: {stage}"

        if self.training_stage == stage:
            return self.optimizer, self.lr_scheduler

        if isinstance(self.network, DDP):
            params = self.network.module.parameters()
            # Get decoder parameters for ResEnc and Primus architectures
            heads = self.network.module.decoder.parameters()
        else:
            params = self.network.parameters()
            heads = self.network.decoder.parameters()

        if stage == 'warmup_decoder':
            self.print_to_log_file("Stage 1: Warmup decoder with linear warmup")
            optimizer = torch.optim.SGD(
                heads, self.initial_lr,
                weight_decay=self.weight_decay,
                momentum=0.99, 
                nesterov=True
            )
            lr_scheduler = Lin_incr_LRScheduler(
                optimizer, self.initial_lr, self.warmup_duration_decoder // 2
            )
        elif stage == 'train_decoder':
            self.print_to_log_file("Stage 2: Train decoder with polynomial decay")
            if self.training_stage == 'warmup_decoder':
                # Keep existing optimizer to maintain momentum
                optimizer = self.optimizer
            else:
                optimizer = torch.optim.SGD(
                    heads, self.initial_lr,
                    weight_decay=self.weight_decay,
                    momentum=0.99, 
                    nesterov=True
                )
            lr_scheduler = PolyLRScheduler_offset(
                optimizer, self.initial_lr, self.warmup_duration_decoder,
                self.warmup_duration_decoder // 2
            )
        elif stage == 'warmup_all':
            self.print_to_log_file("Stage 3: Warmup whole network with linear warmup")
            optimizer = torch.optim.SGD(
                params, self.initial_lr,
                weight_decay=self.weight_decay,
                momentum=0.99, 
                nesterov=True
            )
            lr_scheduler = Lin_incr_offset_LRScheduler(
                optimizer, self.initial_lr,
                self.warmup_duration_decoder + self.warmup_duration_whole_net,
                self.warmup_duration_decoder
            )
        else:  # stage == 'train'
            self.print_to_log_file("Stage 4: Train whole network with polynomial decay")
            if self.training_stage == 'warmup_all':
                # Keep existing optimizer to maintain momentum
                optimizer = self.optimizer
            else:
                optimizer = torch.optim.SGD(
                    params, self.initial_lr,
                    weight_decay=self.weight_decay,
                    momentum=0.99, 
                    nesterov=True
                )
            lr_scheduler = PolyLRScheduler_offset(
                optimizer, self.initial_lr, self.num_epochs,
                self.warmup_duration_decoder + self.warmup_duration_whole_net
            )

        self.training_stage = stage
        empty_cache(self.device)
        return optimizer, lr_scheduler


class nnUNetTrainerSawtoothFinetuning_150ep(nnUNetTrainerSawtoothFinetuning):
    """
    Sawtooth fine-tuning trainer with 150 epochs instead of 1000.
    
    Suitable for smaller datasets or faster experimentation.
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
        # Proportionally scaled (aligned with sawtooth_trainer_150ep.py)
        self.warmup_duration_decoder = 15
        self.warmup_duration_whole_net = 15
        self.num_epochs = 150


# Primus (Transformer) specific trainers with different hyperparameters
class nnUNetTrainerWarmupFinetuning_Primus(nnUNetTrainer):
    """
    nnUNet Trainer with warmup learning rate schedule for Primus (Transformer) architectures.
    
    Uses AdamW optimizer and lower learning rate compared to ResEnc trainers.
    Aligned with PretrainedTrainer_Primus_sawtooth hyperparameters.
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
        # Primus-specific hyperparameters (aligned with PretrainedTrainer_Primus_sawtooth)
        self.initial_lr = 1e-4
        self.weight_decay = 5e-2
        self.warmup_duration_whole_net = 50
        self.num_epochs = 1000
        self.training_stage = None

    def get_stage(self):
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
        """Configure AdamW optimizer for Primus architecture."""
        assert stage in ['warmup_all', 'train'], f"Invalid stage: {stage}"

        if self.training_stage == stage:
            return self.optimizer, self.lr_scheduler

        if isinstance(self.network, DDP):
            params = self.network.module.parameters()
        else:
            params = self.network.parameters()

        if stage == 'warmup_all':
            self.print_to_log_file("Warmup stage: training whole network with linear warmup")
            optimizer = torch.optim.AdamW(
                params, self.initial_lr,
                weight_decay=self.weight_decay,
                amsgrad=False, betas=(0.9, 0.98), fused=True
            )
            lr_scheduler = Lin_incr_LRScheduler(optimizer, self.initial_lr, self.warmup_duration_whole_net)
        else:  # stage == 'train'
            self.print_to_log_file("Training stage: polynomial learning rate decay")
            if self.training_stage == 'warmup_all':
                optimizer = self.optimizer
            else:
                optimizer = torch.optim.AdamW(
                    params, self.initial_lr,
                    weight_decay=self.weight_decay,
                    amsgrad=False, betas=(0.9, 0.98), fused=True
                )
            lr_scheduler = PolyLRScheduler_offset(
                optimizer, self.initial_lr, self.num_epochs, self.warmup_duration_whole_net
            )

        self.training_stage = stage
        empty_cache(self.device)
        return optimizer, lr_scheduler


class nnUNetTrainerWarmupFinetuning_Primus_150ep(nnUNetTrainerWarmupFinetuning_Primus):
    """
    Warmup fine-tuning trainer for Primus with 150 epochs instead of 1000.
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
        # Note: warmup_duration_whole_net stays at 50 (aligned with PretrainedTrainer_Primus_sawtooth_150ep)
        self.warmup_duration_whole_net = 15
        self.num_epochs = 150


class nnUNetTrainerValleyFinetuning_Primus(nnUNetTrainer):
    """
    Valley schedule trainer for Primus (Transformer) architectures.
    
    Uses AdamW optimizer and lower learning rate.
    Aligned with PretrainedTrainer_Primus_sawtooth hyperparameters.
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
        # Primus-specific hyperparameters
        self.initial_lr = 1e-4
        self.weight_decay = 5e-2
        self.valley_decoder_duration = 50
        self.warmup_duration_whole_net = 50
        self.num_epochs = 1000
        self.training_stage = None

    def get_stage(self):
        """Determine current training stage based on epoch."""
        if self.current_epoch < self.valley_decoder_duration:
            return 'train_decoder'
        elif self.current_epoch < self.valley_decoder_duration + self.warmup_duration_whole_net:
            return 'warmup_all'
        else:
            return 'train'

    def on_train_epoch_start(self):
        """Handle learning rate schedule transitions at epoch boundaries."""
        if self.current_epoch == 0:
            self.optimizer, self.lr_scheduler = self.configure_optimizers('train_decoder')
        elif self.current_epoch == self.valley_decoder_duration:
            self.optimizer, self.lr_scheduler = self.configure_optimizers('warmup_all')
        elif self.current_epoch == self.valley_decoder_duration + self.warmup_duration_whole_net:
            self.optimizer, self.lr_scheduler = self.configure_optimizers('train')

        self.network.train()
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: "
            f"{np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}"
        )
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

    def configure_optimizers(self, stage: str = 'train_decoder'):
        """Configure AdamW optimizer for Primus architecture."""
        assert stage in ['train_decoder', 'warmup_all', 'train'], f"Invalid stage: {stage}"

        if self.training_stage == stage:
            return self.optimizer, self.lr_scheduler

        if isinstance(self.network, DDP):
            params = self.network.module.parameters()
            # For Primus, use up_projection instead of decoder
            heads = self.network.module.up_projection.parameters()
        else:
            params = self.network.parameters()
            heads = self.network.up_projection.parameters()

        if stage == 'train_decoder':
            self.print_to_log_file("Stage 1 (Valley): Train decoder with linear decreasing lr")
            optimizer = torch.optim.AdamW(
                heads, self.initial_lr,
                weight_decay=self.weight_decay,
                amsgrad=False, betas=(0.9, 0.98), fused=True
            )
            lr_scheduler = PolyLRScheduler_offset(
                optimizer, self.initial_lr, self.valley_decoder_duration, 0
            )
        elif stage == 'warmup_all':
            self.print_to_log_file("Stage 2 (Valley): Warmup entire network with linear warmup")
            optimizer = torch.optim.AdamW(
                params, self.initial_lr,
                weight_decay=self.weight_decay,
                amsgrad=False, betas=(0.9, 0.98), fused=True
            )
            lr_scheduler = Lin_incr_offset_LRScheduler(
                optimizer, self.initial_lr,
                self.valley_decoder_duration + self.warmup_duration_whole_net,
                self.valley_decoder_duration
            )
        else:  # stage == 'train'
            self.print_to_log_file("Stage 3 (Valley): Train whole network with polynomial decay")
            if self.training_stage == 'warmup_all':
                optimizer = self.optimizer
            else:
                optimizer = torch.optim.AdamW(
                    params, self.initial_lr,
                    weight_decay=self.weight_decay,
                    amsgrad=False, betas=(0.9, 0.98), fused=True
                )
            lr_scheduler = PolyLRScheduler_offset(
                optimizer, self.initial_lr, self.num_epochs,
                self.valley_decoder_duration + self.warmup_duration_whole_net
            )

        self.training_stage = stage
        empty_cache(self.device)
        return optimizer, lr_scheduler


class nnUNetTrainerValleyFinetuning_Primus_150ep(nnUNetTrainerValleyFinetuning_Primus):
    """
    Valley fine-tuning trainer for Primus with 150 epochs.
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
        self.valley_decoder_duration = 15
        self.warmup_duration_whole_net = 50  # Stays at 50 for Primus
        self.num_epochs = 150


class nnUNetTrainerSawtoothFinetuning_Primus(nnUNetTrainer):
    """
    Sawtooth schedule trainer for Primus (Transformer) architectures.
    
    Uses AdamW optimizer, lower learning rate, and warmup_lr_factor.
    Aligned with PretrainedTrainer_Primus_sawtooth hyperparameters.
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
        # Primus-specific hyperparameters (aligned with PretrainedTrainer_Primus_sawtooth)
        self.initial_lr = 1e-4
        self.warmup_lr_factor = 0.01  # Decoder warmup uses lower lr
        self.weight_decay = 5e-2
        self.warmup_duration_decoder = 50
        self.warmup_duration_whole_net = 50
        self.num_epochs = 1000
        self.training_stage = None

    def get_stage(self):
        """Determine current training stage based on epoch."""
        if self.current_epoch < self.warmup_duration_decoder // 2:
            return 'warmup_decoder'
        elif self.current_epoch < self.warmup_duration_decoder:
            return 'train_decoder'
        elif self.current_epoch < self.warmup_duration_decoder + self.warmup_duration_whole_net:
            return 'warmup_all'
        else:
            return 'train'

    def on_train_epoch_start(self):
        """Handle learning rate schedule transitions at epoch boundaries."""
        if self.current_epoch == 0:
            self.optimizer, self.lr_scheduler = self.configure_optimizers('warmup_decoder')
        elif self.current_epoch == self.warmup_duration_decoder // 2:
            self.optimizer, self.lr_scheduler = self.configure_optimizers('train_decoder')
        elif self.current_epoch == self.warmup_duration_decoder:
            self.optimizer, self.lr_scheduler = self.configure_optimizers('warmup_all')
        elif self.current_epoch == self.warmup_duration_decoder + self.warmup_duration_whole_net:
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
        """Configure AdamW optimizer for Primus architecture."""
        assert stage in ['warmup_all', 'train', 'warmup_decoder', 'train_decoder'], \
            f"Invalid stage: {stage}"

        if self.training_stage == stage:
            return self.optimizer, self.lr_scheduler

        if isinstance(self.network, DDP):
            params = self.network.module.parameters()
            # For Primus, use up_projection instead of decoder
            heads = self.network.module.up_projection.parameters()
        else:
            params = self.network.parameters()
            heads = self.network.up_projection.parameters()

        if stage == 'warmup_decoder':
            self.print_to_log_file("Stage 1: Warmup decoder with linear warmup")
            optimizer = torch.optim.AdamW(
                heads, self.initial_lr,
                weight_decay=self.weight_decay,
                amsgrad=False, betas=(0.9, 0.98), fused=True
            )
            lr_scheduler = Lin_incr_LRScheduler(
                optimizer, self.initial_lr * self.warmup_lr_factor,
                self.warmup_duration_decoder // 2
            )
            self.print_to_log_file(
                f"Initialized warmup only decoder optimizer and lr_scheduler at epoch {self.current_epoch}"
            )
        elif stage == 'train_decoder':
            self.print_to_log_file("Stage 2: Train decoder with polynomial decay")
            if self.training_stage == 'warmup_decoder':
                optimizer = self.optimizer
            else:
                optimizer = torch.optim.AdamW(
                    heads, self.initial_lr,
                    weight_decay=self.weight_decay,
                    amsgrad=False, betas=(0.9, 0.98), fused=True
                )
            lr_scheduler = PolyLRScheduler_offset(
                optimizer, self.initial_lr * self.warmup_lr_factor,
                self.warmup_duration_decoder,
                self.warmup_duration_decoder // 2
            )
            self.print_to_log_file(
                f"Initialized train only decoder optimizer and lr_scheduler at epoch {self.current_epoch}"
            )
        elif stage == 'warmup_all':
            self.print_to_log_file("Stage 3: Warmup whole network with linear warmup")
            optimizer = torch.optim.AdamW(
                params, self.initial_lr,
                weight_decay=self.weight_decay,
                amsgrad=False, betas=(0.9, 0.98), fused=True
            )
            lr_scheduler = Lin_incr_offset_LRScheduler(
                optimizer, self.initial_lr,
                self.warmup_duration_decoder + self.warmup_duration_whole_net,
                self.warmup_duration_decoder
            )
            self.print_to_log_file(
                f"Initialized warmup_all optimizer and lr_scheduler at epoch {self.current_epoch}"
            )
        else:  # stage == 'train'
            self.print_to_log_file("Stage 4: Train whole network with polynomial decay")
            if self.training_stage == 'warmup_all':
                optimizer = self.optimizer
            else:
                optimizer = torch.optim.AdamW(
                    params, self.initial_lr,
                    weight_decay=self.weight_decay,
                    amsgrad=False, betas=(0.9, 0.98), fused=True
                )
            lr_scheduler = PolyLRScheduler_offset(
                optimizer, self.initial_lr, self.num_epochs,
                self.warmup_duration_decoder + self.warmup_duration_whole_net
            )
            self.print_to_log_file(
                f"Initialized train optimizer and lr_scheduler at epoch {self.current_epoch}"
            )

        self.training_stage = stage
        empty_cache(self.device)
        return optimizer, lr_scheduler


class nnUNetTrainerSawtoothFinetuning_Primus_150ep(nnUNetTrainerSawtoothFinetuning_Primus):
    """
    Sawtooth fine-tuning trainer for Primus with 150 epochs.
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
        # Aligned with PretrainedTrainer_Primus_sawtooth_150ep
        self.warmup_lr_factor = 0.1  # Different from 1000ep version
        self.warmup_duration_decoder = 15
        self.warmup_duration_whole_net = 50  # Stays at 50 for Primus
        self.num_epochs = 150
