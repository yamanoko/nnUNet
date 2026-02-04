from abc import abstractmethod
from typing import List, Tuple, Union
import torch
from torch import nn, autocast
from dynamic_network_architectures.architectures.primus import Primus
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.lr_schedule.nnUNetTrainer_warmup import nnUNetTrainer_warmup
from torch.nn.parallel import DistributedDataParallel as DDP
from nnunetv2.training.lr_scheduler.warmup import Lin_incr_LRScheduler, PolyLRScheduler_offset
from nnunetv2.utilities.helpers import empty_cache, dummy_context

######################################################
# See this paper for information on Primus!
# Wald*, T., Roy*, S., Isensee*, F., Ulrich, C., Ziegler, S., Trofimova, D., ... & Maier-Hein, K. (2025). Primus: Enforcing attention usage for 3d medical image segmentation. arXiv preprint arXiv:2503.01835.
# * equal contribution
######################################################

class AbstractPrimus(nnUNetTrainer_warmup):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 3e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = False

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

    def configure_optimizers(self, stage: str = "warmup_all"):
        assert stage in ["warmup_all", "train"]

        if self.training_stage == stage:
            return self.optimizer, self.lr_scheduler

        if isinstance(self.network, DDP):
            params = self.network.module.parameters()
        else:
            params = self.network.parameters()

        if stage == "warmup_all":
            self.print_to_log_file("train whole net, warmup")
            optimizer = torch.optim.AdamW(
                params, self.initial_lr, weight_decay=self.weight_decay, amsgrad=False, betas=(0.9, 0.98), fused=True
            )
            lr_scheduler = Lin_incr_LRScheduler(optimizer, self.initial_lr, self.warmup_duration_whole_net)
            self.print_to_log_file(f"Initialized warmup_all optimizer and lr_scheduler at epoch {self.current_epoch}")
        else:
            self.print_to_log_file("train whole net, default schedule")
            if self.training_stage == "warmup_all":
                # we can keep the existing optimizer and don't need to create a new one. This will allow us to keep
                # the accumulated momentum terms which already point in a useful driection
                optimizer = self.optimizer
            else:
                optimizer = torch.optim.AdamW(
                    params,
                    self.initial_lr,
                    weight_decay=self.weight_decay,
                    amsgrad=False,
                    betas=(0.9, 0.98),
                    fused=True,
                )
            lr_scheduler = PolyLRScheduler_offset(
                optimizer, self.initial_lr, self.num_epochs, self.warmup_duration_whole_net
            )
            self.print_to_log_file(f"Initialized train optimizer and lr_scheduler at epoch {self.current_epoch}")
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
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            # del data
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


class nnUNet_Primus_S_Trainer(AbstractPrimus):

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        # this architecture will crash if the patch size is not divisible by 8!
        model = Primus(
            num_input_channels,
            396,
            (8, 8, 8),
            num_output_channels,
            12,
            6,
            self.configuration_manager.patch_size,
            drop_path_rate=0.2,
            scale_attn_inner=True,
            init_values=0.1,
        )
        return model


class nnUNet_Primus_B_Trainer(AbstractPrimus):

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        # this architecture will crash if the patch size is not divisible by 8!
        model = Primus(
            num_input_channels,
            792,
            (8, 8, 8),
            num_output_channels,
            12,
            12,
            self.configuration_manager.patch_size,
            drop_path_rate=0.2,
            scale_attn_inner=True,
            init_values=0.1,
        )
        return model


class nnUNet_Primus_M_Trainer(AbstractPrimus):

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        # this architecture will crash if the patch size is not divisible by 8!
        model = Primus(
            num_input_channels,
            864,
            (8, 8, 8),
            num_output_channels,
            16,
            12,
            self.configuration_manager.patch_size,
            drop_path_rate=0.2,
            scale_attn_inner=True,
            init_values=0.1,
        )
        return model


class nnUNet_Primus_M_Trainer_BS8(nnUNet_Primus_M_Trainer):

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.configuration_manager.configuration["batch_size"] = 8


class nnUNet_Primus_M_Trainer_BS8_2e4(nnUNet_Primus_M_Trainer):

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 2e-4
        self.configuration_manager.configuration["batch_size"] = 8


class nnUNet_Trainer_BS8(nnUNetTrainer):

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.configuration_manager.configuration["batch_size"] = 8


class nnUNet_Primus_L_Trainer(AbstractPrimus):

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        # this architecture will crash if the patch size is not divisible by 8!
        model = Primus(
            num_input_channels,
            1056,
            (8, 8, 8),
            num_output_channels,
            24,
            16,
            self.configuration_manager.patch_size,
            drop_path_rate=0.2,
            scale_attn_inner=True,
            init_values=0.1,
        )
        return model


class _Primus_S_96_BS1(nnUNet_Primus_S_Trainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plans["configurations"][configuration]["patch_size"] = (96, 96, 96)  # As per repository
        plans["configurations"][configuration]["batch_size"] = 1
        super().__init__(plans, configuration, fold, dataset_json, device)


class _Primus_B_96_BS1(nnUNet_Primus_B_Trainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plans["configurations"][configuration]["patch_size"] = (96, 96, 96)  # As per repository
        plans["configurations"][configuration]["batch_size"] = 1
        super().__init__(plans, configuration, fold, dataset_json, device)


class _Primus_M_96_BS1(nnUNet_Primus_M_Trainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plans["configurations"][configuration]["patch_size"] = (96, 96, 96)  # As per repository
        plans["configurations"][configuration]["batch_size"] = 1
        super().__init__(plans, configuration, fold, dataset_json, device)


class _Primus_L_48_BS1(nnUNet_Primus_L_Trainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plans["configurations"][configuration]["patch_size"] = (48, 48, 48)  # As per repository
        plans["configurations"][configuration]["batch_size"] = 1
        super().__init__(plans, configuration, fold, dataset_json, device)


######################################################
# Multi-Task Primus Trainers
# These trainers enable multi-task segmentation using Primus architecture
# with a shared EVA encoder and task-specific PatchDecode heads.
######################################################

# Primus size configurations for reference:
# S: embed_dim=396, depth=12, heads=6
# B: embed_dim=792, depth=12, heads=12
# M: embed_dim=864, depth=16, heads=12
# L: embed_dim=1056, depth=24, heads=16

PRIMUS_CONFIGS = {
    "S": {"embed_dim": 396, "eva_depth": 12, "eva_numheads": 6},
    "B": {"embed_dim": 792, "eva_depth": 12, "eva_numheads": 12},
    "M": {"embed_dim": 864, "eva_depth": 16, "eva_numheads": 12},
    "L": {"embed_dim": 1056, "eva_depth": 24, "eva_numheads": 16},
}


class AbstractPrimusMultiTask(nnUNetTrainer_warmup):
    """
    Abstract base class for multi-task Primus trainers.
    
    This trainer enables multi-task segmentation using Primus architecture:
    - Shared EVA Vision Transformer encoder
    - Task-specific PatchDecode heads for each segmentation task
    - No deep supervision (Primus does not support it)
    
    Subclasses should implement `primus_config` property to return the size configuration.
    
    Usage:
        nnUNetv2_train DATASET 3d_fullres FOLD -tr nnUNet_Primus_S_MultiTask_Trainer
    """
    
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
        task_loss_weights: dict = None,
    ):
        # Validate dataset.json format for multi-task
        if 'tasks' not in dataset_json:
            raise ValueError(
                "Multi-task training requires 'tasks' key in dataset.json. "
                "Format: {'tasks': {'task_name': {'labels': {...}}, ...}}"
            )
        
        self.task_loss_weights = task_loss_weights
        
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # Primus-specific settings
        self.initial_lr = 3e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = False  # Primus does not support deep supervision
        
        # Import multi-task label manager
        from nnunetv2.utilities.label_handling.multi_task_label_handling import (
            MultiTaskLabelManager,
            get_multi_task_label_manager_from_plans
        )
        
        # Override label_manager with multi-task version
        self.label_manager: MultiTaskLabelManager = get_multi_task_label_manager_from_plans(
            plans, dataset_json
        )
        self.task_names = self.label_manager.task_names
        self.num_tasks = self.label_manager.num_tasks
        
        self.print_to_log_file(f"Multi-task Primus training with {self.num_tasks} tasks: {self.task_names}")
        for task_name in self.task_names:
            task_lm = self.label_manager.get_task_label_manager(task_name)
            self.print_to_log_file(
                f"  {task_name}: {task_lm.num_segmentation_heads} heads, "
                f"regions={task_lm.has_regions}"
            )
    
    @property
    @abstractmethod
    def primus_config(self) -> dict:
        """
        Return Primus size configuration.
        
        Returns:
            dict with keys: embed_dim, eva_depth, eva_numheads
        """
        raise NotImplementedError()
    
    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        """Build multi-head Primus network."""
        from nnunetv2.architectures.multi_head_wrapper import wrap_primus_for_multi_task
        
        config = self.primus_config
        patch_size = self.configuration_manager.patch_size
        
        # Create base Primus with dummy output channels (will be replaced)
        base_model = Primus(
            num_input_channels,
            config["embed_dim"],
            (8, 8, 8),  # patch_embed_size
            1,  # dummy num_classes, will be replaced
            config["eva_depth"],
            config["eva_numheads"],
            patch_size,
            drop_path_rate=0.2,
            scale_attn_inner=True,
            init_values=0.1,
        )
        
        # Get task_num_classes from label_manager
        task_num_classes = self.label_manager.num_segmentation_heads_per_task
        
        # Wrap with multi-head wrapper
        model = wrap_primus_for_multi_task(
            base_network=base_model,
            task_num_classes=task_num_classes,
            patch_embed_size=(8, 8, 8),
        )
        
        return model
    
    def _build_loss(self) -> nn.Module:
        """Build multi-task loss function (without deep supervision for Primus)."""
        from nnunetv2.training.loss.multi_task_loss import MultiTaskLoss
        
        return MultiTaskLoss(
            label_manager=self.label_manager,
            task_loss_weights=self.task_loss_weights,
            batch_dice=self.configuration_manager.batch_dice,
            smooth=1e-5,
            ddp=self.is_ddp
        )
    
    def configure_optimizers(self, stage: str = "warmup_all"):
        """Configure optimizer with warmup schedule."""
        assert stage in ["warmup_all", "train"]

        if self.training_stage == stage:
            return self.optimizer, self.lr_scheduler

        if isinstance(self.network, DDP):
            params = self.network.module.parameters()
        else:
            params = self.network.parameters()

        if stage == "warmup_all":
            self.print_to_log_file("train whole net, warmup")
            optimizer = torch.optim.AdamW(
                params, self.initial_lr, weight_decay=self.weight_decay, amsgrad=False, betas=(0.9, 0.98), fused=True
            )
            lr_scheduler = Lin_incr_LRScheduler(optimizer, self.initial_lr, self.warmup_duration_whole_net)
            self.print_to_log_file(f"Initialized warmup_all optimizer and lr_scheduler at epoch {self.current_epoch}")
        else:
            self.print_to_log_file("train whole net, default schedule")
            if self.training_stage == "warmup_all":
                optimizer = self.optimizer
            else:
                optimizer = torch.optim.AdamW(
                    params,
                    self.initial_lr,
                    weight_decay=self.weight_decay,
                    amsgrad=False,
                    betas=(0.9, 0.98),
                    fused=True,
                )
            lr_scheduler = PolyLRScheduler_offset(
                optimizer, self.initial_lr, self.num_epochs, self.warmup_duration_whole_net
            )
            self.print_to_log_file(f"Initialized train optimizer and lr_scheduler at epoch {self.current_epoch}")
        
        self.training_stage = stage
        empty_cache(self.device)
        return optimizer, lr_scheduler
    
    def train_step(self, batch: dict) -> dict:
        """Perform a single multi-task training step."""
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            # Forward pass - returns Dict[str, Tensor]
            output = self.network(data)
            
            # Convert targets to per-task format
            targets_per_task = self._convert_targets_to_per_task(target)
            
            # Compute multi-task loss
            l = self.loss(output, targets_per_task)

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
    
    def _convert_targets_to_per_task(self, target: torch.Tensor) -> dict:
        """
        Convert multi-channel target to per-task format.
        
        Args:
            target: Tensor of shape (B, num_tasks, *spatial)
        
        Returns:
            Dict mapping task names to target tensors (B, 1, *spatial)
        """
        from nnunetv2.training.dataloading.multi_task_data_loader import (
            convert_multi_task_seg_to_per_task_targets
        )
        return convert_multi_task_seg_to_per_task_targets(target, self.label_manager)
    
    def set_deep_supervision_enabled(self, enabled: bool):
        """Primus does not support deep supervision - this is a no-op."""
        if enabled:
            import warnings
            warnings.warn(
                "Primus architecture does not support deep supervision. "
                "This setting will be ignored."
            )
    
    def get_dataloaders(self):
        """Get multi-task data loaders."""
        from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
        from nnunetv2.training.dataloading.multi_task_data_loader import nnUNetDataLoaderMultiTask
        from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
        from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
        from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
        
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        
        patch_size = self.configuration_manager.patch_size
        
        # No deep supervision for Primus
        deep_supervision_scales = None
        
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        
        # Get transforms
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self._get_combined_foreground_labels(),
            regions=None,
            ignore_label=None
        )
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self._get_combined_foreground_labels(),
            regions=None,
            ignore_label=None
        )
        
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
        
        # Set up augmenters
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
    
    def _get_combined_foreground_labels(self) -> list:
        """Get combined foreground labels from all tasks."""
        all_labels = set()
        for task_name in self.task_names:
            task_lm = self.label_manager.get_task_label_manager(task_name)
            all_labels.update(task_lm.foreground_labels)
        return sorted(all_labels)


class nnUNet_Primus_S_MultiTask_Trainer(AbstractPrimusMultiTask):
    """
    Multi-task Primus-S trainer.
    
    Primus-S configuration:
    - embed_dim: 396
    - eva_depth: 12
    - eva_numheads: 6
    """
    
    @property
    def primus_config(self) -> dict:
        return PRIMUS_CONFIGS["S"]


class nnUNet_Primus_B_MultiTask_Trainer(AbstractPrimusMultiTask):
    """
    Multi-task Primus-B trainer.
    
    Primus-B configuration:
    - embed_dim: 792
    - eva_depth: 12
    - eva_numheads: 12
    """
    
    @property
    def primus_config(self) -> dict:
        return PRIMUS_CONFIGS["B"]


class nnUNet_Primus_M_MultiTask_Trainer(AbstractPrimusMultiTask):
    """
    Multi-task Primus-M trainer.
    
    Primus-M configuration:
    - embed_dim: 864
    - eva_depth: 16
    - eva_numheads: 12
    """
    
    @property
    def primus_config(self) -> dict:
        return PRIMUS_CONFIGS["M"]


class nnUNet_Primus_L_MultiTask_Trainer(AbstractPrimusMultiTask):
    """
    Multi-task Primus-L trainer.
    
    Primus-L configuration:
    - embed_dim: 1056
    - eva_depth: 24
    - eva_numheads: 16
    """
    
    @property
    def primus_config(self) -> dict:
        return PRIMUS_CONFIGS["L"]