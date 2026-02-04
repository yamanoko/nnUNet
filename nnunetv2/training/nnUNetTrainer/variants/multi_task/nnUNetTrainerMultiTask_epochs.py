"""
nnUNetTrainerMultiTask with various epoch configurations.

This module provides multi-task trainers with different epoch settings
for experimentation purposes.

Usage:
    nnUNetv2_train DATASET 3d_fullres FOLD -tr nnUNetTrainerMultiTask_250epochs
    nnUNetv2_train DATASET 3d_fullres FOLD -tr nnUNet_Primus_M_MultiTask_500epochs
"""

import torch
from typing import Optional, Dict

from nnunetv2.training.nnUNetTrainer.variants.multi_task.nnUNetTrainerMultiTask import (
    nnUNetTrainerMultiTask
)
from nnunetv2.training.nnUNetTrainer.primus.primus_trainers import (
    AbstractPrimusMultiTask,
    PRIMUS_CONFIGS
)


######################################################
# Standard Multi-Task Trainers with Different Epochs
######################################################

class nnUNetTrainerMultiTask_250epochs(nnUNetTrainerMultiTask):
    """Multi-task trainer with 250 epochs."""
    
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device('cuda'),
        task_loss_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__(plans, configuration, fold, dataset_json, device, task_loss_weights)
        self.num_epochs = 250


class nnUNetTrainerMultiTask_500epochs(nnUNetTrainerMultiTask):
    """Multi-task trainer with 500 epochs."""
    
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device('cuda'),
        task_loss_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__(plans, configuration, fold, dataset_json, device, task_loss_weights)
        self.num_epochs = 500


class nnUNetTrainerMultiTask_1000epochs(nnUNetTrainerMultiTask):
    """Multi-task trainer with 1000 epochs (default)."""
    
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device('cuda'),
        task_loss_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__(plans, configuration, fold, dataset_json, device, task_loss_weights)
        self.num_epochs = 1000


class nnUNetTrainerMultiTask_1500epochs(nnUNetTrainerMultiTask):
    """Multi-task trainer with 1500 epochs."""
    
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device('cuda'),
        task_loss_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__(plans, configuration, fold, dataset_json, device, task_loss_weights)
        self.num_epochs = 1500


class nnUNetTrainerMultiTask_2000epochs(nnUNetTrainerMultiTask):
    """Multi-task trainer with 2000 epochs."""
    
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device('cuda'),
        task_loss_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__(plans, configuration, fold, dataset_json, device, task_loss_weights)
        self.num_epochs = 2000


######################################################
# Primus-S Multi-Task Trainers with Different Epochs
######################################################

class nnUNet_Primus_S_MultiTask_250epochs(AbstractPrimusMultiTask):
    """Multi-task Primus-S trainer with 250 epochs."""
    
    @property
    def primus_config(self) -> dict:
        return PRIMUS_CONFIGS["S"]
    
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device('cuda'),
        task_loss_weights: dict = None,
    ):
        super().__init__(plans, configuration, fold, dataset_json, device, task_loss_weights)
        self.num_epochs = 250


class nnUNet_Primus_S_MultiTask_500epochs(AbstractPrimusMultiTask):
    """Multi-task Primus-S trainer with 500 epochs."""
    
    @property
    def primus_config(self) -> dict:
        return PRIMUS_CONFIGS["S"]
    
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device('cuda'),
        task_loss_weights: dict = None,
    ):
        super().__init__(plans, configuration, fold, dataset_json, device, task_loss_weights)
        self.num_epochs = 500


class nnUNet_Primus_S_MultiTask_1500epochs(AbstractPrimusMultiTask):
    """Multi-task Primus-S trainer with 1500 epochs."""
    
    @property
    def primus_config(self) -> dict:
        return PRIMUS_CONFIGS["S"]
    
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device('cuda'),
        task_loss_weights: dict = None,
    ):
        super().__init__(plans, configuration, fold, dataset_json, device, task_loss_weights)
        self.num_epochs = 1500


class nnUNet_Primus_S_MultiTask_2000epochs(AbstractPrimusMultiTask):
    """Multi-task Primus-S trainer with 2000 epochs."""
    
    @property
    def primus_config(self) -> dict:
        return PRIMUS_CONFIGS["S"]
    
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device('cuda'),
        task_loss_weights: dict = None,
    ):
        super().__init__(plans, configuration, fold, dataset_json, device, task_loss_weights)
        self.num_epochs = 2000


######################################################
# Primus-B Multi-Task Trainers with Different Epochs
######################################################

class nnUNet_Primus_B_MultiTask_250epochs(AbstractPrimusMultiTask):
    """Multi-task Primus-B trainer with 250 epochs."""
    
    @property
    def primus_config(self) -> dict:
        return PRIMUS_CONFIGS["B"]
    
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device('cuda'),
        task_loss_weights: dict = None,
    ):
        super().__init__(plans, configuration, fold, dataset_json, device, task_loss_weights)
        self.num_epochs = 250


class nnUNet_Primus_B_MultiTask_500epochs(AbstractPrimusMultiTask):
    """Multi-task Primus-B trainer with 500 epochs."""
    
    @property
    def primus_config(self) -> dict:
        return PRIMUS_CONFIGS["B"]
    
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device('cuda'),
        task_loss_weights: dict = None,
    ):
        super().__init__(plans, configuration, fold, dataset_json, device, task_loss_weights)
        self.num_epochs = 500


class nnUNet_Primus_B_MultiTask_1500epochs(AbstractPrimusMultiTask):
    """Multi-task Primus-B trainer with 1500 epochs."""
    
    @property
    def primus_config(self) -> dict:
        return PRIMUS_CONFIGS["B"]
    
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device('cuda'),
        task_loss_weights: dict = None,
    ):
        super().__init__(plans, configuration, fold, dataset_json, device, task_loss_weights)
        self.num_epochs = 1500


class nnUNet_Primus_B_MultiTask_2000epochs(AbstractPrimusMultiTask):
    """Multi-task Primus-B trainer with 2000 epochs."""
    
    @property
    def primus_config(self) -> dict:
        return PRIMUS_CONFIGS["B"]
    
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device('cuda'),
        task_loss_weights: dict = None,
    ):
        super().__init__(plans, configuration, fold, dataset_json, device, task_loss_weights)
        self.num_epochs = 2000


######################################################
# Primus-M Multi-Task Trainers with Different Epochs
######################################################

class nnUNet_Primus_M_MultiTask_250epochs(AbstractPrimusMultiTask):
    """Multi-task Primus-M trainer with 250 epochs."""
    
    @property
    def primus_config(self) -> dict:
        return PRIMUS_CONFIGS["M"]
    
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device('cuda'),
        task_loss_weights: dict = None,
    ):
        super().__init__(plans, configuration, fold, dataset_json, device, task_loss_weights)
        self.num_epochs = 250


class nnUNet_Primus_M_MultiTask_500epochs(AbstractPrimusMultiTask):
    """Multi-task Primus-M trainer with 500 epochs."""
    
    @property
    def primus_config(self) -> dict:
        return PRIMUS_CONFIGS["M"]
    
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device('cuda'),
        task_loss_weights: dict = None,
    ):
        super().__init__(plans, configuration, fold, dataset_json, device, task_loss_weights)
        self.num_epochs = 500


class nnUNet_Primus_M_MultiTask_1500epochs(AbstractPrimusMultiTask):
    """Multi-task Primus-M trainer with 1500 epochs."""
    
    @property
    def primus_config(self) -> dict:
        return PRIMUS_CONFIGS["M"]
    
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device('cuda'),
        task_loss_weights: dict = None,
    ):
        super().__init__(plans, configuration, fold, dataset_json, device, task_loss_weights)
        self.num_epochs = 1500


class nnUNet_Primus_M_MultiTask_2000epochs(AbstractPrimusMultiTask):
    """Multi-task Primus-M trainer with 2000 epochs."""
    
    @property
    def primus_config(self) -> dict:
        return PRIMUS_CONFIGS["M"]
    
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device('cuda'),
        task_loss_weights: dict = None,
    ):
        super().__init__(plans, configuration, fold, dataset_json, device, task_loss_weights)
        self.num_epochs = 2000


######################################################
# Primus-L Multi-Task Trainers with Different Epochs
######################################################

class nnUNet_Primus_L_MultiTask_250epochs(AbstractPrimusMultiTask):
    """Multi-task Primus-L trainer with 250 epochs."""
    
    @property
    def primus_config(self) -> dict:
        return PRIMUS_CONFIGS["L"]
    
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device('cuda'),
        task_loss_weights: dict = None,
    ):
        super().__init__(plans, configuration, fold, dataset_json, device, task_loss_weights)
        self.num_epochs = 250


class nnUNet_Primus_L_MultiTask_500epochs(AbstractPrimusMultiTask):
    """Multi-task Primus-L trainer with 500 epochs."""
    
    @property
    def primus_config(self) -> dict:
        return PRIMUS_CONFIGS["L"]
    
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device('cuda'),
        task_loss_weights: dict = None,
    ):
        super().__init__(plans, configuration, fold, dataset_json, device, task_loss_weights)
        self.num_epochs = 500


class nnUNet_Primus_L_MultiTask_1500epochs(AbstractPrimusMultiTask):
    """Multi-task Primus-L trainer with 1500 epochs."""
    
    @property
    def primus_config(self) -> dict:
        return PRIMUS_CONFIGS["L"]
    
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device('cuda'),
        task_loss_weights: dict = None,
    ):
        super().__init__(plans, configuration, fold, dataset_json, device, task_loss_weights)
        self.num_epochs = 1500


class nnUNet_Primus_L_MultiTask_2000epochs(AbstractPrimusMultiTask):
    """Multi-task Primus-L trainer with 2000 epochs."""
    
    @property
    def primus_config(self) -> dict:
        return PRIMUS_CONFIGS["L"]
    
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device('cuda'),
        task_loss_weights: dict = None,
    ):
        super().__init__(plans, configuration, fold, dataset_json, device, task_loss_weights)
        self.num_epochs = 2000
