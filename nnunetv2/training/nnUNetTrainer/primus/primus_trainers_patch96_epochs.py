from typing import List, Tuple, Union
import torch
from torch import nn
from nnunetv2.training.nnUNetTrainer.primus.primus_trainers import (
    nnUNet_Primus_S_Trainer,
    nnUNet_Primus_B_Trainer,
    nnUNet_Primus_M_Trainer,
)

######################################################
# Primus trainers with patch_size fixed to 96x96x96 and epochs set to 1000 or 150
######################################################


# Primus S with patch_size 96x96x96 and 1000 epochs
class Primus_S_96_1000epochs(nnUNet_Primus_S_Trainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plans["configurations"][configuration]["patch_size"] = (96, 96, 96)
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000


# Primus S with patch_size 96x96x96 and 150 epochs
class Primus_S_96_150epochs(nnUNet_Primus_S_Trainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plans["configurations"][configuration]["patch_size"] = (96, 96, 96)
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 150


# Primus B with patch_size 96x96x96 and 1000 epochs
class Primus_B_96_1000epochs(nnUNet_Primus_B_Trainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plans["configurations"][configuration]["patch_size"] = (96, 96, 96)
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000


# Primus B with patch_size 96x96x96 and 150 epochs
class Primus_B_96_150epochs(nnUNet_Primus_B_Trainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plans["configurations"][configuration]["patch_size"] = (96, 96, 96)
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 150


# Primus M with patch_size 96x96x96 and 1000 epochs
class Primus_M_96_1000epochs(nnUNet_Primus_M_Trainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plans["configurations"][configuration]["patch_size"] = (96, 96, 96)
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000


# Primus M with patch_size 96x96x96 and 150 epochs
class Primus_M_96_150epochs(nnUNet_Primus_M_Trainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plans["configurations"][configuration]["patch_size"] = (96, 96, 96)
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 150
