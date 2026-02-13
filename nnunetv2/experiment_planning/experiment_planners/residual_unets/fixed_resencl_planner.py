"""
FixedResEncLPlanner — A planner that produces a FIXED ResEncL architecture
regardless of dataset properties (spacing, shape, etc.).

Architecture parameters are hardcoded to match TaWald/nnUNet's ResEncL exactly:
  - 6 stages, features [32, 64, 128, 256, 320, 320]
  - strides [[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]]
  - kernel_sizes [[3,3,3]] * 6
  - n_blocks_per_stage [1, 3, 4, 6, 6, 6]
  - n_conv_per_stage_decoder [1, 1, 1, 1, 1]
  - patch_size [160, 160, 160], batch_size 2
  - 1 mm isotropic spacing, ZScoreNormalization, no transpose

Usage:
  # 1. Plan & preprocess (architecture is fixed for any dataset)
  nnUNetv2_plan_and_preprocess -d DATASET_ID -pl FixedResEncLPlanner

  # 2. Pretrain on a large dataset
  nnUNetv2_train PRETRAIN_ID 3d_fullres all -p FixedResEncLPlans

  # 3. Transfer plans to downstream dataset
  nnUNetv2_move_plans_between_datasets -s PRETRAIN_ID -t FINETUNE_ID \
      -sp FixedResEncLPlans -tp FixedResEncLPlans

  # 4. Finetune
  nnUNetv2_train FINETUNE_ID 3d_fullres FOLD -p FixedResEncLPlans \
      -pretrained_weights /path/to/checkpoint_final.pth
"""

from typing import Union, List, Tuple

import numpy as np
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

from nnunetv2.experiment_planning.experiment_planners.residual_unets.residual_encoder_unet_planners import (
    nnUNetPlannerResEncL,
)


class FixedResEncLPlanner(nnUNetPlannerResEncL):
    """
    Produces a dataset-independent, fully hardcoded ResEncL architecture that is
    identical to TaWald/nnUNet ``get_network_from_name("ResEncL")``.

    Key differences from the standard ``nnUNetPlannerResEncL``:

    * Target spacing is always 1 mm isotropic (``overwrite_target_spacing``).
    * Transpose is suppressed so axis order is always [0, 1, 2].
    * ``get_plans_for_configuration`` ignores the adaptive patch/topology logic
      and returns hardcoded architecture kwargs, patch size 160³, and batch size 2.
    * Normalization is forced to ``ZScoreNormalization`` for every channel.
    * Only ``3d_fullres`` is generated (``3d_lowres`` and ``2d`` are skipped).
    """

    def __init__(
        self,
        dataset_name_or_id: Union[str, int],
        gpu_memory_target_in_gb: float = 24,
        preprocessor_name: str = "DefaultPreprocessor",
        plans_name: str = "FixedResEncLPlans",
        overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
        suppress_transpose: bool = True,
    ):
        # Always force 1 mm isotropic & suppress transpose
        super().__init__(
            dataset_name_or_id,
            gpu_memory_target_in_gb,
            preprocessor_name,
            plans_name,
            overwrite_target_spacing=[1.0, 1.0, 1.0],
            suppress_transpose=True,
        )

    # ------------------------------------------------------------------
    # Force ZScoreNormalization for every channel
    # ------------------------------------------------------------------
    def determine_normalization_scheme_and_whether_mask_is_used_for_norm(
        self,
    ) -> Tuple[List[str], List[bool]]:
        modalities = (
            self.dataset_json["channel_names"]
            if "channel_names" in self.dataset_json
            else self.dataset_json["modality"]
        )
        num_channels = len(modalities)
        normalization_schemes = ["ZScoreNormalization"] * num_channels
        use_mask_for_norm = [True] * num_channels
        return normalization_schemes, use_mask_for_norm

    # ------------------------------------------------------------------
    # Hardcode all architecture and plan parameters
    # ------------------------------------------------------------------
    def get_plans_for_configuration(
        self,
        spacing: Union[np.ndarray, Tuple[float, ...], List[float]],
        median_shape: Union[np.ndarray, Tuple[int, ...]],
        data_identifier: str,
        approximate_n_voxels_dataset: float,
        _cache: dict,
    ) -> dict:
        # ---- resampling / softmax export (dataset-specific, inherited) ----
        resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs = (
            self.determine_resampling()
        )
        resampling_softmax, resampling_softmax_kwargs = (
            self.determine_segmentation_softmax_export_fn()
        )

        # ---- normalization (forced ZScore above) ----
        normalization_schemes, mask_is_used_for_norm = (
            self.determine_normalization_scheme_and_whether_mask_is_used_for_norm()
        )

        # ---- fixed architecture — matches TaWald/nnUNet ResEncL exactly ----
        architecture_kwargs = {
            "network_class_name": (
                f"{ResidualEncoderUNet.__module__}.{ResidualEncoderUNet.__name__}"
            ),
            "arch_kwargs": {
                "n_stages": 6,
                "features_per_stage": (32, 64, 128, 256, 320, 320),
                "conv_op": "torch.nn.modules.conv.Conv3d",
                "kernel_sizes": [
                    [3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3],
                ],
                "strides": [
                    [1, 1, 1],
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2],
                ],
                "n_blocks_per_stage": [1, 3, 4, 6, 6, 6],
                "n_conv_per_stage_decoder": [1, 1, 1, 1, 1],
                "conv_bias": True,
                "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                "norm_op_kwargs": {"eps": 1e-5, "affine": True},
                "dropout_op": None,
                "dropout_op_kwargs": None,
                "nonlin": "torch.nn.LeakyReLU",
                "nonlin_kwargs": {"inplace": True},
            },
            "_kw_requires_import": ("conv_op", "norm_op", "dropout_op", "nonlin"),
        }

        plan = {
            "data_identifier": data_identifier,
            "preprocessor_name": self.preprocessor_name,
            "batch_size": 2,
            "patch_size": [160, 160, 160],
            "median_image_size_in_voxels": median_shape,
            "spacing": spacing,
            "normalization_schemes": normalization_schemes,
            "use_mask_for_norm": mask_is_used_for_norm,
            "resampling_fn_data": resampling_data.__name__,
            "resampling_fn_seg": resampling_seg.__name__,
            "resampling_fn_data_kwargs": resampling_data_kwargs,
            "resampling_fn_seg_kwargs": resampling_seg_kwargs,
            "resampling_fn_probabilities": resampling_softmax.__name__,
            "resampling_fn_probabilities_kwargs": resampling_softmax_kwargs,
            "architecture": architecture_kwargs,
        }
        return plan

    # ------------------------------------------------------------------
    # Only produce 3d_fullres — skip 3d_lowres and 2d
    # ------------------------------------------------------------------
    def plan_experiment(self):
        """
        Calls the parent plan_experiment and then removes any configuration
        other than 3d_fullres (we only need the fixed 3D plan).
        """
        plans = super().plan_experiment()
        configs_to_keep = {"3d_fullres"}
        plans["configurations"] = {
            k: v for k, v in plans["configurations"].items() if k in configs_to_keep
        }
        return plans
