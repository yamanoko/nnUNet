# Architectures module for nnU-Net custom network architectures

# Multi-head wrapper for multi-task segmentation (supports ANY nnU-Net architecture including Primus)
from nnunetv2.architectures.multi_head_wrapper import (
    MultiHeadSegmentationWrapper,
    wrap_network_for_multi_task,
    get_multi_head_network_from_plans_v2
)

__all__ = [
    'MultiHeadSegmentationWrapper',
    'wrap_network_for_multi_task',
    'get_multi_head_network_from_plans_v2',
]
