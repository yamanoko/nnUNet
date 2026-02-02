# Architectures module for nnU-Net custom network architectures

# Legacy multi-head implementation (for reference, supports PlainConvUNet and ResidualEncoderUNet only)
from nnunetv2.architectures.multi_head_unet import MultiHeadUNet, get_multi_head_network_from_plans

# Recommended: Wrapper approach (supports ANY nnU-Net architecture including Primus)
from nnunetv2.architectures.multi_head_wrapper import (
    MultiHeadSegmentationWrapper,
    wrap_network_for_multi_task,
    get_multi_head_network_from_plans_v2
)

__all__ = [
    # Legacy
    'MultiHeadUNet',
    'get_multi_head_network_from_plans',
    # Wrapper (recommended)
    'MultiHeadSegmentationWrapper',
    'wrap_network_for_multi_task',
    'get_multi_head_network_from_plans_v2',
]
