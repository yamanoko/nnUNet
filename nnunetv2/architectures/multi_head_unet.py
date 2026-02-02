"""
Multi-Head U-Net Architecture for Multi-Task Segmentation

This module provides a wrapper around standard U-Net architectures (PlainConvUNet, ResidualEncoderUNet)
that adds multiple segmentation heads for simultaneous multi-task learning.

Each task gets its own independent segmentation head, but shares the encoder and decoder features.
This enables learning multiple segmentation tasks (e.g., organ segmentation + lesion segmentation)
from the same input with shared representations.

The pretrained encoder/decoder weights can be used for fine-tuning on single-task segmentation
by using load_pretrained_weights, which automatically skips all segmentation heads.
"""

from typing import Union, List, Tuple, Type, Dict
import torch
from torch import nn
import numpy as np
import pydoc

from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp


class MultiHeadSegmentationLayers(nn.Module):
    """
    Multiple segmentation heads, one per task.
    
    Each head produces segmentation outputs at multiple resolution levels (for deep supervision).
    """
    def __init__(
        self,
        task_num_classes: Dict[str, int],
        features_per_stage: List[int],
        conv_op: Type[nn.Module],
        n_conv_per_stage_decoder: List[int],
        deep_supervision: bool = True
    ):
        """
        Args:
            task_num_classes: Dict mapping task names to number of classes (including background).
                              e.g., {"organ": 5, "lesion": 3}
            features_per_stage: Number of features at each decoder stage
            conv_op: Convolution operation (Conv2d or Conv3d)
            n_conv_per_stage_decoder: Number of conv blocks per decoder stage (determines number of DS outputs)
            deep_supervision: Whether to produce outputs at multiple resolutions
        """
        super().__init__()
        
        self.task_names = list(task_num_classes.keys())
        self.task_num_classes = task_num_classes
        self.deep_supervision = deep_supervision
        self.n_stages = len(n_conv_per_stage_decoder)
        
        # Create segmentation layers for each task
        self.task_heads = nn.ModuleDict()
        for task_name, num_classes in task_num_classes.items():
            # Each task has seg_layers at each resolution level (for deep supervision)
            seg_layers = nn.ModuleList()
            for stage_idx in range(self.n_stages):
                # 1x1(x1) convolution to produce class probabilities
                seg_layers.append(
                    conv_op(features_per_stage[stage_idx], num_classes, kernel_size=1, stride=1, padding=0, bias=True)
                )
            self.task_heads[task_name] = seg_layers
    
    def forward(
        self, 
        decoder_outputs: List[torch.Tensor]
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Apply segmentation heads to decoder outputs.
        
        Args:
            decoder_outputs: List of decoder feature maps at each resolution level
                            (highest resolution first)
        
        Returns:
            Dict mapping task names to:
                - If deep_supervision: List of segmentation outputs at each resolution
                - If not: Single segmentation output at highest resolution
        """
        result = {}
        
        for task_name in self.task_names:
            seg_layers = self.task_heads[task_name]
            
            if self.deep_supervision:
                task_outputs = []
                for stage_idx, decoder_output in enumerate(decoder_outputs):
                    seg_output = seg_layers[stage_idx](decoder_output)
                    task_outputs.append(seg_output)
                result[task_name] = task_outputs
            else:
                # Only use highest resolution output
                result[task_name] = seg_layers[0](decoder_outputs[0])
        
        return result


class MultiHeadUNet(nn.Module):
    """
    Multi-Head U-Net for Multi-Task Segmentation.
    
    Wraps a standard U-Net architecture and replaces the single segmentation head
    with multiple task-specific heads. Each head produces independent segmentation outputs.
    
    The encoder and decoder are shared across all tasks, enabling efficient multi-task learning.
    
    Attributes:
        encoder: Shared encoder module
        decoder: Shared decoder module (without original seg_layers)
        multi_head_seg_layers: Task-specific segmentation heads
    """
    
    def __init__(
        self,
        input_channels: int,
        task_num_classes: Dict[str, int],
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[nn.Module],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage_decoder: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Type[nn.Module] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Type[nn.Module] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Type[nn.Module] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        nonlin_first: bool = False,
        base_architecture: str = "PlainConvUNet"
    ):
        """
        Args:
            input_channels: Number of input channels
            task_num_classes: Dict mapping task names to number of output classes
                              e.g., {"organ": 5, "lesion": 3}
            n_stages: Number of encoder/decoder stages
            features_per_stage: Number of features at each stage
            conv_op: Convolution operation class (Conv2d or Conv3d)
            kernel_sizes: Kernel sizes for each stage
            strides: Strides for each stage
            n_conv_per_stage: Number of conv blocks per encoder stage
            n_conv_per_stage_decoder: Number of conv blocks per decoder stage
            conv_bias: Whether to use bias in convolutions
            norm_op: Normalization operation class
            norm_op_kwargs: Kwargs for norm_op
            dropout_op: Dropout operation class
            dropout_op_kwargs: Kwargs for dropout_op
            nonlin: Non-linearity class
            nonlin_kwargs: Kwargs for non-linearity
            deep_supervision: Whether to output at multiple resolutions
            nonlin_first: Whether nonlin comes before norm in conv blocks
            base_architecture: Base architecture type ("PlainConvUNet" or "ResidualEncoderUNet")
        """
        super().__init__()
        
        self.task_names = list(task_num_classes.keys())
        self.task_num_classes = task_num_classes
        self.deep_supervision = deep_supervision
        
        # Ensure features_per_stage is a list
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        
        # Create base network with a dummy single class (we'll replace seg_layers)
        # We use num_classes=1 just to instantiate the base network
        if base_architecture == "PlainConvUNet":
            base_net = PlainConvUNet(
                input_channels=input_channels,
                n_stages=n_stages,
                features_per_stage=features_per_stage,
                conv_op=conv_op,
                kernel_sizes=kernel_sizes,
                strides=strides,
                n_conv_per_stage=n_conv_per_stage,
                num_classes=1,  # Dummy, will be replaced
                n_conv_per_stage_decoder=n_conv_per_stage_decoder,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                dropout_op=dropout_op,
                dropout_op_kwargs=dropout_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                deep_supervision=deep_supervision,
                nonlin_first=nonlin_first
            )
        elif base_architecture == "ResidualEncoderUNet":
            base_net = ResidualEncoderUNet(
                input_channels=input_channels,
                n_stages=n_stages,
                features_per_stage=features_per_stage,
                conv_op=conv_op,
                kernel_sizes=kernel_sizes,
                strides=strides,
                n_blocks_per_stage=n_conv_per_stage,
                num_classes=1,  # Dummy, will be replaced
                n_conv_per_stage_decoder=n_conv_per_stage_decoder,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                dropout_op=dropout_op,
                dropout_op_kwargs=dropout_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                deep_supervision=deep_supervision,
                nonlin_first=nonlin_first
            )
        else:
            raise ValueError(f"Unknown base_architecture: {base_architecture}")
        
        # Extract encoder and decoder
        self.encoder = base_net.encoder
        self.decoder = base_net.decoder
        
        # Remove original seg_layers from decoder (we'll use our multi-head version)
        # Note: We keep the decoder.seg_layers as None to maintain compatibility
        # but won't use them - we use multi_head_seg_layers instead
        
        # Get decoder feature channels for seg_layers
        # In the decoder, features go from deep to shallow
        # features_per_stage for decoder: [stage_n-1_features, ..., stage_0_features]
        decoder_features = list(features_per_stage[:-1])  # All except deepest (bottleneck)
        
        # Create multi-head segmentation layers
        self.multi_head_seg_layers = MultiHeadSegmentationLayers(
            task_num_classes=task_num_classes,
            features_per_stage=decoder_features,
            conv_op=conv_op,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision
        )
        
        # Store some attributes for compatibility
        self.conv_op = conv_op
        self.n_stages = n_stages
        self.features_per_stage = features_per_stage
    
    def forward(self, x: torch.Tensor) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass through the multi-head network.
        
        Args:
            x: Input tensor of shape (B, C, *spatial_dims)
        
        Returns:
            Dict mapping task names to segmentation outputs.
            If deep_supervision is True, each value is a list of tensors at different resolutions.
            Otherwise, each value is a single tensor.
        """
        # Encoder forward
        encoder_outputs = self.encoder(x)
        
        # Decoder forward (get intermediate features, not segmentation)
        # We need to modify this to get decoder features without applying seg_layers
        decoder_outputs = self._decoder_forward_features(encoder_outputs)
        
        # Apply multi-head segmentation layers
        return self.multi_head_seg_layers(decoder_outputs)
    
    def _decoder_forward_features(self, encoder_outputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Run decoder and collect feature maps at each resolution level.
        
        This is similar to the decoder's forward but returns intermediate features
        instead of segmentation outputs.
        """
        # The encoder outputs are [stage_0, stage_1, ..., stage_n] from shallow to deep
        # We need to reverse and skip the deepest (bottleneck) for skip connections
        skips = encoder_outputs[:-1][::-1]  # Reverse order, exclude bottleneck
        
        # Start from bottleneck
        x = encoder_outputs[-1]
        
        decoder_outputs = []
        
        for stage_idx, (skip, stage, transpconv) in enumerate(
            zip(skips, self.decoder.stages, self.decoder.transpconvs)
        ):
            # Upsample
            x = transpconv(x)
            # Concatenate with skip connection
            x = torch.cat([x, skip], dim=1)
            # Apply decoder stage convolutions
            x = stage(x)
            # Collect for segmentation heads
            decoder_outputs.append(x)
        
        return decoder_outputs
    
    @staticmethod
    def initialize(module):
        """Weight initialization function."""
        if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, a=1e-2, nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


def get_multi_head_network_from_plans(
    arch_class_name: str,
    arch_kwargs: dict,
    arch_kwargs_req_import: List[str],
    input_channels: int,
    task_num_classes: Dict[str, int],
    allow_init: bool = True,
    deep_supervision: bool = True
) -> MultiHeadUNet:
    """
    Create a MultiHeadUNet from nnU-Net plans.
    
    This function mirrors get_network_from_plans but creates a multi-head network.
    
    Args:
        arch_class_name: Class name of base architecture (PlainConvUNet or ResidualEncoderUNet)
        arch_kwargs: Architecture keyword arguments from plans
        arch_kwargs_req_import: Keys in arch_kwargs that need to be imported (e.g., conv_op)
        input_channels: Number of input channels
        task_num_classes: Dict mapping task names to number of classes
        allow_init: Whether to apply weight initialization
        deep_supervision: Whether to enable deep supervision
    
    Returns:
        MultiHeadUNet instance
    """
    # Import required classes
    architecture_kwargs = dict(**arch_kwargs)
    for ri in arch_kwargs_req_import:
        if architecture_kwargs[ri] is not None:
            architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])
    
    # Determine base architecture type
    if "ResidualEncoder" in arch_class_name or "ResEnc" in arch_class_name:
        base_architecture = "ResidualEncoderUNet"
        # ResidualEncoderUNet uses n_blocks_per_stage instead of n_conv_per_stage
        if 'n_blocks_per_stage' in architecture_kwargs:
            architecture_kwargs['n_conv_per_stage'] = architecture_kwargs.pop('n_blocks_per_stage')
    else:
        base_architecture = "PlainConvUNet"
    
    # Create multi-head network
    network = MultiHeadUNet(
        input_channels=input_channels,
        task_num_classes=task_num_classes,
        deep_supervision=deep_supervision,
        base_architecture=base_architecture,
        **architecture_kwargs
    )
    
    if allow_init:
        network.apply(MultiHeadUNet.initialize)
    
    return network
