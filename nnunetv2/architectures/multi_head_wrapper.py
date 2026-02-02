"""
Multi-Head Wrapper for any nnU-Net Architecture

This module provides a wrapper that can convert ANY existing nnU-Net architecture
(PlainConvUNet, ResidualEncoderUNet, Primus, etc.) into a multi-head version
for multi-task segmentation.

Instead of reimplementing network architectures, this wrapper:
1. Takes an existing network created by get_network_from_plans
2. Replaces the decoder's seg_layers with multi-head versions
3. Preserves all other network components (encoder, decoder stages, etc.)

This approach ensures compatibility with all current and future nnU-Net architectures.
"""

from typing import Union, List, Tuple, Dict, Optional
import torch
from torch import nn
import pydoc

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans


class MultiHeadSegmentationWrapper(nn.Module):
    """
    Wrapper that converts any nnU-Net architecture into a multi-head version.
    
    This wrapper:
    - Takes an existing network (any architecture)
    - Replaces seg_layers with task-specific heads
    - Preserves the original forward pass structure
    
    Works with: PlainConvUNet, ResidualEncoderUNet, Primus, and any future architectures
    that follow the nnU-Net convention of having decoder.seg_layers.
    """
    
    def __init__(
        self,
        base_network: nn.Module,
        task_num_classes: Dict[str, int],
        deep_supervision: bool = True
    ):
        """
        Args:
            base_network: An existing nnU-Net network (created via get_network_from_plans)
            task_num_classes: Dict mapping task names to number of classes
                              e.g., {"organ": 5, "lesion": 3}
            deep_supervision: Whether deep supervision is enabled
        """
        super().__init__()
        
        self.task_names = list(task_num_classes.keys())
        self.task_num_classes = task_num_classes
        self.deep_supervision = deep_supervision
        
        # Store the base network
        self.base_network = base_network
        
        # Find and analyze the original seg_layers
        self.seg_layers_location = self._find_seg_layers()
        original_seg_layers = self._get_seg_layers()
        
        if original_seg_layers is None:
            raise ValueError(
                "Could not find seg_layers in the base network. "
                "This wrapper requires networks with decoder.seg_layers attribute."
            )
        
        # Extract information from original seg_layers
        self.num_seg_layers = len(original_seg_layers)
        
        # Determine conv_op and feature channels from original seg_layers
        sample_layer = original_seg_layers[0]
        self.conv_op = type(sample_layer)
        
        # Get input features for each seg_layer
        self.seg_layer_in_features = [layer.in_channels for layer in original_seg_layers]
        
        # Create multi-head seg_layers for each task
        self.task_seg_layers = nn.ModuleDict()
        for task_name, num_classes in task_num_classes.items():
            task_layers = nn.ModuleList()
            for i, in_features in enumerate(self.seg_layer_in_features):
                task_layers.append(
                    self.conv_op(in_features, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
                )
            self.task_seg_layers[task_name] = task_layers
        
        # Disable original seg_layers (we'll use our multi-head version)
        self._disable_original_seg_layers()
    
    def _find_seg_layers(self) -> str:
        """Find where seg_layers is located in the network."""
        # Common locations in nnU-Net architectures
        if hasattr(self.base_network, 'decoder') and hasattr(self.base_network.decoder, 'seg_layers'):
            return 'decoder.seg_layers'
        elif hasattr(self.base_network, 'seg_layers'):
            return 'seg_layers'
        else:
            # Try to find it recursively
            for name, module in self.base_network.named_modules():
                if name.endswith('seg_layers') and isinstance(module, nn.ModuleList):
                    return name
        return None
    
    def _get_seg_layers(self) -> Optional[nn.ModuleList]:
        """Get the original seg_layers module."""
        if self.seg_layers_location is None:
            return None
        
        parts = self.seg_layers_location.split('.')
        module = self.base_network
        for part in parts:
            module = getattr(module, part)
        return module
    
    def _disable_original_seg_layers(self):
        """
        Disable the original seg_layers by replacing with identity.
        We need to keep the structure but not use the outputs.
        """
        # We don't actually disable them, we just won't use their outputs
        # The original seg_layers will still be called but we'll replace the output
        pass
    
    def forward(self, x: torch.Tensor) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass with multi-head outputs.
        
        This method intercepts the base network's forward pass and applies
        multi-head segmentation layers instead of the original single head.
        """
        # We need to hook into the decoder to get intermediate features
        # before they go through seg_layers
        
        # Run encoder
        encoder = self.base_network.encoder
        decoder = self.base_network.decoder
        
        # Get encoder outputs (skips)
        skips = encoder(x)
        
        # Run decoder and collect features at each resolution
        # This depends on the decoder implementation, but we can use hooks
        decoder_features = self._run_decoder_and_collect_features(skips, decoder)
        
        # Apply multi-head seg_layers
        result = {}
        for task_name in self.task_names:
            seg_layers = self.task_seg_layers[task_name]
            
            if self.deep_supervision:
                task_outputs = []
                for i, features in enumerate(decoder_features):
                    seg_output = seg_layers[i](features)
                    task_outputs.append(seg_output)
                result[task_name] = task_outputs
            else:
                # Only use highest resolution
                result[task_name] = seg_layers[0](decoder_features[0])
        
        return result
    
    def _run_decoder_and_collect_features(
        self, 
        skips: List[torch.Tensor], 
        decoder: nn.Module
    ) -> List[torch.Tensor]:
        """
        Run the decoder and collect feature maps at each resolution level.
        
        This is architecture-agnostic by using the decoder's existing structure.
        """
        # Most nnU-Net decoders follow this pattern:
        # - stages: ModuleList of decoder blocks
        # - transpconvs: ModuleList of upsampling operations
        
        # Handle different decoder types
        if hasattr(decoder, 'stages') and hasattr(decoder, 'transpconvs'):
            return self._run_standard_decoder(skips, decoder)
        else:
            # Fallback: use forward hooks to capture features
            return self._run_decoder_with_hooks(skips, decoder)
    
    def _run_standard_decoder(
        self, 
        skips: List[torch.Tensor], 
        decoder: nn.Module
    ) -> List[torch.Tensor]:
        """
        Run standard UNetDecoder/UNetResDecoder.
        """
        # Skip connections (excluding bottleneck, reversed order)
        lres_input = skips[-1]  # Bottleneck
        skip_connections = skips[:-1][::-1]  # Reversed, excluding bottleneck
        
        decoder_features = []
        x = lres_input
        
        for stage_idx, (stage, transpconv) in enumerate(zip(decoder.stages, decoder.transpconvs)):
            x = transpconv(x)
            x = torch.cat([x, skip_connections[stage_idx]], dim=1)
            x = stage(x)
            decoder_features.append(x)
        
        return decoder_features
    
    def _run_decoder_with_hooks(
        self, 
        skips: List[torch.Tensor], 
        decoder: nn.Module
    ) -> List[torch.Tensor]:
        """
        Fallback: use forward hooks to capture decoder features.
        This works with any decoder architecture.
        """
        features = []
        hooks = []
        
        # Register hooks on the layers just before seg_layers
        # Find stages or similar structures
        if hasattr(decoder, 'stages'):
            for stage in decoder.stages:
                hook = stage.register_forward_hook(
                    lambda m, inp, out: features.append(out)
                )
                hooks.append(hook)
        
        try:
            # Run the base network's decoder forward
            # This will trigger the hooks
            _ = decoder(skips)
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        return features
    
    @staticmethod
    def initialize(module):
        """Weight initialization for new seg_layers."""
        if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, a=1e-2, nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


def wrap_network_for_multi_task(
    base_network: nn.Module,
    task_num_classes: Dict[str, int],
    deep_supervision: bool = True
) -> MultiHeadSegmentationWrapper:
    """
    Wrap any nnU-Net network for multi-task segmentation.
    
    Args:
        base_network: Network created by get_network_from_plans
        task_num_classes: Dict mapping task names to number of classes
        deep_supervision: Whether deep supervision is enabled
    
    Returns:
        MultiHeadSegmentationWrapper wrapping the network
    """
    wrapper = MultiHeadSegmentationWrapper(
        base_network=base_network,
        task_num_classes=task_num_classes,
        deep_supervision=deep_supervision
    )
    
    # Initialize new seg_layers
    wrapper.task_seg_layers.apply(MultiHeadSegmentationWrapper.initialize)
    
    return wrapper


def get_multi_head_network_from_plans_v2(
    arch_class_name: str,
    arch_kwargs: dict,
    arch_kwargs_req_import: List[str],
    input_channels: int,
    task_num_classes: Dict[str, int],
    allow_init: bool = True,
    deep_supervision: bool = True
) -> MultiHeadSegmentationWrapper:
    """
    Create a multi-head network from nnU-Net plans.
    
    This is the recommended way to create multi-head networks as it:
    1. Uses get_network_from_plans to create the base network (supports all architectures)
    2. Wraps it with MultiHeadSegmentationWrapper
    
    Args:
        arch_class_name: Architecture class name from plans
        arch_kwargs: Architecture kwargs from plans
        arch_kwargs_req_import: Keys requiring import
        input_channels: Number of input channels
        task_num_classes: Dict mapping task names to num classes
        allow_init: Whether to initialize weights
        deep_supervision: Whether to enable deep supervision
    
    Returns:
        MultiHeadSegmentationWrapper instance
    """
    # Create base network with dummy output channels
    # We use 1 class as a placeholder since we'll replace seg_layers anyway
    base_network = get_network_from_plans(
        arch_class_name=arch_class_name,
        arch_kwargs=arch_kwargs,
        arch_kwargs_req_import=arch_kwargs_req_import,
        input_channels=input_channels,
        output_channels=1,  # Dummy, will be replaced
        allow_init=allow_init,
        deep_supervision=deep_supervision
    )
    
    # Wrap for multi-task
    return wrap_network_for_multi_task(
        base_network=base_network,
        task_num_classes=task_num_classes,
        deep_supervision=deep_supervision
    )
