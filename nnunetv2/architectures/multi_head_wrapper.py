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

For Primus architecture, a specialized PrimusMultiHeadWrapper is provided that handles
the unique structure of Primus (up_projection instead of decoder.seg_layers).
"""

from typing import Union, List, Dict, Optional, Tuple
import torch
from torch import nn

try:
    from einops import rearrange
    from dynamic_network_architectures.building_blocks.patch_encode_decode import PatchDecode, LayerNormNd
    PRIMUS_AVAILABLE = True
except ImportError:
    PRIMUS_AVAILABLE = False

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
        
        # Note: We manage deep_supervision independently from base_network.decoder.deep_supervision
        # The wrapper's forward() controls whether deep supervision outputs are generated,
        # regardless of the base network's setting. The base network's seg_layers outputs
        # are ignored anyway since we use our own multi-head seg_layers.
        
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
        
        When deep_supervision=False:
            Uses the original decoder's forward pass and only replaces the final seg_layer
            output with multi-head outputs. This preserves the original model flow exactly.
        
        When deep_supervision=True:
            Intercepts decoder features at each resolution level to apply multi-head
            seg_layers. This requires reimplementing the decoder forward pass.
        """
        if not self.deep_supervision:
            # Simple approach: use original decoder, only replace final output
            return self._forward_single_head(x)
        else:
            # Deep supervision: need features at all resolution levels
            return self._forward_deep_supervision(x)
    
    def _forward_single_head(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass without deep supervision.
        
        This method preserves the original model flow exactly by:
        1. Running the original encoder
        2. Running the original decoder (but capturing the final features before seg_layers)
        3. Applying only the multi-head final seg_layers
        
        This approach is safer as it doesn't reimplement the decoder logic.
        """
        encoder = self.base_network.encoder
        decoder = self.base_network.decoder
        
        # Validate decoder has stages attribute
        if not hasattr(decoder, 'stages'):
            raise ValueError(
                "This wrapper requires decoder with 'stages' attribute. "
                f"Got decoder of type {type(decoder).__name__}."
            )
        
        # Run encoder
        skips = encoder(x)
        
        # Get the final decoder features using a hook on the last stage
        final_features = [None]
        
        def capture_final_features(module, inp, out):
            final_features[0] = out
        
        # Register hook on the last decoder stage
        # Use the last seg_layer (highest resolution, matches decoder.stages[-1])
        # Note: task_seg_layers are ordered low-to-high resolution (same as original seg_layers)
        last_stage = decoder.stages[-1]
        hook = last_stage.register_forward_hook(capture_final_features)
        
        try:
            # Run the original decoder forward (this preserves exact original behavior)
            # We ignore the output since we'll compute our own from captured features
            _ = decoder(skips)
        finally:
            hook.remove()
        
        # Apply multi-head seg_layers to final features only
        result = {}
        for task_name in self.task_names:
            seg_layer = self.task_seg_layers[task_name][-1]
            result[task_name] = seg_layer(final_features[0])
        
        return result
    
    def _forward_deep_supervision(
        self, x: torch.Tensor
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Forward pass with deep supervision.
        
        This requires capturing decoder features at all resolution levels,
        which means we need to reimplement/intercept the decoder's forward pass.
        
        Note: This approach may not work correctly for all decoder architectures.
        For maximum compatibility, consider using deep_supervision=False.
        """
        encoder = self.base_network.encoder
        decoder = self.base_network.decoder
        
        # Get encoder outputs (skips)
        skips = encoder(x)
        
        # Collect features at each resolution using hooks
        decoder_features = self._collect_decoder_features_with_hooks(skips, decoder)
        
        # Apply multi-head seg_layers at each resolution
        # decoder_features: [0]=highest res, [-1]=lowest res (reversed in _collect...)
        # task_seg_layers:  [0]=lowest res,  [-1]=highest res (same as original)
        # So we need to reverse the seg_layers indexing
        result = {}
        num_layers = len(decoder_features)
        
        # Validate that we have the same number of features as seg_layers
        if num_layers != self.num_seg_layers:
            raise RuntimeError(
                f"Mismatch between decoder features ({num_layers}) and "
                f"seg_layers ({self.num_seg_layers}). This may indicate an "
                "incompatible decoder architecture."
            )
        
        for task_name in self.task_names:
            seg_layers = self.task_seg_layers[task_name]
            task_outputs = []
            for i, features in enumerate(decoder_features):
                # Map: features[0] (highest) -> seg_layers[-1] (highest)
                #      features[-1] (lowest) -> seg_layers[0] (lowest)
                seg_layer_idx = num_layers - 1 - i
                seg_output = seg_layers[seg_layer_idx](features)
                task_outputs.append(seg_output)
            result[task_name] = task_outputs
        
        return result
    
    def _collect_decoder_features_with_hooks(
        self, 
        skips: List[torch.Tensor], 
        decoder: nn.Module
    ) -> List[torch.Tensor]:
        """
        Collect decoder features at all resolution levels using forward hooks.
        
        This preserves the original decoder's forward logic while capturing
        intermediate features needed for deep supervision.
        
        The features are returned in order from highest to lowest resolution
        (matching the expected order for deep supervision).
        """
        features = []
        hooks = []
        
        # Register hooks on all decoder stages
        if hasattr(decoder, 'stages'):
            for stage in decoder.stages:
                # Use a default argument to capture the current stage reference
                def make_hook():
                    def hook(m, inp, out):
                        features.append(out)
                    return hook
                h = stage.register_forward_hook(make_hook())
                hooks.append(h)
        else:
            raise ValueError(
                "Deep supervision requires decoder with 'stages' attribute. "
                "Consider using deep_supervision=False for this architecture."
            )
        
        try:
            # Run the original decoder forward
            _ = decoder(skips)
        finally:
            # Remove all hooks
            for h in hooks:
                h.remove()
        
        # Features are collected low-to-high resolution, reverse for deep supervision
        # (deep supervision expects highest resolution first)
        return features[::-1]
    
    @staticmethod
    def initialize(module):
        """Weight initialization for new seg_layers."""
        if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, a=1e-2, nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


class PrimusMultiHeadWrapper(nn.Module):
    """
    Multi-head wrapper specifically designed for Primus architecture.
    
    Primus has a fundamentally different structure than standard nnU-Net:
    - Uses `eva` (Vision Transformer) as encoder instead of CNN encoder
    - Uses `up_projection` (PatchDecode) instead of decoder.seg_layers
    - No skip connections or deep supervision support
    
    This wrapper:
    - Shares the encoding path (down_projection + eva)
    - Creates task-specific PatchDecode instances for each task
    - Returns Dict[str, torch.Tensor] output format
    
    Note: Deep supervision is NOT supported for Primus architecture.
    """
    
    def __init__(
        self,
        base_network: nn.Module,
        task_num_classes: Dict[str, int],
        patch_embed_size: Tuple[int, ...] = (8, 8, 8),
        decoder_norm=None,
        decoder_act=None
    ):
        """
        Args:
            base_network: A Primus network instance
            task_num_classes: Dict mapping task names to number of classes
                              e.g., {"organ": 5, "lesion": 3}
            patch_embed_size: Patch embedding size (typically (8, 8, 8) for Primus)
            decoder_norm: Normalization for decoder (default: LayerNormNd)
            decoder_act: Activation for decoder (default: nn.GELU)
        """
        super().__init__()
        
        if not PRIMUS_AVAILABLE:
            raise ImportError(
                "PrimusMultiHeadWrapper requires 'einops' and 'dynamic_network_architectures' packages. "
                "Install with: pip install einops dynamic-network-architectures"
            )
        
        self.task_names = list(task_num_classes.keys())
        self.task_num_classes = task_num_classes
        self.deep_supervision = False  # Primus does not support deep supervision
        
        # Store the base network
        self.base_network = base_network
        
        # Extract configuration from base network's up_projection
        original_up_proj = base_network.up_projection
        
        # Get embed_dim from the first layer of PatchDecode's decode sequential
        # PatchDecode.decode is a nn.Sequential of ConvTranspose3d layers
        if hasattr(original_up_proj, 'decode') and len(original_up_proj.decode) > 0:
            first_block = original_up_proj.decode[0]
            if isinstance(first_block, nn.Sequential) and len(first_block) > 0:
                # Structure: decode[i] = Sequential(ConvTranspose3d, Norm, Act)
                embed_dim = first_block[0].in_channels
            else:
                # Direct ConvTranspose3d
                embed_dim = first_block.in_channels
        else:
            raise ValueError("Could not determine embed_dim from base network's up_projection")
        
        self.embed_dim = embed_dim
        self.patch_embed_size = patch_embed_size
        
        # Set defaults for norm and activation
        if decoder_norm is None:
            decoder_norm = LayerNormNd
        if decoder_act is None:
            decoder_act = nn.GELU
        
        # Create task-specific PatchDecode (up_projection) for each task
        self.task_up_projections = nn.ModuleDict()
        for task_name, num_classes in task_num_classes.items():
            self.task_up_projections[task_name] = PatchDecode(
                patch_size=patch_embed_size,
                embed_dim=embed_dim,
                out_channels=num_classes,
                norm=decoder_norm,
                activation=decoder_act
            )
        
        # Initialize the new task-specific decoders
        from dynamic_network_architectures.initialization.weight_init import InitWeights_He
        for task_decoder in self.task_up_projections.values():
            task_decoder.apply(InitWeights_He(1e-2))
    
    def forward(self, x: torch.Tensor, ret_mask: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-head outputs.
        
        Shares the encoding path (down_projection + eva) and applies task-specific
        PatchDecode for each task.
        
        Args:
            x: Input tensor (B, C, W, H, D)
            ret_mask: Whether to return restoration mask (not used in multi-head)
        
        Returns:
            Dict mapping task names to segmentation outputs
        """
        base = self.base_network
        
        # Store full input dimensions for mask computation if needed
        FW, FH, FD = x.shape[2:]
        
        # Shared encoding: down_projection (PatchEmbed)
        x = base.down_projection(x)
        B, C, W, H, D = x.shape
        num_patches = W * H * D
        
        # Rearrange for transformer: (B, C, W, H, D) -> (B, num_patches, C)
        x = rearrange(x, "b c w h d -> b (w h d) c")
        
        # Add register tokens if present
        if base.register_tokens is not None:
            x = torch.cat(
                (base.register_tokens.expand(x.shape[0], -1, -1), x),
                dim=1
            )
        
        # Shared encoding: EVA transformer
        x, keep_indices = base.eva(x)
        
        # Remove register tokens if they were added
        if base.register_tokens is not None:
            x = x[:, base.register_tokens.shape[1]:]
        
        # Restore full sequence (handle patch dropout)
        restored_x, restoration_mask = base.restore_full_sequence(x, keep_indices, num_patches)
        
        # Rearrange back to spatial format: (B, num_patches, C) -> (B, C, W, H, D)
        x = rearrange(restored_x, "b (w h d) c -> b c w h d", h=H, w=W, d=D)
        
        # Task-specific decoding
        outputs = {}
        for task_name in self.task_names:
            outputs[task_name] = self.task_up_projections[task_name](x)
        
        return outputs
    
    def set_deep_supervision_enabled(self, enabled: bool):
        """Primus does not support deep supervision - this is a no-op."""
        if enabled:
            import warnings
            warnings.warn(
                "Primus architecture does not support deep supervision. "
                "This setting will be ignored."
            )
    
    @staticmethod
    def initialize(module):
        """Weight initialization for new decoders."""
        if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, a=1e-2, nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


def wrap_primus_for_multi_task(
    base_network: nn.Module,
    task_num_classes: Dict[str, int],
    patch_embed_size: Tuple[int, ...] = (8, 8, 8),
    decoder_norm=None,
    decoder_act=None
) -> PrimusMultiHeadWrapper:
    """
    Wrap a Primus network for multi-task segmentation.
    
    Args:
        base_network: Primus network instance
        task_num_classes: Dict mapping task names to number of classes
        patch_embed_size: Patch embedding size (typically (8, 8, 8))
        decoder_norm: Normalization for decoder (default: LayerNormNd)
        decoder_act: Activation for decoder (default: nn.GELU)
    
    Returns:
        PrimusMultiHeadWrapper wrapping the network
    """
    return PrimusMultiHeadWrapper(
        base_network=base_network,
        task_num_classes=task_num_classes,
        patch_embed_size=patch_embed_size,
        decoder_norm=decoder_norm,
        decoder_act=decoder_act
    )


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
