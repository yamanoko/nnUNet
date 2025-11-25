import torch
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


def load_pretrained_weights(network, fname, verbose=False):
    """
    Transfers all weights between matching keys in state_dicts. matching is done by name and we only transfer if the
    shape is also the same. Segmentation layers (the 1x1(x1) layers that produce the segmentation maps)
    identified by keys ending with '.seg_layers') are not transferred!

    If the pretrained weights were obtained with a training outside nnU-Net and DDP or torch.optimize was used,
    you need to change the keys of the pretrained state_dict. DDP adds a 'module.' prefix and torch.optim adds
    '_orig_mod'. You DO NOT need to worry about this if pretraining was done with nnU-Net as
    nnUNetTrainer.save_checkpoint takes care of that!

    """
    if dist.is_initialized():
        saved_model = torch.load(fname, map_location=torch.device('cuda', dist.get_rank()), weights_only=False)
    else:
        saved_model = torch.load(fname, weights_only=False)
    pretrained_dict = saved_model['network_weights']

    skip_strings_in_pretrained = [
        '.seg_layers.',
    ]

    if isinstance(network, DDP):
        mod = network.module
    else:
        mod = network
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    model_dict = mod.state_dict()

    # Handle input layer shape mismatch for different number of input channels
    # Check if the first conv layer has different input channels
    first_conv_keys = [
        k for k in model_dict.keys()
        if 'conv_blocks_context.0.blocks.0.conv.weight' in k or
        'encoder.stages.0.0.conv.weight' in k
    ]

    for key in first_conv_keys:
        if key in pretrained_dict:
            pretrained_shape = pretrained_dict[key].shape
            model_shape = model_dict[key].shape

            # Check if only input channels differ
            # (shape[1] for conv weights: [out_ch, in_ch, ...])
            if (len(pretrained_shape) == len(model_shape) and
                pretrained_shape[0] == model_shape[0] and
                pretrained_shape[2:] == model_shape[2:] and
                    pretrained_shape[1] != model_shape[1]):

                pretrained_in_ch = pretrained_shape[1]
                model_in_ch = model_shape[1]

                if verbose:
                    print(f"Expanding input channels for {key}: "
                          f"{pretrained_in_ch} -> {model_in_ch}")

                # Repeat the pretrained weights across input channels
                # Strategy: repeat the single channel weights for each
                # new channel
                weight = pretrained_dict[key]

                # Calculate how many times to repeat and if remainder
                repeat_times = model_in_ch // pretrained_in_ch
                remainder = model_in_ch % pretrained_in_ch

                # Repeat the weights
                expanded_weights = weight.repeat(
                    1, repeat_times, *([1] * (len(model_shape) - 2)))

                # Handle remainder channels
                if remainder > 0:
                    expanded_weights = torch.cat([
                        expanded_weights,
                        weight[:, :remainder, ...]
                    ], dim=1)

                # Normalize by dividing by the number of input channels
                # to maintain similar activation magnitudes
                expanded_weights = (expanded_weights / model_in_ch *
                                    pretrained_in_ch)

                # Update the pretrained_dict with expanded weights
                pretrained_dict[key] = expanded_weights

    # verify that all but the segmentation layers have the same shape
    for key, _ in model_dict.items():
        if all([i not in key for i in skip_strings_in_pretrained]):
            assert key in pretrained_dict, \
                f"Key {key} is missing in the pretrained model weights. The pretrained weights do not seem to be " \
                f"compatible with your network."
            assert model_dict[key].shape == pretrained_dict[key].shape, \
                f"The shape of the parameters of key {key} is not the same. Pretrained model: " \
                f"{pretrained_dict[key].shape}; your network: {model_dict[key]}. The pretrained model " \
                f"does not seem to be compatible with your network."

    # fun fact: in principle this allows loading from parameters that do not cover the entire network. For example pretrained
    # encoders. Not supported by this function though (see assertions above)

    # commenting out this abomination of a dict comprehension for preservation in the archives of 'what not to do'
    # pretrained_dict = {'module.' + k if is_ddp else k: v
    #                    for k, v in pretrained_dict.items()
    #                    if (('module.' + k if is_ddp else k) in model_dict) and
    #                    all([i not in k for i in skip_strings_in_pretrained])}

    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict.keys() and all([i not in k for i in skip_strings_in_pretrained])}

    model_dict.update(pretrained_dict)

    print("################### Loading pretrained weights from file ", fname, '###################')
    if verbose:
        print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
        for key, value in pretrained_dict.items():
            print(key, 'shape', value.shape)
        print("################### Done ###################")
    mod.load_state_dict(model_dict)


