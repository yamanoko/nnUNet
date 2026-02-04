import numpy as np
from scipy.ndimage import binary_fill_holes
from acvl_utils.cropping_and_padding.bounding_boxes import get_bbox_from_mask, bounding_box_to_slice


def create_nonzero_mask(data):
    """

    :param data:
    :return: the mask is True where the data is nonzero
    """
    assert data.ndim in (3, 4), "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = data[0] != 0
    for c in range(1, data.shape[0]):
        nonzero_mask |= data[c] != 0
    return binary_fill_holes(nonzero_mask)


def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    """

    :param data: image data with shape (C, X, Y, Z) or (C, X, Y)
    :param seg: segmentation with shape (1, X, Y, Z), (1, X, Y) for single-task,
                or (num_tasks, X, Y, Z), (num_tasks, X, Y) for multi-task
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask)
    slicer = bounding_box_to_slice(bbox)
    nonzero_mask = nonzero_mask[slicer][None]
    
    slicer = (slice(None), ) + slicer
    data = data[slicer]
    if seg is not None:
        seg = seg[slicer]
        # For multi-task segmentation (num_tasks, X, Y, Z), we need to broadcast
        # nonzero_mask across all task channels
        num_seg_channels = seg.shape[0]
        if num_seg_channels > 1:
            # Multi-task: broadcast nonzero_mask to match seg shape
            nonzero_mask_broadcast = np.broadcast_to(nonzero_mask, seg.shape)
            seg[(seg == 0) & (~nonzero_mask_broadcast)] = nonzero_label
        else:
            # Single-task: original behavior
            seg[(seg == 0) & (~nonzero_mask)] = nonzero_label
    else:
        seg = np.where(nonzero_mask, np.int8(0), np.int8(nonzero_label))
    return data, seg, bbox


