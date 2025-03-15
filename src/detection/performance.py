"""
Methods for performance evaluation of object detections
Andreas Werdich
Core for Computational Biomedicine
Harvard Medical School, Boston, MA, USA
"""

import numpy as np
from detection.imageproc import xywh2xyxy, xyxy2xywh, determine_bbox_format

def get_iou(true_bbox: list, pred_bbox: list, bbox_format: str ='xyxy') -> float:
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.

    The IoU is a measure of how much overlap exists between the true bounding box
    and the predicted bounding box. It is calculated as the area of intersection
    divided by the area of union, providing a value between 0 and 1 where 1 indicates
    perfect overlap.

    Args:
        true_bbox (list): A list containing the coordinates of the true bounding box.
            The format is defined by `bbox_format` (e.g., 'xyxy' or 'xywh').
        pred_bbox (list): A list containing the coordinates of the predicted bounding box.
            The format is defined by `bbox_format` (e.g., 'xyxy' or 'xywh').
        bbox_format (str): The format of the bounding boxes. Supported formats are
            'xyxy' (top-left and bottom-right coordinates) or 'xywh' (top-left
            coordinates, width, and height). Defaults to 'xyxy'.

    Returns:
        float: The computed IoU value between 0 and 1.
    """
    # Check bounding box format
    assert determine_bbox_format(true_bbox) == bbox_format, f'true_bbox: format is not consistent with "{bbox_format}"'
    assert determine_bbox_format(pred_bbox) == bbox_format, f'pred_bbox: format is not consistent with "{bbox_format}"'
    ground_truth, pred = true_bbox, pred_bbox
    if bbox_format == 'xywh':
        ground_truth, pred = xywh2xyxy(true_bbox), xywh2xyxy(pred_bbox)
    # coordinates of the area of intersection.
    ix1 = np.maximum(ground_truth[0], pred[0])
    iy1 = np.maximum(ground_truth[1], pred[1])
    ix2 = np.minimum(ground_truth[2], pred[2])
    iy2 = np.minimum(ground_truth[3], pred[3])

    # Intersection height and width.
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))

    area_of_intersection = i_height * i_width

    # Ground Truth dimensions.
    gt_height = ground_truth[3] - ground_truth[1] + 1
    gt_width = ground_truth[2] - ground_truth[0] + 1

    # Prediction dimensions.
    pd_height = pred[3] - pred[1] + 1
    pd_width = pred[2] - pred[0] + 1

    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection

    iou = area_of_intersection / area_of_union

    return iou