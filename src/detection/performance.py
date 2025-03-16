"""
Methods for performance evaluation of object detections
Andreas Werdich
Core for Computational Biomedicine
Harvard Medical School, Boston, MA, USA
"""

import numpy as np
from detection.imageproc import xywh2xyxy, xyxy2xywh, determine_bbox_format

def get_iou(bbox_1: list, bbox_2: list, bbox_format: str = 'xyxy') -> float:
    """
    Calculates the Intersection over Union (IoU) for two bounding boxes.
    The IoU is a measure used to quantify the overlap between two bounding
    boxes. It is computed as the ratio of the intersection area to the union
    area of the two bounding boxes. This function supports bounding boxes in
    two different formats: 'xyxy' (top-left and bottom-right coordinates) or
    'xywh' (top-left coordinates, width, and height). If the format is 'xywh',
    the bounding boxes are converted to 'xyxy' before the calculation.
    Args:
        bbox_1 (list): The first bounding box provided as a list of coordinates.
        bbox_2 (list): The second bounding box provided as a list of coordinates.
        bbox_format (str): The format of the bounding boxes ('xyxy' or 'xywh').
            Defaults to 'xyxy'.
    Returns:
        float: The Intersection over Union (IoU) value, a ratio between 0 and 1.
    """
    if bbox_format == 'xywh':
        bbox_1, bbox_2 = xywh2xyxy(bbox_1), xywh2xyxy(bbox_2)
    # coordinates of the area of intersection.
    ix1 = np.maximum(bbox_1[0], bbox_2[0])
    iy1 = np.maximum(bbox_1[1], bbox_2[1])
    ix2 = np.minimum(bbox_1[2], bbox_2[2])
    iy2 = np.minimum(bbox_1[3], bbox_2[3])

    # Intersection height and width.
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))

    area_of_intersection = i_height * i_width

    # Ground Truth dimensions.
    gt_height = bbox_1[3] - bbox_1[1] + 1
    gt_width = bbox_1[2] - bbox_1[0] + 1

    # Prediction dimensions.
    pd_height = bbox_2[3] - bbox_2[1] + 1
    pd_width = bbox_2[2] - bbox_2[0] + 1

    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection

    iou = area_of_intersection / area_of_union

    return iou