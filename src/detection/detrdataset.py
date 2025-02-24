""" PyTorch Dataset for DETR Object Detection Model """
# Imports
from dataclasses import dataclass
import pandas as pd
import logging
from torch.utils.data import Dataset
import torch
import albumentations as alb

from detection.imageproc import ImageData, validate_image_data, clipxywh, determine_bbox_format

logger = logging.getLogger(__name__)

# GPU checks
def get_gpu_info(device_str: str = None):
    if device_str is None:
        is_cuda = torch.cuda.is_available()
        print(f'CUDA available: {is_cuda}')
        print(f'Number of GPUs found:  {torch.cuda.device_count()}')
        if is_cuda:
            print(f'Current device ID: {torch.cuda.current_device()}')
            print(f'GPU device name:   {torch.cuda.get_device_name(0)}')
            print(f'PyTorch version:   {torch.__version__}')
            print(f'CUDA version:      {torch.version.cuda}')
            print(f'CUDNN version:     {torch.backends.cudnn.version()}')
            device_str = 'cuda:0'
            torch.cuda.empty_cache()
        else:
            device_str = 'cpu'
        info_msg = f'Device for model training/inference: {device_str}'
        device = torch.device(device_str)
        logger.info(info_msg)
        return device, device_str

class DetectionDatasetFromDF(Dataset):
    def __init__(self,
                 data:pd.DataFrame,
                 processor,
                 file_col: str,
                 label_col: str,
                 bbox_col: str,
                 transform: alb.Compose=None,
                 bbox_format: str='xywh',
                 validate: bool=False):
        self.data = data
        self.processor = processor
        self.file_col = file_col
        self.label_col = label_col
        self.bbox_col = bbox_col
        self.transform = transform
        self.validate = validate
        self.bbox_format = bbox_format
        self.file_list = list(data[file_col].unique())
        if self.validate:
            self.data = self.validate_input_data()

    def validate_input_data(self):
        """
        Validates the input data for an image dataset, including bounding boxes, and checks for
        transformations applied to the dataset.
        """
        df = validate_image_data(data_df=self.data, file_path_col=self.file_col)
        # Validate bounding boxes
        box_error = 'Bounding box format must be COCO (XYWH)'
        box_format_list = [determine_bbox_format(list(bbox)) for bbox in self.data[self.bbox_col].tolist()]
        assert all(box_format_list) and self.bbox_format == 'xywh', box_error
        assert None not in box_format_list, box_error
        assert 'xywh' in box_format_list, box_error
        # Produce a warning if no transformation is provided
        if not isinstance(self.transform, alb.Compose):
            logger.warning('No transformations provided for data set')
        return df

    @staticmethod
    def annotations_as_coco(image_id, label_list, bbox_list):
        annotations = []
        for label, bbox in zip(label_list, bbox_list):
            x, y, w, h = bbox
            formatted_annotation = {'image_id': image_id,
                                    'category_id': label,
                                    'bbox': bbox,
                                    'iscrowd': 0,
                                    'area': w * h}
            annotations.append(formatted_annotation)
        return {'image_id': image_id, 'annotations': annotations}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        idx = idx % self.__len__()
        file = self.file_list[idx]
        image = ImageData().load_image(file)
        # Histogram equalization improves contrast
        image = ImageData().hist_eq(image)
        bbox_list = self.data.loc[self.data[self.file_col] == file, self.bbox_col].tolist()
        label_list = self.data.loc[self.data[self.file_col] == file, self.label_col].tolist()
        # Clip the bounding boxes to make sure that they are within the limits of the images
        bbox_list = [clipxywh(xywh=list(bbox), xlim=(0, image.shape[1]), ylim=(0, image.shape[0]), decimals=0) \
                     for bbox in bbox_list]
        if self.transform:
            transformed = self.transform(image=image, bboxes=bbox_list, labels=label_list)
            image = transformed['image']
            bbox_list = transformed['bboxes']
            label_list = transformed['labels']
        formatted_annotations = self.annotations_as_coco(image_id=idx,
                                                         label_list=label_list,
                                                         bbox_list=bbox_list)
        # Run the model preprocessing on the input image and the bounding boxes
        output = self.processor(images=image, annotations=formatted_annotations,return_tensors='pt')
        # The processor returns output as a list with images, but we only return one image. Remove the list.
        output = {k: v[0] for k, v in output.items()}
        return output