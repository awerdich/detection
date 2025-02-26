""" Script to train the detex detector model """
#%% Imports
import os
import numpy as np
import pandas as pd
import logging
import glob
import json
import datetime
from pathlib import Path

# PyTorch framework
import torch
# Hugging Face Library
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
from transformers import TrainingArguments, Trainer
import detection as dt
from detection.detrdataset import get_gpu_info, DetectionDatasetFromDF
from detection.detransform import DetrTransform
from detection.imageproc import clipxywh, ImageData
from detection.mapeval import MAPEvaluator

#%% Set important model parameters
device, device_str = get_gpu_info()
date_str = datetime.date.today().strftime('%y%m%d')

#%% Set up the data directory and define the name of the annotation file
data_root = os.path.join(os.environ.get('HOME'), 'data')
data_dir = os.path.join(data_root, 'dentex_detection')
annotation_file_name = 'train_split_250224.parquet'

# All other directories use the data_dir
model_dir = os.path.join(data_dir, 'model')
Path(model_dir).mkdir(exist_ok=True, parents=True)
log_dir = os.path.join(model_dir, 'log')
Path(log_dir).mkdir(exist_ok=True, parents=True)
image_dir = os.path.join(data_dir, 'quadrants')
annotation_file = os.path.join(image_dir, annotation_file_name)

# Check the images on disk
file_list = glob.glob(os.path.join(image_dir, '*.png'))
expected_n_images = 2531
if not len(file_list) == expected_n_images:
    print(f'WARNING: expected number of images ({expected_n_images}) does not match the number of images on disk.')
    print(f'Delete files and start over.')
else:
    print(f'Found {len(file_list)} images.')

#%% Training and model parameters (will be saved in log file)
model_version = 1
model_name = f'rtdetr_{date_str}_{str(model_version).zfill(2)}'

training_dict = {'model_version': model_version,
                 'model_name': model_name,
                 'hf_checkpoint': 'PekingU/rtdetr_v2_r101vd',
                 'use_transform': 'transform_1'}

training_args_dict = {'output_dir': os.path.join(model_dir, model_name),
                      'num_train_epochs': 20,
                      'max_grad_norm': 0.1,
                      'learning_rate': 5e-5,
                      'warmup_steps': 300,
                      'per_device_train_batch_size': 4,
                      'dataloader_num_workers': 2,
                      'metric_for_best_model': 'eval_map',
                      'greater_is_better': True,
                      'load_best_model_at_end': True,
                      'eval_strategy': 'epoch',
                      'save_strategy': 'epoch',
                      'save_total_limit': 2,
                      'remove_unused_columns': False,
                      'eval_do_concat_batches': False}

#%% Set up logger
log_file_name = f'train_log_{date_str}.log'
log_file = os.path.join(log_dir, log_file_name)
dtfmt = '%y%m%d-%H:%M'
logfmt = '%(asctime)s-%(name)s-%(levelname)s-%(message)s'

logging.basicConfig(filename=log_file,
                    filemode='w',
                    level=logging.INFO,
                    format=logfmt,
                    datefmt=dtfmt)

logger = logging.getLogger(name=__name__)

#%% Load the annotations
df = pd.read_parquet(annotation_file)
label_name_list = sorted(list(df['ada'].unique()))
id2label = dict(zip(range(len(label_name_list)), label_name_list))
id2label = {int(label_id): str(name) for label_id, name in id2label.items()}
label2id = {str(name): int(label_id) for label_id, name in id2label.items()}
# Add columns with the file paths and the labels
df = df.assign(file=df['file_name'].apply(lambda f: os.path.join(image_dir, f)),
               label=df['ada'].apply(lambda name: label2id.get(str(name))))
print(f'Loaded annotation data with {len(df)} samples.')

#%% Load the model and image processor
hf_checkpoint = training_dict.get('hf_checkpoint')
image_processor = RTDetrImageProcessor.from_pretrained(hf_checkpoint)
model = RTDetrV2ForObjectDetection.from_pretrained(hf_checkpoint,
                                                   id2label=id2label,
                                                   label2id=label2id,
                                                   anchor_image_size=None,
                                                   ignore_mismatched_sizes=True)
# Set the evaluation metrics
eval_compute_metrics_fn = MAPEvaluator(image_processor=image_processor, threshold=0.01, id2label=id2label)
training_args = TrainingArguments(**training_args_dict)

#%% Create data sets
use_transform = training_dict.get('use_transform')
print(f'Using transform: {use_transform}')
train_transform = DetrTransform(use_transform).train_transform()
val_transform = DetrTransform(use_transform).val_transform()

# Create the data sets
dset_list = sorted(list(df['dset'].unique()))
print(dset_list)
dataset_dict = {}
for dset in dset_list:
    df_dset = df.loc[df['dset'] == dset]
    if dset == 'train':
        transform = train_transform
    else:
        transform = val_transform
    dataset = DetectionDatasetFromDF(data=df_dset,
                                     processor=image_processor,
                                     file_col='file',
                                     label_col='label',
                                     bbox_col='bbox',
                                     transform=transform,
                                     bbox_format='xywh',
                                     validate=True)
    dataset_dict.update({dset: dataset})
    print(f'Number of images in {dset.upper()}: {len(dataset)}')

#%% Instantiate the Trainer object

# Dump the training arguments into the log file
logger.info(json.dumps(training_args_dict))
logger.info(json.dumps(training_dict))

def collate_fn(batch):
    """
    Collates a batch of data samples into a single dictionary for model input.
    """
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    return data

trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=dataset_dict.get('train'),
                  eval_dataset=dataset_dict.get('val'),
                  processing_class=image_processor,
                  data_collator=collate_fn,
                  compute_metrics=eval_compute_metrics_fn)
#%% Run the training
trainer.train()