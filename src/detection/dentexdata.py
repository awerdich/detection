"""
Tools for working with the Dentex Challenge data
Andreas Werdich
Center for Computational Biomedicine
"""

import os
import numpy as np
import pandas as pd
import logging
import tarfile
import json

from detection.fileutils import FileOP

logger = logging.getLogger(__name__)


def val_test_split(data, label_col=None, n_test_per_class=30, n_val_per_class=30, random_state=123):
    """
    Splits the given data into training, validation, and test sets based on the specified parameters.
    Args:
        data: A pandas DataFrame containing the data to be split.
        label_col: The name of the column in the data DataFrame that contains the labels.
        n_test_per_class: The number of samples per class to be allocated to the test set. Default is 30.
        n_val_per_class: The number of samples per class to be allocated to the validation set. Default is 30.
        random_state: The seed value used by the random number generator. Default is 123.
    Returns:
        A new pandas DataFrame with an additional column, 'dataset', indicating the split for each sample.
        The 'dataset' column will have one of the following values: 'train', 'val', or 'test'.
    """
    image_numbers = {'test': n_test_per_class,
                     'val': n_val_per_class}
    dset_df = data.copy().sample(frac=1, random_state=random_state). \
        assign(dataset=None).reset_index(drop=True)
    for dataset in image_numbers.keys():
        if label_col is not None:
            labels = dset_df[label_col].unique()
            for label in labels:
                np.random.seed(random_state)
                idx_list = np.random.choice(dset_df. \
                                            loc[(dset_df[label_col] == label) & (dset_df['dataset'].isnull())]. \
                                            index, size=image_numbers.get(dataset), replace=False)
                dset_df.loc[dset_df.index.isin(idx_list), 'dataset'] = dataset
        else:
            np.random.seed(random_state)
            idx_list = np.random.choice(dset_df.
                                        loc[dset_df['dataset'].isnull()].
                                        index, size=image_numbers.get(dataset), replace=False)
            dset_df.loc[dset_df.index.isin(idx_list), 'dataset'] = dataset

    # Use the remaining samples for training
    dset_df.loc[dset_df['dataset'].isnull(), 'dataset'] = 'train'
    return dset_df

def fdi_ada_mapping() -> dict:
    """ Returns a dictionary mapping FDI to ADA numbers"""
    fdi, ada = [], []
    for q in range(1, 5):
        for p in range(1, 9):
            fdi.append(int(str(q) + str(p)))
        if q in [1, 3]:
            for a in range(8*q, 8*(q-1), -1):
                ada.append(a)
        else:
            for a in range(8*(q-1), 8*q, 1):
                ada.append(a + 1)
    return dict(zip(fdi, ada))

class DentexData:
    def __init__(self, data_dir):
        self.annotations_file = None
        self.annotations = None
        self.data_dir = data_dir
        self.classification_url = 'https://dsets.s3.amazonaws.com/dentex/dentex-quadrant-enumeration-disease.tar.gz'
        self.detection_url = 'https://dsets.s3.amazonaws.com/dentex/dentex-quadrant-enumeration.tar.gz'
        if not os.path.exists(self.data_dir):
            logger.warning('Data directory does not exist')

    def annotations_to_df(self):
        """
        Transforms the instance's annotation data into a pandas DataFrame.

        This method processes the annotation data stored in the instance by converting
        each component (e.g., images, annotations, categories) into pandas DataFrames
        and merging them together into a unified format. It includes additional
        processing to handle category-level information.

        Returns:
            pandas.DataFrame: A DataFrame containing the combined annotation data
            with resolved category and image information if annotations are available.
            If no annotations are loaded, returns None.

        Raises:
            None
        """
        if self.annotations is not None:
            annotations_dict = {k: pd.DataFrame(v) for k, v in self.annotations.items()}
            images = annotations_dict.get('images'). \
                rename(columns={'id': 'image_id'})
            output = annotations_dict.get('annotations')
            output = output.merge(images, on='image_id', how='left')
            # Get a list of category tables in the annotation dictionary
            category_names = sorted([k for k in annotations_dict.keys() if 'cat' in k])
            category_column_names = {category: f'category_id_{c + 1}' \
                                     for c, category in enumerate(category_names)}
            for category, column_name in category_column_names.items():
                category_df = annotations_dict.get(category). \
                    rename(columns={'id': column_name, 'name': f'{column_name}_name'}). \
                    drop('supercategory', axis=1)
                output = output.merge(category_df, on=column_name, how='left')
        else:
            logger.error('No annotations loaded')
            output = None
        return output

    def download_image_data(self, url):
        """
        Downloads image data from a specified URL
        Parameters:
        - url (str): The URL from which to download the image data.
        If not provided, the method will use the default classification URL.
        Returns:
        - data_tar_file (str): The path to the downloaded data TAR file.
        Raises:
        - ValueError: If an incomplete file exists at the specified URL.
        Example usage:
        download_image_data(url='https://www.example.com/data.tar')
        """
        data_tar_file = FileOP().download_from_url(url=url, download_dir=self.data_dir)
        if data_tar_file is not None and os.path.exists(data_tar_file):
            try:
                with tarfile.open(data_tar_file) as tar:
                    tar.extractall(path=self.data_dir)
            except Exception as e:
                logger.error(f'Could not extract: {e}')
        return data_tar_file

    def load_annotations(self, json_file):
        try:
            with open(json_file, 'r') as f:
                self.annotations = json.load(f)
                self.annotations_file = json_file
        except IOError as e:
            logger.error(f'Could not read {json_file}: {e}')
        return self.annotations
