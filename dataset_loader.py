import json
import logging
import os
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import dtlpy as dl
import requests
from datasets import load_dataset
from dtlpyconverters.uploaders import ConvertersUploader

logger = logging.getLogger(name='dataset-huggingFace')


class DatasetHF(dl.BaseServiceRunner):
    """
    A class for handling the process of loading a dataset from HuggingFace,
    preparing it, and uploading it to the Dataloop platform.
    """

    def __init__(self):
        """
        Initializes the DatasetHF class by setting up the logger, converter,
        COCO format template, and loading the dataset from HuggingFace.
        """
        self.logger = logger
        self.logger.info('Initializing HuggingFace dataset loader')
        self.converter = ConvertersUploader()
        self.coco_format = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "Chair"},
                {"id": 2, "name": "Sofa"},
                {"id": 3, "name": "Table"},
            ],
        }

        self.logger.info('Hugging Face dataset loaded')

    @staticmethod
    def save_image(image, path):
        """
        Saves an image to a specified path.

        :param image: The image to be saved.
        :param path: The file path where the image will be saved.
        """
        image.save(path)

    def save_images_in_threads(self, images_and_paths, max_threads, progress):
        """
        Saves multiple images in parallel using threading, controlled by a semaphore.

        :param images_and_paths: A list of tuples, each containing an image and its save path.
        :param max_threads: The maximum number of threads to use for saving images.
        """
        with ThreadPoolExecutor(max_threads) as executor:
            futures = [
                executor.submit(self.save_image, image, path)
                for image, path in images_and_paths
            ]

            self.upload_progress(
                progress, futures, 'Uploading items and annotations ...', 0, 100
            )

    def upload_dataset_text(self, dataset: dl.Dataset, source: str, progress=None):
        """
        Uploads a text dataset to the specified destination.

        Args:
            dataset (dl.Dataset): The dataset object to upload.
            source (str): The source URL of the dataset.
            progress (optional): An optional progress object to track the upload progress.

        Returns:
            None

        Raises:
            requests.exceptions.RequestException: If there is an issue with the HTTP request.
            zipfile.BadZipFile: If the downloaded zip file is corrupted.
            KeyError: If there is an issue with the dataset metadata or configuration.
        """

        if progress is not None:
            progress.update(
                progress=0, message='Creating dataset...', status='Creating dataset...'
            )

        self.logger.info('Downloading zip file...')
        url = (
            "https://storage.googleapis.com/model-mgmt-snapshots/datasets_imdb/imdb.zip"
        )
        direc = os.getcwd()
        zip_dir = os.path.join(direc, 'imdb.zip')

        response = requests.get(url, timeout=100)
        if response.status_code == 200:
            with open(zip_dir, 'wb') as f:
                f.write(response.content)
        else:
            self.logger.error(
                'Failed to download the file. Status code: %s', response.status_code
            )
            return

        with zipfile.ZipFile(zip_dir, 'r') as zip_ref:
            zip_ref.extractall(direc)
        self.logger.info('Zip file downloaded and extracted.')

        config = dataset.metadata['system'].get('importConfig', dict())
        id_to_label_map = config.get('id_to_label_map', {"0": "neg", "1": "pos"})

        if progress is not None:
            progress.update(
                progress=0,
                message='Uploading items and annotations ...',
                status='Uploading items and annotations ...',
            )

        progress_tracker = {'last_progress': 0}

        def progress_callback_all(progress_class, progress, context):
            new_progress = progress // 2
            if (
                new_progress > progress_tracker['last_progress']
                and new_progress % 5 == 0
            ):
                logger.info(f'Progress: {new_progress}%')
                progress_tracker['last_progress'] = new_progress
                if progress_class is not None:
                    progress_class.update(
                        progress=new_progress,
                        message='Uploading items and annotations ...',
                        status='Uploading items and annotations ...',
                    )

        progress_callback = partial(progress_callback_all, progress)

        dl.client_api.add_callback(
            func=progress_callback, event=dl.CallbackEvent.ITEMS_UPLOAD
        )

        # Upload features
        vectors_file = os.path.join(direc, 'vectors/vectors.json')
        with open(vectors_file, 'r') as f:
            vectors = json.load(f)

        annotations_files = os.path.join(direc, 'annotations/')
        items_files = os.path.join(direc, 'items/')
        dataset.items.upload(
            local_path=items_files,
            local_annotations_path=annotations_files,
        )

        # Setup dataset recipe and ontology
        recipe = dataset.recipes.list()[0]
        ontology = recipe.ontologies.list()[0]
        ontology.add_labels(label_list=['pos', 'neg'])

        feature_set = self.ensure_feature_set(dataset)

        with ThreadPoolExecutor(max_workers=32) as executor:
            vector_features = [
                executor.submit(self.create_feature, key, value, dataset, feature_set)
                for key, value in vectors.items()
            ]

            self.upload_progress(
                progress, vector_features, 'Uploading feature set ...', 50, 100
            )

        self.logger.info('Dataset uploaded successfully')

    @staticmethod
    def upload_progress(progress, futures, massage, min_progress, max_progress):
        """
        Tracks and logs the progress of a set of asynchronous tasks.

        Args:
            progress (object): An object that has an `update` method to report progress.
            futures (list): A list of futures representing the asynchronous tasks.
            massage (str): A message to be logged and passed to the progress object.
            min_progress (int): The minimum progress value (usually 0).
            max_progress (int): The maximum progress value (usually 100).

        Logs:
            Logs the progress percentage at each step.

        Updates:
            Calls the `update` method of the `progress` object with the new progress value and message.
        """
        total_tasks = len(futures)
        tasks_completed = 0
        task_progress = 0
        for _ in as_completed(futures):
            tasks_completed += 1
            new_progress = (
                int(tasks_completed / total_tasks * (max_progress - min_progress))
                + min_progress
            )
            if new_progress > task_progress and new_progress % 1 == 0:
                logger.info(f'Progress: {new_progress}%')
                task_progress = new_progress
                if progress is not None:
                    progress.update(
                        progress=new_progress,
                        message=massage,
                        status=massage,
                    )

    def upload_dataset_coco(self, dataset: dl.Dataset, source: str, progress=None):
        """
        Prepares and uploads the dataset to the Dataloop platform.

        :param dataset: The Dataloop dataset object where the data will be uploaded.
        :param source: The source of the dataset, used for logging purposes.
        """
        temp_dir = tempfile.TemporaryDirectory()
        temp_dir_ann = tempfile.TemporaryDirectory()
        images_and_paths = []
        self.logger.info('Uploading dataset...')
        hf_location = source.replace('https://huggingface.co/datasets/', '')
        datasets_hug = load_dataset(hf_location, split="train")
        for item in datasets_hug:
            image_path = os.path.join(temp_dir.name, f"image_{item['image_id']}.jpg")
            images_and_paths.append((item["image"], image_path))

            # Add image information
            self.coco_format["images"].append(
                {
                    "id": item["image_id"],
                    "width": item["width"],
                    "height": item["height"],
                    "file_name": f"image_{item['image_id']}.jpg",
                }
            )

            # Add annotations for each object
            objects = item["objects"]
            self.coco_format["annotations"].extend(
                [
                    {
                        "id": obj_id,
                        "image_id": item["image_id"],
                        "category_id": category,
                        "bbox": bbox,
                        "area": area,
                    }
                    for obj_id, area, bbox, category in zip(
                        objects["id"],
                        objects["area"],
                        objects["bbox"],
                        objects["category"],
                    )
                ]
            )

        max_threads = 10
        self.save_images_in_threads(images_and_paths, max_threads, progress)

        label_file = os.path.join(temp_dir_ann.name, "coco_format.json")
        with open(label_file, "w") as f:
            json.dump(self.coco_format, f)
        self.logger.info('Images and annotations saved temporarily')

        loop = self.converter._get_event_loop()
        loop.run_until_complete(
            self.converter.coco_to_dataloop(
                dataset=dataset,
                input_items_path=temp_dir.name,
                input_annotations_path=temp_dir_ann.name,
                coco_json_filename='coco_format.json',
                annotation_options=[dl.AnnotationType.BOX],
            )
        )
        self.logger.info('Dataset uploaded successfully')

    @staticmethod
    def ensure_feature_set(dataset):
        """
        Ensures that the feature set exists or creates a new one if not found.

        :param dataset: The dataset where the feature set is to be managed.
        """
        try:
            feature_set = dataset.project.feature_sets.get(
                feature_set_name='clip-feature-set'
            )
            logger.info(
                'Feature Set found! Name: %s, ID: %s', feature_set.name, feature_set.id
            )
        except dl.exceptions.NotFound:
            logger.info('Feature Set not found, creating...')
            feature_set = dataset.project.feature_sets.create(
                name='clip-feature-set',
                entity_type=dl.FeatureEntityType.ITEM,
                project_id=dataset.project.id,
                set_type='clip',
                size=512,
            )
        return feature_set

    @staticmethod
    def create_feature(key, value, dataset, feature_set):
        """
        Creates a feature for a given item.

        :param key: The key identifying the item.
        :param value: The feature value to be added.
        :param dataset: The dataset containing the item.
        :param feature_set: The feature set to which the feature will be added.
        """

        item = dataset.items.get(filepath=key)
        feature_set.features.create(entity=item, value=value)
