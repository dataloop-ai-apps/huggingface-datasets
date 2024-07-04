from dtlpyconverters.uploaders import ConvertersUploader
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
import dtlpy as dl
import threading
import tempfile
import logging
import json
import os
import io
import requests
import zipfile

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
            "categories": [{"id": 1, "name": "Chair"}, {"id": 2, "name": "Sofa"}, {"id": 3, "name": "Table"}]
        }

        self.logger.info('Hugging Face dataset loaded')

    @staticmethod
    def save_image(image, path, semaphore):
        """
        Saves an image to a specified path while using a semaphore to limit
        the number of concurrent threads.

        :param image: The image to be saved.
        :param path: The file path where the image will be saved.
        :param semaphore: A threading.Semaphore instance to control concurrency.
        """
        with semaphore:
            image.save(path)

    def save_images_in_threads(self, images_and_paths, max_threads):
        """
        Saves multiple images in parallel using threading, controlled by a semaphore.

        :param images_and_paths: A list of tuples, each containing an image and its save path.
        :param max_threads: The maximum number of threads to use for saving images.
        """
        semaphore = threading.Semaphore(max_threads)
        threads = []

        for image, path in images_and_paths:
            thread = threading.Thread(target=self.save_image, args=(image, path, semaphore))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    def _upload_single_text(self, dataset: dl.Dataset, text: str, filename: str, annotations: dl.AnnotationCollection):
        try:
            byte_io = io.BytesIO()
            byte_io.name = filename
            byte_io.write(text.encode())
            byte_io.seek(0)
            item = dataset.items.upload(local_path=byte_io)
            item.annotations.upload(annotations=annotations)
        except Exception:
            self.logger.exception('failed while uploading')

    def upload_dataset_text(self, dataset: dl.Dataset, source: str):

        self.logger.info('Downloading zip file...')
        url = "https://storage.googleapis.com/model-mgmt-snapshots/datasets_imdb/vectors.zip"
        direc = os.getcwd()
        zip_dir = os.path.join(direc, 'vectors.zip')

        response = requests.get(url)
        if response.status_code == 200:
            with open(zip_dir, 'wb') as f:
                f.write(response.content)
        else:
            self.logger.error(f'Failed to download the file. Status code: {response.status_code}')
            return

        with zipfile.ZipFile(zip_dir, 'r') as zip_ref:
            zip_ref.extractall(direc)
        self.logger.info('Zip file downloaded and extracted.')

        config = dataset.metadata['system'].get('importConfig', dict())
        id_to_label_map = config.get('id_to_label_map')
        hf_location = source.replace('https://huggingface.co/datasets/', '')
        datasets_hug = load_dataset(hf_location, split="train")

        pool = ThreadPoolExecutor(max_workers=32)
        for i_item, item in enumerate(datasets_hug):
            text = item['text']
            label = item['label']
            filename = f'{i_item:05}.txt'
            annotations = dl.AnnotationCollection()
            annotations.add(annotation_definition=dl.Classification(label=id_to_label_map[str(label)]))
            pool.submit(self._upload_single_text,
                        text=text,
                        filename=filename,
                        dataset=dataset,
                        annotations=annotations)

        pool.shutdown()

        feature_set = self.ensure_feature_set(dataset)

        filters = dl.Filters()
        filters.add(field='filename', values=["/07235.txt", "/14141.txt", "/14142.txt", "/04003.txt", "/16696.txt"], operator=dl.FiltersOperations.IN)
        dataset.items.delete(filters=filters)

        # Upload features
        vectors_file = os.path.join(direc, 'vectors/vectors.json')
        with open(vectors_file, 'r') as f:
            vectors = json.load(f)

        with ThreadPoolExecutor(max_workers=32) as executor:
            for key, value in vectors.items():
                executor.submit(self.create_feature, key, value, dataset, feature_set)

        self.logger.info('Dataset uploaded successfully')

    def upload_dataset_coco(self, dataset: dl.Dataset, source: str):
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
            self.coco_format["images"].append({
                "id": item["image_id"],
                "width": item["width"],
                "height": item["height"],
                "file_name": f"image_{item['image_id']}.jpg"
            })

            # Add annotations for each object
            objects = item["objects"]
            self.coco_format["annotations"].extend([
                {
                    "id": obj_id,
                    "image_id": item["image_id"],
                    "category_id": category,
                    "bbox": bbox,
                    "area": area
                }
                for obj_id, area, bbox, category in
                zip(objects["id"], objects["area"], objects["bbox"], objects["category"])
            ])

        max_threads = 10
        self.save_images_in_threads(images_and_paths, max_threads)

        label_file = os.path.join(temp_dir_ann.name, "coco_format.json")
        with open(label_file, "w") as f:
            json.dump(self.coco_format, f)
        self.logger.info('Images and annotations saved temporarily')

        loop = self.converter._get_event_loop()
        loop.run_until_complete(self.converter.coco_to_dataloop(dataset=dataset,
                                                                input_items_path=temp_dir.name,
                                                                input_annotations_path=temp_dir_ann.name,
                                                                coco_json_filename='coco_format.json',
                                                                annotation_options=[dl.AnnotationType.BOX]))
        self.logger.info('Dataset uploaded successfully')

    @staticmethod
    def ensure_feature_set(dataset):
        """
        Ensures that the feature set exists or creates a new one if not found.

        :param dataset: The dataset where the feature set is to be managed.
        """
        try:
            feature_set = dataset.project.feature_sets.get(feature_set_name='clip-feature-set')
            logger.info(f'Feature Set found! Name: {feature_set.name}, ID: {feature_set.id}')
        except dl.exceptions.NotFound:
            logger.info('Feature Set not found, creating...')
            feature_set = dataset.project.feature_sets.create(
                name='clip-feature-set',
                entity_type=dl.FeatureEntityType.ITEM,
                project_id=dataset.project.id,
                set_type='clip',
                size=512
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
