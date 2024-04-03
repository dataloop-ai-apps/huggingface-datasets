import logging
from datasets import load_dataset
import dtlpy as dl
import tempfile
import os
import json
import threading

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
            "categories": [{"id": 1, "name": "Chair"}, {"id": 2, "name": "Sofa"}, {"id": 3, "name": "Table"}]
        }

        self.datasets_hug = load_dataset("Francesco/furniture-ngpea", split="train")
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

    def upload_dataset(self, dataset: dl.Dataset, source: str):
        """
        Prepares and uploads the dataset to the Dataloop platform.

        :param dataset: The Dataloop dataset object where the data will be uploaded.
        :param source: The source of the dataset, used for logging purposes.
        """
        temp_dir = tempfile.TemporaryDirectory()
        temp_dir_ann = tempfile.TemporaryDirectory()
        images_and_paths = []
        self.logger.info('Uploading dataset...')
        for item in self.datasets_hug:
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
                for obj_id, area, bbox, category in zip(objects["id"], objects["area"], objects["bbox"], objects["category"])
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
