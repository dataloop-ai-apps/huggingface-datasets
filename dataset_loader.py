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
        # dataset = dl.datasets.get(None, '664445f66d405a38aebfe619')
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
