"""Unit tests for ``dataset_loader`` validating the dtlpy-converters >= 3.0.12
migration (``ConvertersUploader`` removed; per-format classes with synchronous
``.convert(...)`` used directly).

These tests intentionally have no behavioral expectations beyond what the
pre-migration code did, so they double as a regression suite for the
migration described in the change set.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

from tests import _stubs  # noqa: F401  -- side effects: stub modules + sys.path

import dataset_loader  # noqa: E402
import dtlpy as dl  # noqa: E402  -- the stub installed by _stubs


REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_LOADER_SRC = (REPO_ROOT / "dataset_loader.py").read_text(encoding="utf-8")


def _make_hf_items():
    """Return a small, deterministic fake HuggingFace dataset iterable."""
    return [
        {
            "image_id": 1,
            "image": mock.MagicMock(name="pil_image_1"),
            "width": 640,
            "height": 480,
            "objects": {
                "id": [10, 11],
                "area": [100.0, 200.0],
                "bbox": [[0, 0, 10, 10], [5, 5, 20, 20]],
                "category": [1, 2],
            },
        },
        {
            "image_id": 2,
            "image": mock.MagicMock(name="pil_image_2"),
            "width": 800,
            "height": 600,
            "objects": {
                "id": [20],
                "area": [300.0],
                "bbox": [[1, 2, 3, 4]],
                "category": [3],
            },
        },
    ]


class ImportAndModuleShapeTests(unittest.TestCase):
    """Static / source-level checks that the migration is in place."""

    def test_module_imports_cleanly(self):
        # If the import had failed, the test module itself would have failed
        # to load. This assertion documents the requirement explicitly.
        self.assertIsNotNone(dataset_loader)

    def test_coco_to_dataloop_symbol_is_bound(self):
        from dtlpyconverters.coco import CocoToDataloop as StubCls

        self.assertIs(dataset_loader.CocoToDataloop, StubCls)

    def test_no_legacy_uploader_symbol(self):
        self.assertFalse(
            hasattr(dataset_loader, "ConvertersUploader"),
            "ConvertersUploader should not be re-exported by dataset_loader",
        )

    def test_source_has_no_legacy_references(self):
        forbidden = [
            "ConvertersUploader",
            "ConvertersDownloader",
            "dtlpyconverters.uploaders",
            "dtlpyconverters.downloaders",
            "_get_event_loop",
            "run_until_complete",
            "nest_asyncio",
        ]
        for needle in forbidden:
            self.assertNotIn(
                needle,
                DATASET_LOADER_SRC,
                f"Legacy reference {needle!r} still present in dataset_loader.py",
            )

    def test_source_uses_synchronous_convert(self):
        self.assertIn("CocoToDataloop(", DATASET_LOADER_SRC)
        self.assertIn(".convert(", DATASET_LOADER_SRC)


class DatasetHFInitTests(unittest.TestCase):
    """Constructor behavior preserved minus the removed ``self.converter``."""

    def test_init_does_not_create_converter_attribute(self):
        instance = dataset_loader.DatasetHF()
        self.assertFalse(
            hasattr(instance, "converter"),
            "self.converter should be gone after the migration",
        )

    def test_init_preserves_coco_format_template(self):
        instance = dataset_loader.DatasetHF()
        self.assertEqual(instance.coco_format["images"], [])
        self.assertEqual(instance.coco_format["annotations"], [])
        self.assertEqual(
            instance.coco_format["categories"],
            [
                {"id": 1, "name": "Chair"},
                {"id": 2, "name": "Sofa"},
                {"id": 3, "name": "Table"},
            ],
        )

    def test_init_sets_logger(self):
        instance = dataset_loader.DatasetHF()
        self.assertIs(instance.logger, dataset_loader.logger)


class UploadDatasetCocoTests(unittest.TestCase):
    """End-to-end behavior of ``upload_dataset_coco`` using the new sync API."""

    def setUp(self):
        self.instance = dataset_loader.DatasetHF()
        self.dataset = mock.MagicMock(name="dl.Dataset")
        self.source = "https://huggingface.co/datasets/some_org/some_ds"
        self.hf_items = _make_hf_items()

        # Capture state observable from inside the (mocked) converter.
        self.captured = {}

        convert_mock = mock.MagicMock(name="CocoToDataloop.convert")

        def fake_ctor(**kwargs):
            self.captured["ctor_kwargs"] = kwargs
            ann_dir = kwargs["input_annotations_path"]
            items_dir = kwargs["input_items_path"]
            self.captured["ann_dir_existed"] = os.path.isdir(ann_dir)
            self.captured["items_dir_existed"] = os.path.isdir(items_dir)
            json_path = os.path.join(ann_dir, "coco_format.json")
            self.captured["coco_json_existed"] = os.path.isfile(json_path)
            with open(json_path, "r", encoding="utf-8") as fp:
                self.captured["coco_json"] = json.load(fp)
            self.captured["items_dir_listing"] = sorted(os.listdir(items_dir))
            instance = mock.MagicMock(name="CocoToDataloop_instance")
            instance.convert = convert_mock
            return instance

        self.ctor_patch = mock.patch.object(
            dataset_loader, "CocoToDataloop", side_effect=fake_ctor
        )
        self.ctor_mock = self.ctor_patch.start()
        self.addCleanup(self.ctor_patch.stop)
        self.convert_mock = convert_mock

        # Mock the HF loader to return our fake items.
        self.load_dataset_patch = mock.patch.object(
            dataset_loader, "load_dataset", return_value=self.hf_items
        )
        self.load_dataset_mock = self.load_dataset_patch.start()
        self.addCleanup(self.load_dataset_patch.stop)

    def _run(self):
        self.instance.upload_dataset_coco(self.dataset, self.source)

    def test_constructor_called_with_expected_kwargs(self):
        self._run()
        self.ctor_mock.assert_called_once()
        kwargs = self.captured["ctor_kwargs"]
        self.assertIs(kwargs["dataset"], self.dataset)
        self.assertTrue(kwargs["upload_items"])
        self.assertTrue(os.path.isabs(kwargs["input_items_path"]))
        self.assertTrue(os.path.isabs(kwargs["input_annotations_path"]))
        # The ctor must NOT receive call-site kwargs that belong on .convert(...)
        self.assertNotIn("coco_json_filename", kwargs)
        self.assertNotIn("annotation_options", kwargs)
        self.assertNotIn("to_polygon", kwargs)

    def test_convert_called_with_expected_kwargs(self):
        self._run()
        self.convert_mock.assert_called_once()
        call = self.convert_mock.call_args
        # All converter args must be passed as kwargs (matches the migration).
        self.assertEqual(call.args, ())
        self.assertEqual(
            call.kwargs,
            {
                "coco_json_filename": "coco_format.json",
                "annotation_options": [dl.AnnotationType.BOX],
            },
        )
        # Behavior parity: do NOT pass to_polygon (old call site didn't).
        self.assertNotIn("to_polygon", call.kwargs)

    def test_no_event_loop_is_constructed(self):
        with mock.patch.object(
            asyncio, "new_event_loop", side_effect=AssertionError("event loop created")
        ), mock.patch.object(
            asyncio, "set_event_loop", side_effect=AssertionError("set_event_loop called")
        ), mock.patch.object(
            asyncio,
            "get_event_loop",
            side_effect=AssertionError("get_event_loop called"),
        ):
            self._run()

    def test_load_dataset_called_with_huggingface_location(self):
        self._run()
        self.load_dataset_mock.assert_called_once_with(
            "some_org/some_ds", split="train"
        )

    def test_coco_json_built_from_hf_items(self):
        self._run()
        coco = self.captured["coco_json"]

        self.assertEqual(
            coco["categories"],
            [
                {"id": 1, "name": "Chair"},
                {"id": 2, "name": "Sofa"},
                {"id": 3, "name": "Table"},
            ],
        )
        self.assertEqual(
            coco["images"],
            [
                {"id": 1, "width": 640, "height": 480, "file_name": "image_1.jpg"},
                {"id": 2, "width": 800, "height": 600, "file_name": "image_2.jpg"},
            ],
        )
        self.assertEqual(
            coco["annotations"],
            [
                {"id": 10, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100.0},
                {"id": 11, "image_id": 1, "category_id": 2, "bbox": [5, 5, 20, 20], "area": 200.0},
                {"id": 20, "image_id": 2, "category_id": 3, "bbox": [1, 2, 3, 4], "area": 300.0},
            ],
        )

    def test_images_saved_before_converter_runs(self):
        self._run()
        # When CocoToDataloop is constructed the JSON + image saves must already
        # be on disk (the function blocks synchronously through save_images_in_threads
        # and json.dump *before* invoking the converter).
        self.assertTrue(self.captured["ann_dir_existed"])
        self.assertTrue(self.captured["items_dir_existed"])
        self.assertTrue(self.captured["coco_json_existed"])
        items_dir = self.captured["ctor_kwargs"]["input_items_path"]
        for item in self.hf_items:
            item["image"].save.assert_called_once_with(
                os.path.join(items_dir, f"image_{item['image_id']}.jpg")
            )

    def test_function_is_synchronous(self):
        """Sanity check: upload_dataset_coco must remain a regular def."""
        self.assertFalse(
            inspect.iscoroutinefunction(self.instance.upload_dataset_coco),
            "upload_dataset_coco should not be a coroutine function",
        )


if __name__ == "__main__":
    unittest.main()
