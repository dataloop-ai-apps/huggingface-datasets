"""Stub heavy third-party dependencies so ``dataset_loader`` can be imported
in a hermetic test environment (no dtlpy / datasets / dtlpy-converters
installed).

This module is imported for its side effects by every test module BEFORE
``dataset_loader`` is imported. Importing it more than once is safe.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest import mock


def _ensure_repo_on_syspath() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def _install_stub_modules() -> None:
    if "dtlpy" not in sys.modules:
        dtlpy = types.ModuleType("dtlpy")

        class _AnnotationType:
            BOX = "box"
            POLYGON = "polygon"
            SEGMENTATION = "segmentation"

        class _BaseServiceRunner:
            pass

        class _CallbackEvent:
            ITEMS_UPLOAD = "items_upload"

        class _NotFound(Exception):
            pass

        class _Exceptions:
            NotFound = _NotFound

        class _ClientApi:
            def __init__(self) -> None:
                self.add_callback = mock.MagicMock()

        class _FeatureEntityType:
            ITEM = "item"

        class _Dataset:
            pass

        dtlpy.AnnotationType = _AnnotationType
        dtlpy.BaseServiceRunner = _BaseServiceRunner
        dtlpy.CallbackEvent = _CallbackEvent
        dtlpy.exceptions = _Exceptions
        dtlpy.client_api = _ClientApi()
        dtlpy.FeatureEntityType = _FeatureEntityType
        dtlpy.Dataset = _Dataset
        sys.modules["dtlpy"] = dtlpy

    if "datasets" not in sys.modules:
        datasets_mod = types.ModuleType("datasets")
        datasets_mod.load_dataset = mock.MagicMock(name="load_dataset")
        sys.modules["datasets"] = datasets_mod

    if "requests" not in sys.modules:
        requests_mod = types.ModuleType("requests")
        requests_mod.get = mock.MagicMock(name="requests.get")

        class _RequestException(Exception):
            pass

        exceptions_mod = types.ModuleType("requests.exceptions")
        exceptions_mod.RequestException = _RequestException
        requests_mod.exceptions = exceptions_mod
        sys.modules["requests"] = requests_mod
        sys.modules["requests.exceptions"] = exceptions_mod

    if "dtlpyconverters" not in sys.modules:
        converters_pkg = types.ModuleType("dtlpyconverters")
        coco_mod = types.ModuleType("dtlpyconverters.coco")

        class _CocoToDataloop:
            def __init__(self, *args, **kwargs) -> None:
                self.args = args
                self.kwargs = kwargs

            def convert(self, *args, **kwargs) -> None:
                return None

        coco_mod.CocoToDataloop = _CocoToDataloop
        converters_pkg.coco = coco_mod
        sys.modules["dtlpyconverters"] = converters_pkg
        sys.modules["dtlpyconverters.coco"] = coco_mod


_ensure_repo_on_syspath()
_install_stub_modules()
