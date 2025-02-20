from derivaml_test import TestDerivaML

from deriva_ml import MLVocab, Workflow, ExecutionConfiguration
from tempfile import TemporaryDirectory
from random import random
from deriva_ml.demo_catalog import (
    reset_demo_catalog,
)


class TestDownload(TestDerivaML):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_download(self):
        pass
