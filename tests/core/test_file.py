from pathlib import Path
from random import randbytes, choice

import pytest

from deriva_ml import ExecutionConfiguration, FileSpec, MLVocab


class TestFile:
    @pytest.fixture(scope="class", autouse=True)
    def test_file_table_setup(self, test_ml_catalog, test_files, tmp_path):
        self.ml_instance = test_ml_catalog
        self.workflow = self.ml_instance.create_workflow("Test Workflow")
        self.execution = self.ml_instance.create_execution(
            ExecutionConfiguration(workflow=self.workflow, description="Test Execution")
        )
        self.test_dir = Path(tmp_path) / "test_dir"
        d1 = self.test_dir / "d1"
        d2 = self.test_dir / "d2"
        self.file_count = 0
        for d in [self.test_dir, d1, d2]:
            for i in range(5):
                with open(d / f"file{i}.{choice(['txt', 'jpg'])}", "w") as f:
                    f.write(randbytes(256).decode())
                    self.file_count += 1
        self.ml_instance.add_term(MLVocab.workflow_type, "File Test Workflow", description="Test workflow")
        yield self.test_dir
        # Cleanup

    def test_file_table(self, test_ml_catalog):
        with self.execution.execute() as exe:
            filespecs = FileSpec.create_filespecs(self.test_dir, "Test Directory")
            files = list(exe.add_files(filespecs))
            assert len(files) == self.file_count
            for r in files:
                assert self.ml_instance.retrieve_rid(r)

      def test_file_types(self, test_ml_catalog, test_files):
            with self.execution.execute() as exe:
                filespecs = FileSpec.create_filespecs(test_files, "Test Directory")
                files = list(exe.add_files(filespecs))
                assert len(files) == self.file_count
                for r in files:
                    assert self.ml_instance.read_file(r.rid)
