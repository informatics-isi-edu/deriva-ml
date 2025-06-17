import shutil
import string
from pathlib import Path
from random import choice, choices

import pytest

from deriva_ml import DerivaMLInvalidTerm, ExecutionConfiguration, FileSpec, MLVocab


class TestFile:
    @pytest.fixture(scope="function", autouse=True)
    def test_file_table_setup(self, test_ml_catalog, shared_tmp_path):
        def random_string(length: int) -> str:
            alphabet = string.ascii_letters + string.digits
            return "".join(choices(alphabet, k=length))

        self.ml_instance = test_ml_catalog
        self.ml_instance.add_term(MLVocab.workflow_type, "File Test Workflow", description="Workflow for testing files")
        self.workflow = self.ml_instance.create_workflow(name="Test Workflow", workflow_type="File Test Workflow")
        self.execution = self.ml_instance.create_execution(
            ExecutionConfiguration(workflow=self.workflow, description="Test Execution")
        )
        self.test_dir = Path(shared_tmp_path) / "test_dir"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        d1 = self.test_dir / "d1"
        d1.mkdir(parents=True, exist_ok=True)
        d2 = self.test_dir / "d2"
        d2.mkdir(parents=True, exist_ok=True)

        self.file_count = 0
        for d in [self.test_dir, d1, d2]:
            for i in range(5):
                self.file_count += 1
                with open(d / f"file{i}.{choice(['txt', 'jpeg'])}", "w") as f:
                    f.write(random_string(10))
        self.ml_instance.add_term(MLVocab.workflow_type, "File Test Workflow", description="Test workflow")
        yield self
        shutil.rmtree(self.test_dir)
        # Cleanup

    def test_file_table_bad_type(self, test_file_table_setup):
        ml_instance = test_file_table_setup
        with pytest.raises(DerivaMLInvalidTerm):
            filespec = [
                FileSpec(
                    url="tag://test_dir/file1.txt", description="Test file", md5="123", length=0, file_types=["foo"]
                )
            ]
            ml_instance.execution.add_files(filespec)
            filespec = [
                FileSpec(
                    url="tag://test_dir/file1.txt", description="Test file", md5="123", length=0, file_types=["foo"]
                )
            ]
            ml_instance.execution.add_files(filespec)

    def test_create_filespecs(self, test_file_table_setup):
        ml_instance = test_file_table_setup.ml_instance
        test_dir = test_file_table_setup.test_dir
        execution = test_file_table_setup.execution

        def use_extension(filename: Path) -> [str]:
            return [filename.suffix.lstrip(".")]

        with execution.execute() as exe:
            filespecs = list(FileSpec.create_filespecs(test_dir, "Test Directory"))
            assert len(filespecs) == test_file_table_setup.file_count
            assert filespecs[0].file_types == ["File"]
            filespecs = FileSpec.create_filespecs(test_dir, "Test Directory", file_types=["txt"])
            assert all([set(filespec.file_types) == {"txt", "File"} for filespec in filespecs])
            filespecs = FileSpec.create_filespecs(test_dir, "Test Directory", file_types=use_extension)
            assert all([set(filespec.file_types) < {"txt", "jpeg", "File"} for filespec in filespecs])

    def test_add_files(self, test_file_table_setup):
        ml_instance = test_file_table_setup.ml_instance
        test_dir = test_file_table_setup.test_dir
        execution = test_file_table_setup.execution

        def use_extension(filename: Path) -> [str]:
            return [filename.suffix.lstrip(".")]

        ml_instance.add_term(MLVocab.asset_type, "jpeg", description="A Image file")
        ml_instance.add_term(MLVocab.asset_type, "txt", description="A Text file")
        with execution.execute() as exe:
            filespecs = FileSpec.create_filespecs(test_dir, "Test Directory", file_types=use_extension)

            file_dataset = exe.add_files(filespecs)
            assert file_dataset in [ds["RID"] for ds in ml_instance.find_datasets()]
            ds = ml_instance.list_dataset_members(file_dataset)
            assert len(ds["File"]) == 5
            assert len(ds["Dataset"]) == 2
            for subdir in ml_instance.list_dataset_children(file_dataset):
                ds = ml_instance.list_dataset_members(subdir)
                assert len(ds["File"]) == 5

    def test_list_files(self, test_file_table_setup):
        ml_instance = test_file_table_setup.ml_instance
        test_dir = test_file_table_setup.test_dir
        execution = test_file_table_setup.execution

        jpeg_cnt = 0
        txt_cnt = 0

        def use_extension(filename: Path) -> list[str]:
            nonlocal jpeg_cnt, txt_cnt
            ext = filename.suffix.lstrip(".")
            if ext == "jpeg":
                jpeg_cnt += 1
            else:
                txt_cnt += 1
            return [ext]

        ml_instance.add_term(MLVocab.asset_type, "jpeg", description="A Image file")
        ml_instance.add_term(MLVocab.asset_type, "txt", description="A Text file")

        with execution.execute() as exe:
            filespecs = FileSpec.create_filespecs(test_dir, "Test Directory", file_types=use_extension)
            file_dataset = exe.add_files(filespecs)

        files = ml_instance.list_files()
        assert len(files) == 15
        files = ml_instance.list_files(file_types=["jpeg"])
        assert len(files) == jpeg_cnt
        files = ml_instance.list_files(file_types=["txt"])
        assert len(files) == txt_cnt
        files = ml_instance.list_files(file_types=["jpeg", "txt"])
        assert len(files) == jpeg_cnt + txt_cnt
