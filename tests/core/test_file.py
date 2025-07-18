import string
from pathlib import Path
from random import choice, choices
from tempfile import TemporaryDirectory

import pytest
from deriva.core.datapath import DataPathException

from deriva_ml import DerivaML, DerivaMLInvalidTerm, ExecutionConfiguration, FileSpec, MLVocab

FILE_COUNT = 5


class TestFiles:
    def __init__(self, test_ml):
        def random_string(length: int) -> str:
            alphabet = string.ascii_letters + string.digits
            return "".join(choices(alphabet, k=length))

        self.tmp_dir = Path(TemporaryDirectory().name)
        self.test_dir = self.tmp_dir / "test_dir"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        d1 = self.test_dir / "d1"
        d1.mkdir(parents=True, exist_ok=True)
        d2 = self.test_dir / "d2"
        d2.mkdir(parents=True, exist_ok=True)

        self.file_count = 0
        for d in [self.test_dir, d1, d2]:
            for i in range(FILE_COUNT):
                self.file_count += 1
                with (d / f"file{i}.{choice(['txt', 'jpeg'])}").open("w") as f:
                    f.write(random_string(10))

        self.ml_instance = DerivaML(hostname=test_ml.hostname, catalog_id=test_ml.catalog_id, working_dir=self.tmp_dir)
        self.ml_instance.add_term(MLVocab.workflow_type, "File Test Workflow", description="Workflow for testing files")
        self.workflow = self.ml_instance.create_workflow(name="Test Workflow", workflow_type="File Test Workflow")
        self.execution = self.ml_instance.create_execution(
            ExecutionConfiguration(workflow=self.workflow, description="Test Execution")
        )
        self.ml_instance.add_term(MLVocab.workflow_type, "File Test Workflow", description="Test workflow")

        self.ml_instance.add_term(MLVocab.asset_type, "jpeg", description="A Image file")
        self.ml_instance.add_term(MLVocab.asset_type, "txt", description="A Text file")

    def clean_up(self):
        print("Cleaning up test files....")
        try:
            self.ml_instance.pathBuilder.schemas[self.ml_instance.ml_schema].tables["File"].delete()
        except DataPathException as e:
            print(type(e))


@pytest.fixture(scope="function")
def file_table_setup(deriva_catalog):
    print("Setting up file_table_catalog....")
    test_files = TestFiles(deriva_catalog)
    yield test_files
    test_files.clean_up()

    # Cleanup


class TestFile:
    def test_file_table_bad_type(self, file_table_setup):
        ml_instance = file_table_setup
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

    def test_create_filespecs(self, file_table_setup):
        test_dir = file_table_setup.test_dir
        execution = file_table_setup.execution

        def use_extension(filename: Path) -> list[str]:
            return [filename.suffix.lstrip(".")]

        with execution.execute() as _exe:
            filespecs = list(FileSpec.create_filespecs(test_dir, "Test Directory"))
            assert len(filespecs) == file_table_setup.file_count
            assert filespecs[0].file_types == ["File"]
            filespecs = FileSpec.create_filespecs(test_dir, "Test Directory", file_types=["txt"])
            assert all([set(filespec.file_types) == {"txt", "File"} for filespec in filespecs])
            filespecs = FileSpec.create_filespecs(test_dir, "Test Directory", file_types=use_extension)
            assert all([set(filespec.file_types) < {"txt", "jpeg", "File"} for filespec in filespecs])

    def test_add_files(self, file_table_setup):
        ml_instance = file_table_setup.ml_instance
        test_dir = file_table_setup.test_dir
        execution = file_table_setup.execution

        def use_extension(filename: Path) -> list[str]:
            return [filename.suffix.lstrip(".")]

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

    def test_list_files(self, file_table_setup):
        ml_instance = file_table_setup.ml_instance
        test_dir = file_table_setup.test_dir
        execution = file_table_setup.execution

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

        with execution.execute() as exe:
            filespecs = FileSpec.create_filespecs(test_dir, "Test Directory", file_types=use_extension)
            _file_dataset = exe.add_files(filespecs)

        files = ml_instance.list_files(file_types=["jpeg"])
        assert len(files) == jpeg_cnt
        files = ml_instance.list_files(file_types=["txt"])
        assert len(files) == txt_cnt
        files = ml_instance.list_files(file_types=["jpeg", "txt"])
        assert len(files) == jpeg_cnt + txt_cnt

    def test_files_datasets(self, file_table_setup):
        ml_instance = file_table_setup.ml_instance
        test_dir = file_table_setup.test_dir
        execution = file_table_setup.execution

        ml_instance.add_term(MLVocab.asset_type, "jpeg", description="A Image file")
        ml_instance.add_term(MLVocab.asset_type, "txt", description="A Text file")

        with execution.execute() as exe:
            filespecs = FileSpec.create_filespecs(
                test_dir, "Test Directory", file_types=lambda f: [f.suffix.lstrip(".")]
            )
            file_dataset = exe.add_files(filespecs)

        assert len(ml_instance.list_dataset_children(file_dataset)) == 2
        assert len(ml_instance.list_dataset_members(file_dataset)["File"]) == FILE_COUNT
        for subdir in ml_instance.list_dataset_children(file_dataset):
            assert len(ml_instance.list_dataset_members(subdir)["File"]) == FILE_COUNT

    def test_file_spec_read_write(self, tmp_path):
        """Test reading and writing FileSpecs to JSONL."""
        # Create test files
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("content 1")
        file2.write_text("content 2")

        # Create FileSpecs
        specs = list(FileSpec.create_filespecs(tmp_path, "Test files"))
        assert len(specs) == 2

        # Write to JSONL
        jsonl_file = tmp_path / "specs.jsonl"
        with jsonl_file.open("w") as f:
            for spec in specs:
                f.write(spec.model_dump_json() + "\n")

        # Read back
        read_specs = list(FileSpec.read_filespec(jsonl_file))
        assert len(read_specs) == 2

        # Compare
        for original, read in zip(specs, read_specs):
            assert read.url == original.url
            assert read.description == original.description
            assert read.md5 == original.md5
            assert read.length == original.length
