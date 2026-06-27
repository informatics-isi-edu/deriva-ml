import string
from pathlib import Path
from random import choice, choices
from tempfile import TemporaryDirectory

import pytest
from deriva.core.datapath import DataPathException

from deriva_ml import DerivaML, DerivaMLInvalidTerm, FileSpec, MLVocab
from deriva_ml.execution import ExecutionConfiguration

try:
    from icecream import ic
except ImportError:
    ic = lambda *a, **kw: None


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
            self.ml_instance.pathBuilder().schemas[self.ml_instance.ml_schema].tables["File"].delete()
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
            ic(file_dataset)
            assert file_dataset.dataset_rid in [ds.dataset_rid for ds in ml_instance.find_datasets()]
            ds = file_dataset.list_dataset_members()
            assert len(ds["File"]) == 5
            assert len(ds["Dataset"]) == 2
            for subdir in file_dataset.list_dataset_children():
                ds = subdir.list_dataset_members()
                assert len(ds["File"]) == 5

    def test_add_files_tags_datasets_as_directory(self, file_table_setup):
        """Every dataset add_files creates carries the built-in ``Directory``
        type (force-included like ``File``), marking it as an auto-created
        directory-structure dataset — distinguishing these byproduct datasets
        from curated ones. A caller-supplied dataset_type is additive."""
        test_dir = file_table_setup.test_dir
        execution = file_table_setup.execution

        with execution.execute() as exe:
            filespecs = FileSpec.create_filespecs(test_dir, "Test Directory")
            file_dataset = exe.add_files(filespecs, dataset_types="Complete")

        # The returned (top) dataset and every nested child must be tagged
        # Directory + File, plus the caller's "Complete".
        assert "Directory" in file_dataset.dataset_types
        assert "File" in file_dataset.dataset_types
        assert "Complete" in file_dataset.dataset_types
        for subdir in file_dataset.list_dataset_children():
            assert "Directory" in subdir.dataset_types

    def test_add_files_directory_datasets_record_path(self, file_table_setup):
        """Each directory dataset gets a Directory_Dataset row with its path
        relative to the ingest root; the ingest root stores '.'.

        New contract (approved design decision, backward-compat waived):
        - The ROOT dataset description defaults to the ingest-root directory's
          basename (``ingest_root.name``), NOT the caller's description.
        - Non-root / child datasets keep the caller's description.
        """
        ml_instance = file_table_setup.ml_instance
        test_dir = file_table_setup.test_dir
        execution = file_table_setup.execution

        with execution.execute() as exe:
            filespecs = FileSpec.create_filespecs(test_dir, "Test Directory")
            file_dataset = exe.add_files(filespecs, description="Ingest run")

        # Root description is the ingest-root directory's basename ("test_dir").
        assert file_dataset.description == test_dir.name
        # Child datasets keep the caller's description.
        for child in file_dataset.list_dataset_children():
            assert child.description == "Ingest run"

        # Directory_Dataset.Path holds the relative folder for each dataset.
        pb = ml_instance.pathBuilder()
        rows = list(pb.schemas[ml_instance.ml_schema].tables["Directory_Dataset"].entities().fetch())
        path_by_dataset = {r["Dataset"]: r["Path"] for r in rows}

        assert path_by_dataset[file_dataset.dataset_rid] == "."
        child_paths = {path_by_dataset[c.dataset_rid] for c in file_dataset.list_dataset_children()}
        assert child_paths == {"d1", "d2"}

    def test_add_files_returns_single_root_for_forest(self, file_table_setup):
        """add_files always returns ONE dataset that transitively contains every
        file, even when the source dirs form a forest (sibling branches whose
        common ancestor holds no files of its own).

        Tree: ``base/a/x/f.txt`` and ``base/b/y/g.txt`` — two leaf dirs at the
        same depth, common ancestor ``base`` (and ``base/a``, ``base/b``) holding
        no files. The old loop nested purely on decreasing path-depth, so the two
        same-depth leaves never nested into a shared parent: it returned one leaf
        and orphaned the other. The fix builds nesting from real path containment
        and synthesizes the intermediate directory datasets, so the returned root
        reaches all files with no orphaned directory datasets.
        """
        ml_instance = file_table_setup.ml_instance
        execution = file_table_setup.execution

        base = file_table_setup.tmp_dir / "forest"
        (base / "a" / "x").mkdir(parents=True, exist_ok=True)
        (base / "b" / "y").mkdir(parents=True, exist_ok=True)
        (base / "a" / "x" / "f.txt").write_text("f")
        (base / "b" / "y" / "g.txt").write_text("g")

        with execution.execute() as exe:
            filespecs = FileSpec.create_filespecs(base, "Forest ingest")
            root = exe.add_files(filespecs, description="Forest")

        # Collect every File RID reachable from the returned root, walking the
        # full nested-dataset subtree.
        def all_file_rids(ds):
            rids = {m["RID"] for m in ds.list_dataset_members().get("File", [])}
            for child in ds.list_dataset_children(recurse=True):
                rids |= {m["RID"] for m in child.list_dataset_members().get("File", [])}
            return rids

        reachable_files = all_file_rids(root)
        # Both files (one per branch) must be reachable from the single returned root.
        assert len(reachable_files) == 2, (
            f"returned root reaches {len(reachable_files)} files; both branches' files "
            f"must be transitively contained — no orphaned directory datasets"
        )

        # And no Directory dataset created this run is unreachable from the root.
        reachable_ds = {root.dataset_rid} | {c.dataset_rid for c in root.list_dataset_children(recurse=True)}
        directory_ds = [d.dataset_rid for d in ml_instance.find_datasets() if "Directory" in d.dataset_types]
        orphans = [rid for rid in directory_ds if rid not in reachable_ds]
        assert not orphans, f"orphaned directory datasets not reachable from root: {orphans}"

    def test_add_files_chunked_streaming_matches_single_batch(self, file_table_setup):
        """add_files streams a generator in chunks of ``chunk_size`` and the
        result is identical to a single-batch insert.

        A small chunk_size (< file count) forces several insert batches over
        the 15-file / 3-directory tree. The resulting dataset must have the
        same nested structure, the same per-directory member counts, and the
        same Directory + File tags as the default single-batch path — chunking
        is an internal performance detail, never a behavior change. The input
        is a *generator* (FileSpec.create_filespecs), consumed exactly once.
        """
        ml_instance = file_table_setup.ml_instance
        test_dir = file_table_setup.test_dir
        execution = file_table_setup.execution

        with execution.execute() as exe:
            # create_filespecs returns a generator — passed straight through,
            # never pre-materialized by the caller.
            filespecs = FileSpec.create_filespecs(test_dir, "Chunked Directory")
            file_dataset = exe.add_files(filespecs, dataset_types="Complete", chunk_size=2)

        # Same top-level shape as test_add_files: 5 files + 2 child datasets.
        assert file_dataset.dataset_rid in [ds.dataset_rid for ds in ml_instance.find_datasets()]
        members = file_dataset.list_dataset_members()
        assert len(members["File"]) == 5
        assert len(members["Dataset"]) == 2

        # Same tags as the single-batch path.
        assert "Directory" in file_dataset.dataset_types
        assert "File" in file_dataset.dataset_types
        assert "Complete" in file_dataset.dataset_types

        # Each child directory dataset holds its 5 files and is tagged Directory.
        children = file_dataset.list_dataset_children()
        assert len(children) == 2
        for subdir in children:
            sub_members = subdir.list_dataset_members()
            assert len(sub_members["File"]) == 5
            assert "Directory" in subdir.dataset_types

    def test_dataset_source_directory_and_is_directory_accessor(self, file_table_setup):
        """Dataset.source_directory returns the directory dataset's relative folder and
        is_directory is True for those datasets; both reflect the
        Directory_Dataset row."""
        test_dir = file_table_setup.test_dir
        execution = file_table_setup.execution

        with execution.execute() as exe:
            filespecs = FileSpec.create_filespecs(test_dir, "Test Directory")
            root = exe.add_files(filespecs, description="Ingest run")
            # A non-directory dataset (created directly, not via add_files) has no
            # Directory_Dataset row: source_directory is None and is_directory is False.
            plain = exe.create_dataset(dataset_types="Complete", description="not a dir")

        assert root.source_directory == "."
        assert root.is_directory is True
        child_paths = {child.source_directory for child in root.list_dataset_children()}
        assert child_paths == {"d1", "d2"}
        assert all(child.is_directory for child in root.list_dataset_children())

        assert plain.source_directory is None
        assert plain.is_directory is False

    def test_add_files_links_dataset_as_input(self, file_table_setup):
        """add_files records ONE Dataset_Execution input edge (the root source
        dataset), not per-file File_Execution Input rows. The execution is
        input-complete via the dataset edge; producer + membership intact.

        Contract:
        (a) ZERO File_Execution Input rows were written by add_files.
        (b) EXACTLY ONE Dataset_Execution input edge points at the root dataset.
        (c) The root dataset's current version is produced by this execution
            (Dataset_Version.Execution — written by create_dataset).
        (d) The root dataset transitively contains the registered files.
        """
        ml_instance = file_table_setup.ml_instance
        test_dir = file_table_setup.test_dir
        execution = file_table_setup.execution

        with execution.execute() as exe:
            filespecs = list(FileSpec.create_filespecs(test_dir, "Referenced files"))
            root = exe.add_files(filespecs)

        pb = ml_instance.pathBuilder()

        # (a) ZERO File_Execution Input rows were written by add_files — including
        # no unknown-provenance sentinel. With _skip_input_check=True, provenance
        # enforcement is suppressed during create_dataset; add_files declares the
        # input itself via Dataset_Execution. No per-file edges, no sentinel.
        fe = pb.schemas[ml_instance.ml_schema].File_Execution
        fe_input_rows = [
            r for r in fe.filter(fe.Execution == exe.execution_rid).entities().fetch() if r.get("Asset_Role") == "Input"
        ]
        assert fe_input_rows == [], (
            f"add_files must write NO File_Execution Input rows (no per-file edges, "
            f"no unknown-provenance sentinel); got {len(fe_input_rows)}"
        )

        # (b) EXACTLY ONE Dataset_Execution input edge: the root dataset.
        de = pb.schemas[ml_instance.ml_schema].Dataset_Execution
        de_rows = list(de.filter(de.Execution == exe.execution_rid).entities().fetch())
        assert len(de_rows) == 1, f"expected exactly one Dataset_Execution input edge, got {len(de_rows)}"
        assert de_rows[0]["Dataset"] == root.dataset_rid

        # (c) Producer edge intact: the root dataset's current version is produced
        #     by this execution (Dataset_Version.Execution).
        assert ml_instance._producer_of_dataset(root.dataset_rid) == exe.execution_rid

        # (d) Membership intact: the root dataset still transitively contains the files.
        members = root.list_dataset_members(recurse=True)
        assert members.get("File"), "root dataset should contain File members"

    def test_add_files_has_no_role_parameter(self, file_table_setup):
        """add_files must NOT expose a role parameter — role is not a user
        choice (provenance contract: role is derived from context, and a File
        reference is intrinsically an input).

        ``Execution.add_files`` is ``@validate_call``-wrapped, so an unknown
        kwarg is rejected as a pydantic ``ValidationError`` (a subclass-free
        ``TypeError`` would come from a plain function); accept either.
        """
        from pydantic import ValidationError

        execution = file_table_setup.execution
        with execution.execute() as exe:
            filespecs = list(FileSpec.create_filespecs(file_table_setup.test_dir, "x"))
            with pytest.raises((TypeError, ValidationError)):
                exe.add_files(filespecs, asset_role="Output")  # role param must not exist

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

        assert len(file_dataset.list_dataset_children()) == 2
        assert len(file_dataset.list_dataset_members()["File"]) == FILE_COUNT
        for subdir in file_dataset.list_dataset_children():
            assert len(subdir.list_dataset_members()["File"]) == FILE_COUNT

    def test_add_files_empty_raises_clear_error(self, file_table_setup):
        """add_files with no files raises a clear DerivaMLException, not an
        obscure ValueError from os.path.commonpath([])."""
        from deriva_ml.core.exceptions import DerivaMLException

        execution = file_table_setup.execution
        with execution.execute() as exe:
            with pytest.raises(DerivaMLException):
                exe.add_files([], description="empty")

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


class TestCreateFilespecsLength:
    """Regression tests for ``FileSpec.create_filespecs`` length reporting.

    Earlier versions of ``create_spec`` (the inner closure) used the
    outer ``path`` for ``length`` instead of the per-file
    ``file_path``. In single-file mode this happened to work (``path``
    and ``file_path`` referenced the same file); in directory mode
    every emitted FileSpec reported the *directory's* ``stat().st_size``
    rather than the file's. Asset uploads silently recorded wrong
    sizes — the manifest layer recorded a constant ``~64`` or
    ``~4096`` for every file under a directory walk.

    Pure-Python tests; no catalog required.
    """

    def test_single_file_length_matches_file_size(self, tmp_path):
        """``create_filespecs`` on a single file reports that file's size."""
        f = tmp_path / "lone.txt"
        payload = b"hello world\n"  # 12 bytes
        f.write_bytes(payload)

        specs = list(FileSpec.create_filespecs(f, "single"))
        assert len(specs) == 1
        assert specs[0].length == len(payload)

    def test_directory_walk_reports_per_file_length(self, tmp_path):
        """Each FileSpec in a directory walk reports its own file's size.

        The original bug: every file's ``length`` was the directory's
        stat size, not the file's. This test creates three files of
        distinct sizes (10 / 100 / 1000 bytes) and asserts each
        emitted spec carries the correct one — they must be three
        distinct values, not three copies of the directory's size.
        """
        sizes = {"small.txt": 10, "medium.txt": 100, "large.txt": 1000}
        for name, n in sizes.items():
            (tmp_path / name).write_bytes(b"x" * n)

        specs = {Path(s.url).name: s for s in FileSpec.create_filespecs(tmp_path, "walk")}
        assert set(specs) == set(sizes), f"Expected one spec per file; got {set(specs)} vs {set(sizes)}"
        for name, expected in sizes.items():
            assert specs[name].length == expected, (
                f"FileSpec for {name!r} reported length {specs[name].length}, "
                f"expected {expected}. The pre-fix bug would have reported "
                f"the directory's stat().st_size here, identical across all "
                f"three files."
            )

        # Strongest pin: the three lengths must be the file sizes
        # we actually wrote, not all-equal-to-the-directory-size.
        observed_lengths = {s.length for s in specs.values()}
        assert observed_lengths == set(sizes.values()), (
            f"All filespecs got the same length {observed_lengths} — "
            f"this is the closure-shadowing bug. Each file's length "
            f"must reflect its own stat().st_size."
        )

    def test_nested_directory_walk_reports_per_file_length(self, tmp_path):
        """Recursive walk also reports per-file lengths.

        ``create_filespecs`` uses ``rglob("*")``; a regression that
        re-introduced the closure shadowing would emit the same wrong
        length whether the file lived directly under ``path`` or in
        a nested subdir. Test both layouts in one walk.
        """
        (tmp_path / "top.txt").write_bytes(b"a" * 7)
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.txt").write_bytes(b"b" * 77)
        (sub / "deep").mkdir()
        (sub / "deep" / "deepfile.txt").write_bytes(b"c" * 777)

        specs = {Path(s.url).name: s for s in FileSpec.create_filespecs(tmp_path, "deep")}
        assert specs["top.txt"].length == 7
        assert specs["nested.txt"].length == 77
        assert specs["deepfile.txt"].length == 777
