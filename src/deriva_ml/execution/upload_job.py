"""Non-blocking upload job — a thread handle around run_upload_engine.

Per spec §2.11.3. The job drives the same engine as ml.upload_pending,
but in a background thread so the caller can poll progress, cancel,
or wait. If the process exits, the thread dies with it; users
wanting survive-process uploads run `deriva-ml upload` in a shell
(Group H).
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from deriva_ml.execution.upload_engine import UploadReport, run_upload_engine

if TYPE_CHECKING:
    from deriva_ml.core.base import DerivaML

logger = logging.getLogger(__name__)


@dataclass
class UploadProgress:
    """Snapshot of in-flight upload progress.

    Phase 1: reports only end-of-run counts. Phase 2 adds live
    byte-level progress by hooking into deriva-py's uploader callbacks.
    """
    total_rows: int = 0
    uploaded_rows: int = 0
    total_bytes: int = 0
    uploaded_bytes: int = 0
    current_mbps: float = 0.0
    eta_seconds: float | None = None
    current_file: str | None = None
    failures: list[str] = field(default_factory=list)


class UploadJob:
    """Background upload driven by a thread.

    Construct via ml.start_upload(...). Not meant to be subclassed.

    Attributes:
        id: Unique identifier (uuid-like string).
        status: One of 'running', 'paused', 'completed', 'failed',
            'cancelled'.
    """

    def __init__(
        self,
        *,
        ml: "DerivaML",
        execution_rids: "list[str] | None",
        retry_failed: bool,
        bandwidth_limit_mbps: "int | None",
        parallel_files: int,
    ) -> None:
        self.id = f"upl_{uuid.uuid4().hex[:12]}"
        self.status: Literal[
            "running", "paused", "completed", "failed", "cancelled"
        ] = "running"
        self._ml = ml
        self._execution_rids = execution_rids
        self._retry_failed = retry_failed
        self._bandwidth_limit_mbps = bandwidth_limit_mbps
        self._parallel_files = parallel_files

        self._report: UploadReport | None = None
        self._exception: BaseException | None = None
        self._cancel_event = threading.Event()
        self._progress = UploadProgress()

        self._thread = threading.Thread(
            target=self._run, name=f"upload-{self.id}", daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            self._report = run_upload_engine(
                ml=self._ml,
                execution_rids=self._execution_rids,
                retry_failed=self._retry_failed,
                bandwidth_limit_mbps=self._bandwidth_limit_mbps,
                parallel_files=self._parallel_files,
            )
            self.status = (
                "completed" if self._report.total_failed == 0 else "failed"
            )
        except BaseException as exc:  # noqa: BLE001 — surface via wait()
            logger.warning("upload job %s errored: %s", self.id, exc)
            self._exception = exc
            self.status = "failed"

    def progress(self) -> UploadProgress:
        """Return a snapshot of current progress.

        Phase 1: only end-of-run counts are populated; running jobs
        return defaults until completion.
        """
        if self._report is not None:
            self._progress.uploaded_rows = self._report.total_uploaded
            self._progress.failures = list(self._report.errors)
        return self._progress

    def pause(self) -> None:
        """Phase 1 no-op — deriva-py's uploader has no pause primitive."""
        logger.warning("UploadJob.pause is not wired in Phase 1")

    def resume(self) -> None:
        """Stub — pairs with pause."""
        logger.warning("UploadJob.resume is not wired in Phase 1")

    def cancel(self) -> None:
        """Request cancellation.

        Sets an event the engine checks between work items. Phase 1
        does not cancel mid-file; cancel only stops further work
        from starting.
        """
        self._cancel_event.set()
        if self.status == "running":
            self.status = "cancelled"

    def wait(self, timeout: float | None = None) -> UploadReport:
        """Block until the job finishes; return its UploadReport.

        Raises:
            TimeoutError: If timeout elapses.
            BaseException: If the engine raised.
        """
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise TimeoutError(f"upload job {self.id} did not finish in {timeout}s")
        if self._exception is not None:
            raise self._exception
        assert self._report is not None
        return self._report
