"""Pre-upload metadata pre-processor for deriva-py's GenericUploader.

Part of the Bug C fix — see
``docs/superpowers/specs/2026-04-22-bug-c-asset-metadata-design.md``.
"""
from __future__ import annotations

from deriva.transfer.upload.processors import BaseProcessor


class NullSentinelProcessor(BaseProcessor):
    """Translate the ``"__NULL__"`` sentinel to Python ``None``.

    Runs before deriva-py's ``interpolateDict`` expansion. Mutates
    the in-flight ``self.metadata`` dict in place: any value equal
    to ``"__NULL__"`` (see
    :data:`deriva_ml.dataset.upload.NULL_SENTINEL`) is replaced
    with ``None``. deriva-py then drops None-valued keys, causing
    the resulting catalog insert to send SQL ``NULL`` for those
    columns.

    Not part of the end-user API — configured automatically by
    :func:`deriva_ml.dataset.upload.asset_table_upload_spec`.

    Note: if a user's legitimate metadata value equals the sentinel
    string, it will be corrupted to NULL. Known constraint; chosen
    sentinel is unlikely to collide naturally.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # BaseProcessor stores kwargs on self.kwargs; the metadata
        # dict is passed via kwargs["metadata"] by
        # deriva-py's _execute_processors.
        self.metadata = kwargs.get("metadata", {})

    def process(self):
        from deriva_ml.dataset.upload import NULL_SENTINEL
        for k, v in list(self.metadata.items()):
            if v == NULL_SENTINEL:
                self.metadata[k] = None
