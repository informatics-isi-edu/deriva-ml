from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import FrameType
from typing import Optional

try:  # optional imports — used only when running in notebooks
    from IPython.core.getipython import get_ipython  # type: ignore
except Exception:  # pragma: no cover - optional

    def get_ipython():  # type: ignore
        return None


try:  # optional — only available when inside a kernel
    from ipykernel.connect import get_connection_file as _get_kernel_connection
except Exception:  # pragma: no cover - optional
    _get_kernel_connection = None  # type: ignore

try:  # optional — only available when jupyter-server is installed
    from jupyter_server.serverapp import list_running_servers as _list_running_servers  # type: ignore
except Exception:  # pragma: no cover - optional
    _list_running_servers = None  # type: ignore

try:  # optional — HTTP call to Jupyter server API
    import requests  # type: ignore
    from requests import RequestException  # type: ignore
except Exception:  # pragma: no cover - optional
    requests = None  # type: ignore
    RequestException = Exception  # type: ignore


def _norm(p: str) -> str:
    """Normalize a path string using pathlib.

    - Expands ~
    - Resolves to absolute path
    - Returns a string path
    Note: We no longer apply os.path.normcase explicitly; pathlib's resolve
    provides a consistent absolute path. This should be sufficient for our
    use-cases across platforms.
    """
    try:
        return str(Path(p).expanduser().resolve())
    except Exception:
        # As a very last resort, return the original string
        return p


# Treat certain pseudo filenames from IPython/Jupyter as user code so they
# can be selected as the calling location when appropriate (e.g., in REPL).
def _is_pseudo_user_filename(filename: str) -> bool:
    """Return True if filename looks like an IPython/Jupyter pseudo file.

    Examples that should return True:
    - "<ipython-input-7-abcdef>"
    - "<jupyter-input-3-123456>"
    - "<ipykernel_12345>"

    Other pseudo files like "<stdin>" or "<string>" should return False here
    so they can be treated by the generic pseudo-file handling below.
    """
    if not (filename.startswith("<") and filename.endswith(">")):
        return False
    lower = filename.lower()
    return lower.startswith("<ipython-input-") or lower.startswith("<jupyter-input-") or lower.startswith("<ipykernel_")


# Names that frequently represent “system/tooling” frames rather than user code
_SYSTEM_MODULE_PREFIXES = (
    # pytest + plugin stack
    "pytest",
    "_pytest",
    "pluggy",
    # IPython/Jupyter stack
    "IPython",
    "traitlets",
    "tornado",
    "jupyter_client",
    "jupyter_core",
    "ipykernel",
    # IDE/debugger stack (PyCharm)
    "pydevd",
    "_pydevd_bundle",
    "_pydev_bundle",
    # Python internals
    "importlib",
    "runpy",
    "inspect",
    "traceback",
    "contextlib",
    "asyncio",
    "threading",
)


# --- Helpers focused on determining the current "python model" (file) ---


def _top_user_frame() -> Optional[FrameType]:
    """Return the outermost (top-level) non-tooling frame from the current stack.

    This function traverses the call stack from the current execution point
    back to the entry point, filtering out known tooling (pytest, IDE helpers,
    Jupyter internals) and returns the highest-level frame that belongs to
    user code.
    """
    tooling_prefixes = _SYSTEM_MODULE_PREFIXES
    tooling_filename_parts = (
        "pydevconsole.py",  # PyCharm REPL console
        "/pydev/",  # PyCharm helpers path segment
        "/_pydevd_bundle/",
        "/_pydev_bundle/",
        "_pytest",
        "/pycharm/",
    )

    f = inspect.currentframe()
    last_user_frame = None

    if f is not None:
        f = f.f_back  # Skip the _top_user_frame itself

    while f is not None:
        filename = f.f_code.co_filename or ""
        mod_name = f.f_globals.get("__name__", "") or ""

        # 1. Treat IPython cell as user code
        if _is_pseudo_user_filename(filename):
            last_user_frame = f
            f = f.f_back
            continue

        # 2. Skip other pseudo files like <stdin>, <string>, etc., unless __main__
        if filename.startswith("<") and filename.endswith(">") and mod_name not in ("__main__", "__mp_main__"):
            f = f.f_back
            continue

        # 3. Skip known tooling frames by module prefix
        if any(mod_name == p or mod_name.startswith(p + ".") for p in tooling_prefixes):
            f = f.f_back
            continue

        # 4. Skip known tooling frames by filename patterns
        if any(part in filename for part in tooling_filename_parts):
            f = f.f_back
            continue

        # 5. Skip frames that belong to this helper module (find_caller.py)
        try:
            cur = str(Path(filename).resolve())
            this = str(Path(__file__).resolve())
            if cur == this:
                f = f.f_back
                continue
        except Exception:
            pass

        # If it passed all filters, it is a user frame.
        # We record it and keep going back to find an even "higher" one.
        last_user_frame = f
        f = f.f_back

    return last_user_frame


def _get_notebook_path() -> Optional[str]:
    """Best‑effort to obtain the current Jupyter notebook path.

    Returns absolute path string if discoverable, else None.
    """
    ip = get_ipython()
    if ip is None:
        return None

    # Must be running inside a kernel with a connection file
    if _get_kernel_connection is None:
        return None
    try:
        connection_file = Path(_get_kernel_connection()).name  # type: ignore[operator]
    except Exception:
        return None

    # Need jupyter-server and requests to query sessions
    if _list_running_servers is None or requests is None:
        return None

    kernel_id = connection_file.split("-", 1)[1].split(".")[0]
    try:
        servers = list(_list_running_servers())  # type: ignore[func-returns-value]
    except Exception:
        return None

    for server in servers:
        try:
            token = server.get("token", "")
            headers = {"Authorization": f"token {token}"} if token else {}
            url = server["url"] + "api/sessions"
            resp = requests.get(url, headers=headers, timeout=3)  # type: ignore[attr-defined]
            resp.raise_for_status()
            for sess in resp.json():
                if sess.get("kernel", {}).get("id") == kernel_id:
                    rel = sess.get("notebook", {}).get("path")
                    if rel:
                        root_dir = server.get("root_dir") or server.get("notebook_dir")
                        if root_dir:
                            return str(Path(root_dir) / rel)
        except RequestException:
            continue
        except Exception:
            continue
    return None


def _get_calling_module() -> str:
    """Return the relevant source filename for the current execution context.

    Behavior:
    1) In Jupyter Notebook/Hub: returns the .ipynb file path.
    2) In a script: returns the script filename.
    3) In pytest or any REPL (PyCharm or regular): returns the filename that
       contains the function currently executing (nearest user frame).
    4) If executing code from an installed package in a venv, still returns that
       package module file (we do NOT exclude site-packages).
    """
    # 1) Jupyter notebook
    nb = _get_notebook_path()
    if nb:
        return str(Path(nb))

    # 2) If running as a script (python myscript.py), prefer __main__.__file__ or argv[0]
    def _is_tooling_script_path(p: str) -> bool:
        # Normalize path to forward slashes and lowercase for robust substring checks
        pn = p.replace("\\", "/").casefold()
        # Detect common IDE/console helper scripts
        tooling_markers = (
            "pydevconsole.py",
            "/pydev/",
            "/_pydevd_bundle/",
            "/_pydev_bundle/",
        )
        return any(m in pn for m in tooling_markers)

    f = _top_user_frame()
    if f is not None:
        return _norm(f.f_code.co_filename)
    main_mod = sys.modules.get("__main__")
    main_file = getattr(main_mod, "__file__", None)

    if isinstance(main_file, str) and main_file:
        if not _is_tooling_script_path(main_file):
            return _norm(main_file)
    if sys.argv and sys.argv[0] and sys.argv[0] != "-c":
        if not _is_tooling_script_path(sys.argv[0]):
            return _norm(sys.argv[0])

    # 3) Pytest/REPL/IDE: use nearest user frame
    f = _top_user_frame()

    if f is not None:
        return _norm(f.f_code.co_filename)

    # Fallback: <stdin> or current working directory marker
    return str(Path.cwd() / "REPL")
