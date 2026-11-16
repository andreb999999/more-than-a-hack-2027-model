"""A small on-disk cache for the expensive, deterministic steps (parsing, feature tables, DFL tables).

An entry's key hashes the source code of the modules that produce it plus any parameters, so editing
the code or the configuration never reuses a stale entry. Deleting the cache folder is always safe:
everything in it is rebuilt from the raw data.
"""

import hashlib
import inspect
import logging
import pickle
from importlib.metadata import version
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

log = logging.getLogger(__name__)

# Entries built under other library versions are never reused.
_LIBRARY_VERSIONS = "|".join(f"{name}={version(name)}" for name in ("numpy", "pandas", "scipy", "scikit-learn", "numba"))


def fingerprint(*parts: Any) -> str:
    """A short hex digest of the library versions, modules (their source), paths (size and mtime) and plain values (repr).

    Parameters
    ----------
    *parts : Any
        Modules (hashed by their source code, so editing a module invalidates its entries), paths
        (hashed by file name, size and integer mtime, not content) and any other values (hashed by
        ``repr``), in order.

    Returns
    -------
    str
        The first 14 hex characters of the SHA-1 digest.
    """
    digest = hashlib.sha1(_LIBRARY_VERSIONS.encode())
    for part in parts:
        if isinstance(part, ModuleType):
            digest.update(inspect.getsource(part).encode())
        elif isinstance(part, Path):
            stat = part.stat()
            digest.update(f"{part.name}|{stat.st_size}|{int(stat.st_mtime)}".encode())
        else:
            digest.update(repr(part).encode())
    return digest.hexdigest()[:14]


def cached(folder: Path, name: str, key: str, build: Callable[[], Any]) -> Any:
    """Load `name` from the cache, or build it, store it and return it.

    Parameters
    ----------
    folder : Path
        Cache folder; created when an entry is first stored.
    name : str
        Entry name; the file is ``<folder>/<name>_<key>.pkl``.
    key : str
        Fingerprint of everything the entry depends on (`fingerprint`).
    build : callable
        Zero-argument function that computes the value on a cache miss.

    Returns
    -------
    Any
        The unpickled entry, or the freshly built value (written atomically via a ``.tmp`` file).
    """
    path = Path(folder) / f"{name}_{key}.pkl"
    if path.exists():
        log.debug("cache hit %s", path.name)
        return pickle.loads(path.read_bytes())
    log.info("building %s", name)
    value = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(value, protocol=4))
    tmp.replace(path)
    return value
