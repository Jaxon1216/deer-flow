"""Atomic byte publication with normal file-creation permissions."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from uuid import uuid4


def publish_bytes_atomically(path: Path, content: bytes, *, existing_stat: os.stat_result | None = None) -> None:
    """Replace *path* with complete bytes, preserving its existing mode.

    New files use ``0666 & ~umask``, like ``Path.write_bytes``, so captures
    remain readable through sandbox mounts under the usual ``0022`` umask.
    A hidden, exclusively created sibling keeps partial writes away from the
    final path. Failure before replacement leaves the previous file intact.

    Callers that checked an existing file through a writable descriptor may
    supply its stat snapshot to preserve its POSIX owner/group as well as mode.
    An ownership-copy failure aborts publication. This does not copy ACLs or
    extended attributes and does not preserve the replaced inode.
    """
    try:
        mode = stat.S_IMODE((existing_stat or path.stat()).st_mode)
    except FileNotFoundError:
        mode = None

    # The destination may already use NAME_MAX bytes. Keep staging independent
    # of its basename so every legal destination still has a legal sibling.
    temporary = path.with_name(f".deerflow-{uuid4().hex}.tmp")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    # Let the kernel apply umask; reading it via os.umask would mutate
    # process-global state while other threads may be creating files.
    # Replacement bytes must stay private while staging, including when the
    # original file has a different group. Restore its owner/mode only once the
    # content is complete; brand-new files retain normal umask permissions.
    fd = os.open(temporary, flags, 0o600 if mode is not None else 0o666)
    try:
        handle = os.fdopen(fd, "wb")
        fd = None  # The file object now owns the descriptor.
        with handle:
            handle.write(content)
            handle.flush()
            if existing_stat is not None and os.name != "nt":
                staged_stat = os.fstat(handle.fileno())
                if (staged_stat.st_uid, staged_stat.st_gid) != (existing_stat.st_uid, existing_stat.st_gid):
                    os.fchown(handle.fileno(), existing_stat.st_uid, existing_stat.st_gid)
            if mode is not None:
                if os.name == "nt":
                    os.chmod(temporary, mode)
                else:
                    os.fchmod(handle.fileno(), mode)
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if fd is not None:
            os.close(fd)
        temporary.unlink(missing_ok=True)
        raise
