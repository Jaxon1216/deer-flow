"""Failed local overwrites must retain the previous complete file."""

from __future__ import annotations

import builtins
import errno
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from deerflow.sandbox.local.local_sandbox import LocalSandbox, PathMapping


@pytest.fixture(params=["write_file", "update_file"])
def method(request):
    return request.param


def write(sandbox, method, path, content=b"new complete content\n"):
    return getattr(sandbox, method)(path, content.decode("utf-8") if method == "write_file" else content)


@pytest.fixture(params=[False, True], ids=["generic", "mapped"])
def destination(request, tmp_path):
    target = tmp_path / "existing.txt"
    if request.param:
        return LocalSandbox("atomic", [PathMapping("/mnt/data", str(tmp_path))]), "/mnt/data/existing.txt", target
    return LocalSandbox("atomic"), str(target), target


@pytest.mark.parametrize("stage", ["write", "flush", "replace"])
@pytest.mark.parametrize("existing", [True, False], ids=["overwrite", "new"])
def test_failure_preserves_destination_and_removes_temporary(tmp_path, monkeypatch, method, destination, stage, existing):
    sandbox, path, target = destination
    if existing:
        target.write_bytes(b"previous content\n")
    old_mode = target.stat().st_mode if existing else None
    real_open, real_fdopen = builtins.open, os.fdopen

    class FailingWriter:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            try:
                if exc_type is None:
                    self.flush()
            finally:
                self.handle.close()

        def write(self, content):
            if stage == "write":
                self.handle.write(content[:2])
                self.handle.flush()
                raise OSError(errno.ENOSPC, "injected write failure")
            return self.handle.write(content)

        def flush(self):
            self.handle.flush()
            if stage == "flush":
                raise OSError(errno.ENOSPC, "injected flush failure")

        def __getattr__(self, name):
            return getattr(self.handle, name)

    def intercept_open(filename, mode="r", *args, **kwargs):
        handle = real_open(filename, mode, *args, **kwargs)
        return FailingWriter(handle) if "w" in mode and Path(filename).parent == target.parent else handle

    def fail_replace(*args, **kwargs):
        raise OSError(errno.EBUSY, "injected replace failure")

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", intercept_open)
        patch.setattr(os, "fdopen", lambda *args, **kwargs: FailingWriter(real_fdopen(*args, **kwargs)))
        if stage == "replace":
            patch.setattr(os, "replace", fail_replace)
        with pytest.raises(OSError, match=f"injected {stage} failure") as exc:
            write(sandbox, method, path)
        assert exc.value.filename == path

    if existing:
        assert target.read_bytes() == b"previous content\n"
        assert target.stat().st_mode == old_mode
        assert list(tmp_path.iterdir()) == [target]
    else:
        assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX file-size limit")
def test_real_file_size_failure_keeps_previous_content(tmp_path, method):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json, resource, signal, sys
from pathlib import Path
from deerflow.sandbox.local.local_sandbox import LocalSandbox
path = Path(sys.argv[1])
path.write_bytes(b'previous content\\n')
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
old_limit = resource.getrlimit(resource.RLIMIT_FSIZE)
resource.setrlimit(resource.RLIMIT_FSIZE, (1024, old_limit[1]))
try:
    content = 'N' * 4096 if sys.argv[2] == 'write_file' else b'N' * 4096
    getattr(LocalSandbox('limited'), sys.argv[2])(str(path), content)
except OSError as exc:
    error = exc.errno
else:
    error = None
finally:
    resource.setrlimit(resource.RLIMIT_FSIZE, old_limit)
print(json.dumps({'errno': error, 'content': path.read_text(), 'files': sorted(p.name for p in path.parent.iterdir())}))
""",
            str(tmp_path / "existing.txt"),
            method,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(result.stdout) == {"errno": errno.EFBIG, "content": "previous content\n", "files": ["existing.txt"]}


def test_success_preserves_symlink_target(tmp_path, method, destination):
    sandbox, path, target = destination
    target.write_bytes(b"old")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    link_path = "/mnt/data/link.txt" if path.startswith("/mnt/data/") else str(link)
    write(sandbox, method, link_path)
    assert link.is_symlink()
    assert target.read_bytes() == b"new complete content\n"
    assert set(tmp_path.iterdir()) == {target, link}


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
@pytest.mark.parametrize("mode", [0o600, 0o640, 0o755])
def test_success_preserves_existing_mode_and_owner(tmp_path, method, mode):
    target = tmp_path / "file.txt"
    target.write_bytes(b"old")
    target.chmod(mode)
    before = target.stat()
    write(LocalSandbox("atomic"), method, str(target))
    assert target.read_bytes() == b"new complete content\n"
    after = target.stat()
    assert stat.S_IMODE(after.st_mode) == mode
    assert (after.st_uid, after.st_gid) == (before.st_uid, before.st_gid)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission enforcement")
def test_read_only_file_is_still_refused(tmp_path, method):
    if os.geteuid() == 0:
        pytest.skip("root can write mode-0400 files")
    target = tmp_path / "file.txt"
    target.write_bytes(b"old")
    target.chmod(0o400)
    try:
        with pytest.raises(PermissionError):
            write(LocalSandbox("atomic"), method, str(target))
        assert target.read_bytes() == b"old"
        assert list(tmp_path.iterdir()) == [target]
    finally:
        target.chmod(0o600)


def test_read_only_mapping_is_still_refused(tmp_path, method):
    target = tmp_path / "file.txt"
    target.write_bytes(b"old")
    sandbox = LocalSandbox("atomic", [PathMapping("/mnt/data", str(tmp_path), read_only=True)])
    with pytest.raises(OSError) as exc:
        write(sandbox, method, "/mnt/data/file.txt")
    assert exc.value.errno == errno.EROFS
    assert target.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [target]


def test_hard_linked_destination_replaces_only_the_named_file(tmp_path, method):
    target = tmp_path / "file.txt"
    target.write_bytes(b"old")
    alias = tmp_path / "alias.txt"
    try:
        os.link(target, alias)
    except OSError as exc:
        pytest.skip(f"hard links unavailable: {exc}")
    write(LocalSandbox("atomic"), method, str(target))
    assert target.read_bytes() == b"new complete content\n"
    assert alias.read_bytes() == b"old"
    assert target.stat().st_ino != alias.stat().st_ino


def test_new_file_uses_normal_umask_permissions(tmp_path, method):
    target, reference = tmp_path / "created.txt", tmp_path / "reference.txt"
    reference.write_bytes(b"reference")
    write(LocalSandbox("atomic"), method, str(target))
    assert target.read_bytes() == b"new complete content\n"
    assert stat.S_IMODE(target.stat().st_mode) == stat.S_IMODE(reference.stat().st_mode)


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode/umask")
@pytest.mark.parametrize("operation", ["write_file", "update_file", "publish"])
@pytest.mark.parametrize("mode", [0o600, 0o640])
def test_existing_file_staging_stays_private_during_write_and_flush(tmp_path, operation, mode):
    # Setting umask in a child cannot affect files created by sibling tests.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json, os, stat, sys
from pathlib import Path
from deerflow.sandbox.local.local_sandbox import LocalSandbox
from deerflow.utils.atomic_write import publish_bytes_atomically
path = Path(sys.argv[1])
mode = int(sys.argv[3])
path.write_bytes(b'old private contents')
path.chmod(mode)
os.umask(0o022)
real_fdopen = os.fdopen
observed = []
class Observer:
    def __init__(self, handle): self.handle = handle
    def __enter__(self): return self
    def __exit__(self, *args): return self.handle.__exit__(*args)
    def write(self, content):
        observed.append(['write', stat.S_IMODE(os.fstat(self.handle.fileno()).st_mode)])
        return self.handle.write(content)
    def flush(self):
        observed.append(['flush', stat.S_IMODE(os.fstat(self.handle.fileno()).st_mode)])
        return self.handle.flush()
    def __getattr__(self, name): return getattr(self.handle, name)
os.fdopen = lambda *args, **kwargs: Observer(real_fdopen(*args, **kwargs))
if sys.argv[2] == 'publish':
    publish_bytes_atomically(path, b'new private contents')
else:
    content = 'new private contents' if sys.argv[2] == 'write_file' else b'new private contents'
    getattr(LocalSandbox('private'), sys.argv[2])(str(path), content)
print(json.dumps({'stages': observed, 'mode': stat.S_IMODE(path.stat().st_mode), 'content': path.read_text()}))
""",
            str(tmp_path / "private.txt"),
            operation,
            str(mode),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(result.stdout) == {"stages": [["write", 0o600], ["flush", 0o600]], "mode": mode, "content": "new private contents"}


@pytest.mark.parametrize("existing", [False, True], ids=["new", "overwrite"])
def test_long_legal_filename_can_be_written(tmp_path, method, existing):
    target = tmp_path / ("x" * 230 + ".txt")
    try:
        target.write_bytes(b"old")
    except OSError as exc:
        if exc.errno == errno.ENAMETOOLONG:
            pytest.skip("host path length limit cannot represent this filename")
        raise
    if not existing:
        target.unlink()
    write(LocalSandbox("atomic"), method, str(target))
    assert target.read_bytes() == b"new complete content\n"
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.skipif(os.name == "nt", reason="POSIX ownership")
def test_ownership_copy_failure_keeps_previous_file(tmp_path, monkeypatch, method):
    target = tmp_path / "existing.txt"
    target.write_bytes(b"old")
    original = target.stat()
    real_fstat = os.fstat

    def different_group(fd):
        snapshot = real_fstat(fd)
        if snapshot.st_ino == original.st_ino:
            values = list(snapshot)
            values[5] = snapshot.st_gid + 1
            return os.stat_result(values)
        return snapshot

    def fail_owner(*args):
        raise PermissionError(errno.EPERM, "ownership copy denied")

    with monkeypatch.context() as patch:
        patch.setattr(os, "fstat", different_group)
        patch.setattr(os, "fchown", fail_owner)
        with pytest.raises(PermissionError, match="ownership copy denied"):
            write(LocalSandbox("atomic"), method, str(target))
    assert target.read_bytes() == b"old"
    assert target.stat().st_ino == original.st_ino
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory permission enforcement")
def test_unwritable_parent_is_refused_without_fallback(tmp_path, method):
    if os.geteuid() == 0:
        pytest.skip("root can write mode-0500 directories")
    parent = tmp_path / "readonly"
    parent.mkdir()
    target = parent / "existing.txt"
    target.write_bytes(b"old")
    parent.chmod(0o500)
    try:
        with pytest.raises(PermissionError):
            write(LocalSandbox("atomic"), method, str(target))
        assert target.read_bytes() == b"old"
        assert list(parent.iterdir()) == [target]
    finally:
        parent.chmod(0o700)


def test_append_keeps_existing_inode_and_adds_content(tmp_path):
    target = tmp_path / "file.txt"
    target.write_bytes(b"old\r\n")
    inode = target.stat().st_ino
    LocalSandbox("atomic").write_file(str(target), "new\n", append=True)
    assert target.read_bytes() == b"old\r\nnew\n"
    assert target.stat().st_ino == inode


def test_text_write_preserves_virtual_content_round_trip(tmp_path):
    sandbox = LocalSandbox("atomic", [PathMapping("/mnt/data", str(tmp_path))])
    sandbox.write_file("/mnt/data/settings.txt", "path=/mnt/data/output.txt\r\n")
    assert str(tmp_path).replace("\\", "/") in (tmp_path / "settings.txt").read_text()
    assert sandbox.read_file("/mnt/data/settings.txt") == "path=/mnt/data/output.txt\r\n"
