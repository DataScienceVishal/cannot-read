"""Refuse a predictor that goes looking for the input behind its `View`.

`View` has no field for the input, so a predictor that wanted it would have to
fetch it some other way: open a benchmark file an adapter has downloaded, list
a directory to find one, start a process that reads it, or open a socket. While
`guarded()` is active, a Python audit hook (PEP 578) turns each of those into a
`BlindnessViolation`. `predict` runs every predictor inside it.

Every refusal is appended to the record of each guard that is open, and
`guarded()` raises again on the way out if its record is not empty. Without
that, a predictor could catch the error, or open a guard of its own and catch
that, and carry on with its answers as if nothing happened.

Two kinds of file access are let through. The standard library imports some
modules the first time a function is called (`Counter.most_common` imports
`heapq`), and formatting a traceback or a warning makes linecache read the
source file. Refusing the first would refuse a predictor that does nothing but
count training labels. So an open of a `.py`, `.pyc` or `.so` file, or a
directory listing, is allowed when the code that asked for it directly is the
import system, or tokenize, which linecache reads source through. Only the
direct caller counts: a module whose own top-level code reads a data file is
refused like anything else.

It is meant to catch a mistake in this repository's own predictors, and PEP 578
has a section on why audit hooks are no sandbox. Code that sets out to get
round this one can. It does not see a predictor that:
- reads content a module already holds in memory, for example a global an
  adapter filled after fetching, because reading a variable raises no event.
  Nothing checks that adapters keep fetched content out of globals;
- reads through a C extension that opens files itself, such as DuckDB, or
  through ctypes once a library is loaded;
- calls `os.stat`, which is not audited, or reads from a file descriptor
  opened before the guard went on;
- edits this module, or keeps data in a `.py` file and imports it;
- starts a thread and reads from it, since the guard belongs to the thread
  that opened it. That is also why another thread's file access is not blamed
  on the predictor.
"""

from __future__ import annotations

import os
import sys
import threading
import tokenize
from collections.abc import Iterator
from contextlib import contextmanager

# Audit event names from the CPython audit events table. "open" covers
# builtins.open, io.open, os.open and pathlib's readers, and os.scandir covers
# os.walk and Path.glob.
_REFUSED = frozenset(
    {
        "open",
        "os.listdir",
        "os.scandir",
        "glob.glob",
        "sqlite3.connect",
        "subprocess.Popen",
        "os.system",
        "os.exec",
        "os.fork",
        "os.forkpty",
        "os.posix_spawn",
        "os.spawn",
        "socket.__new__",
        "socket.connect",
        "socket.getaddrinfo",
        "urllib.Request",
    }
)

_CODE_READERS = {
    "<frozen importlib._bootstrap>",
    "<frozen importlib._bootstrap_external>",
    tokenize.__file__,
}
_CODE_SUFFIXES = (".py", ".pyc", ".so")


class BlindnessViolation(RuntimeError):
    pass


_local = threading.local()


def _records() -> list[list[BlindnessViolation]]:
    if not hasattr(_local, "records"):
        _local.records = []
    return _local.records


def is_guarded() -> bool:
    return bool(_records())


def _reading_code(event: str, args: tuple[object, ...]) -> bool:
    if event not in {"open", "os.listdir", "os.scandir"}:
        return False
    # Frame 0 is this function and 1 is `_hook`, so 2 is the Python code that
    # made the call being audited.
    if sys._getframe(2).f_code.co_filename not in _CODE_READERS:
        return False
    if event != "open":
        return True
    path = args[0]
    return isinstance(path, str | bytes) and os.fsdecode(path).endswith(_CODE_SUFFIXES)


def _hook(event: str, args: tuple[object, ...]) -> None:
    if event not in _REFUSED or not is_guarded() or _reading_code(event, args):
        return
    violation = BlindnessViolation(f"{event} {args[0] if args else ''!r}")
    for record in _records():
        record.append(violation)
    raise violation


sys.addaudithook(_hook)


@contextmanager
def guarded(who: str) -> Iterator[None]:
    record: list[BlindnessViolation] = []
    _records().append(record)
    # Writing a .pyc opens a temporary file and then a bare descriptor, which
    # the hook cannot tell from a predictor writing anything else. Imports
    # still read existing .pyc files. This is process-wide, so another
    # thread's imports also skip writing bytecode while a predictor runs.
    writes_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        _records().pop()
        sys.dont_write_bytecode = writes_bytecode
        # Raised from `finally` so that it replaces whatever the predictor
        # raised after the refusal. That may hide an unrelated error, but the
        # refusal is the one that has to reach the report.
        if record:
            raise BlindnessViolation(f"{who} tried {record[0]}") from record[0]
