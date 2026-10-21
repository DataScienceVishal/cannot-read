"""Load a benchmark's scorer from the bytes its pin was checked against.

trail-scorer-audit's first loader imported its scorer through `importlib`,
which hands the file to a SourceFileLoader, and that loader runs a cached
`.pyc` instead of the source whenever the pyc header's mtime and size match the
source file's. The digest then described the file on disk and not the code that
ran. A file stored by `fetch` is named by its digest with no `.py` suffix, so
`spec_from_file_location` finds no loader for it, but a SourceFileLoader given
the path by hand would still look for a cached `.pyc`. Compiling the bytes
`verified_bytes` hashed avoids that, and avoids a second read of the file that
nothing checks.

Only a scorer that is a single file fits this. BFCL's and HumanEval's are
packages, and running a package from pinned bytes needs an importer that
nothing here provides.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

from cannotread.fetch import Pin, stored_at, verified_bytes


def load_scorer(pin: Pin, into: Path) -> ModuleType:
    source = verified_bytes(pin, into)
    # Named after the digest, so the module says which bytes it came from, and
    # never `__main__`, so a scorer's command-line block does not run and parse
    # sys.argv.
    module = ModuleType(f"scorer_{pin.sha256[:12]}")
    module.__file__ = str(stored_at(pin, into))
    # In sys.modules before it runs, as an import would put it, because a
    # dataclass with string annotations looks its module up there, and so does
    # pickle.
    sys.modules[module.__name__] = module
    # dont_inherit, or the `from __future__ import annotations` at the top of
    # this file would reach the scorer and turn its annotations into strings.
    exec(compile(source, module.__file__, "exec", dont_inherit=True), module.__dict__)
    return module
