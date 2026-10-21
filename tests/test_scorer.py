from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from cannotread import scorer as scorer_module
from cannotread.fetch import NotFetched, Pin, PinMismatch, fetch, stored_at, verified_bytes
from cannotread.scorer import load_scorer

REVISION = "0123abc"

# Stands in for evaluate-v2.0.py: a function to call, a command-line block
# behind the usual guard, and a top-level line that raises if one byte of it
# changes. `count` is only there for its annotation.
SCORER = b"""\
def exact(n):
    return n + 1

def count(x: int) -> int:
    return x

if 0:
    raise RuntimeError("the edited copy ran")

if __name__ == "__main__":
    raise SystemExit("the command-line block ran")
"""


def _upstream(tmp_path: Path, content: bytes = SCORER) -> Pin:
    source = tmp_path / "upstream" / REVISION / "evaluate.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(content)
    return Pin(source.as_uri(), REVISION, hashlib.sha256(content).hexdigest())


def _fetched(tmp_path: Path, content: bytes = SCORER) -> tuple[Pin, Path]:
    pin = _upstream(tmp_path, content)
    into = tmp_path / "fetched"
    fetch(pin, into)
    return pin, into


def test_the_scorer_runs_from_its_pinned_bytes(tmp_path: Path) -> None:
    pin, into = _fetched(tmp_path)
    scorer = load_scorer(pin, into)
    assert scorer.exact(1) == 2
    assert scorer.__name__ == f"scorer_{pin.sha256[:12]}"
    assert scorer.__file__ == str(stored_at(pin, into))


def test_one_changed_byte_is_refused_before_anything_runs(tmp_path: Path) -> None:
    pin, into = _fetched(tmp_path)
    edited = SCORER.replace(b"if 0:", b"if 1:")
    assert sum(a != b for a, b in zip(SCORER, edited, strict=True)) == 1
    stored_at(pin, into).write_bytes(edited)
    # Run first and checked after, the edited copy would raise RuntimeError here.
    with pytest.raises(PinMismatch):
        load_scorer(pin, into)


def test_the_file_is_not_read_again_after_the_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pin, into = _fetched(tmp_path)

    def check_then_edit(pin: Pin, into: Path) -> bytes:
        checked = verified_bytes(pin, into)
        stored_at(pin, into).write_bytes(SCORER.replace(b"n + 1", b"n + 2"))
        return checked

    monkeypatch.setattr(scorer_module, "verified_bytes", check_then_edit)
    assert load_scorer(pin, into).exact(1) == 2


def test_an_unfetched_scorer_is_not_downloaded_on_load(tmp_path: Path) -> None:
    pin = _upstream(tmp_path)
    with pytest.raises(NotFetched):
        load_scorer(pin, tmp_path / "fetched")


def test_the_command_line_block_does_not_run(tmp_path: Path) -> None:
    pin, into = _fetched(tmp_path)
    # SystemExit would escape here if the module were named __main__.
    load_scorer(pin, into)


def test_the_loaders_future_import_does_not_reach_the_scorer(tmp_path: Path) -> None:
    pin, into = _fetched(tmp_path)
    # scorer.py has `from __future__ import annotations`. Inherited, this would
    # be the string "int".
    assert load_scorer(pin, into).count.__annotations__["x"] is int


def test_a_scorer_can_define_a_dataclass_with_string_annotations(tmp_path: Path) -> None:
    pin, into = _fetched(
        tmp_path,
        b"from __future__ import annotations\n"
        b"import dataclasses\n"
        b"@dataclasses.dataclass\n"
        b"class Score:\n"
        b"    exact: float\n",
    )
    assert load_scorer(pin, into).Score(1.0).exact == 1.0
