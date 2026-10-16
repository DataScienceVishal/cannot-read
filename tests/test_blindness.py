from __future__ import annotations

import contextlib
import importlib
import importlib._bootstrap
import linecache
import os
import socket
import sqlite3
import subprocess
import sys
import threading
import traceback
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pytest

from cannotread import blindness
from cannotread.blindness import BlindnessViolation, guarded
from cannotread.predictors import REGISTRY, Answers, NotFitted, predict
from cannotread.view import View

# In order: a label the training split never shows, no training split (as for
# `rule_qa` and `scalr`), a single label, no items at all.
VIEWS = [
    View.build("toy", "three", ["q0", "q1", "q2"], ["yes", "no", "maybe"], ["yes", "no", "yes"]),
    View.build("toy", "no_train", ["q0", "q1"], ["a", "b"], None),
    View.build("toy", "one_label", ["q0"], ["only"], ["only", "only"]),
    View.build("toy", "no_items", [], ["a", "b"], ["a"]),
]


def test_the_registry_holds_the_predictors_written_so_far() -> None:
    # With an empty registry the test below would be skipped and the suite stay green.
    assert {"silent", "everything", "majority", "prior", "uniform"} <= set(REGISTRY)


@pytest.mark.parametrize("view", VIEWS, ids=[v.task for v in VIEWS])
@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_registered_predictor_answers_with_the_guard_on(
    monkeypatch: pytest.MonkeyPatch, name: str, view: View
) -> None:
    original = REGISTRY[name]
    seen: list[bool] = []

    def watched(view: View, *, seed: int) -> Answers:
        seen.append(blindness.is_guarded())
        return original(view, seed=seed)

    monkeypatch.setitem(REGISTRY, name, watched)
    try:
        predict(name, view, seed=0)
    except NotFitted:
        assert view.train_label_counts is None
    assert seen == [True]


@pytest.fixture
def gold_file(tmp_path: Path) -> Path:
    gold = tmp_path / "dev.json"
    gold.write_text('{"q0": "yes"}')
    return gold


# pytest-socket refuses an internet socket before the audit event fires, so
# the socket case uses a Unix one, which `--allow-unix-socket` lets through.
CHEATS: dict[str, Callable[[Path], object]] = {
    "open": lambda gold: gold.read_text(),
    "os.listdir": lambda gold: os.listdir(gold.parent),
    "os.scandir": lambda gold: list(os.walk(gold.parent)),
    "sqlite3.connect": lambda gold: sqlite3.connect(gold),
    "subprocess.Popen": lambda gold: subprocess.run(["cat", str(gold)], check=True),
    "socket.__new__": lambda gold: socket.socket(socket.AF_UNIX, socket.SOCK_STREAM),
}


@pytest.mark.parametrize("event", sorted(CHEATS))
def test_a_predictor_that_goes_looking_is_refused_before_the_call_happens(
    monkeypatch: pytest.MonkeyPatch, gold_file: Path, event: str
) -> None:
    got_through: list[object] = []

    def cheat(view: View, *, seed: int) -> Answers:
        got_through.append(CHEATS[event](gold_file))
        return {item_id: () for item_id in view.item_ids}

    monkeypatch.setitem(REGISTRY, "cheat", cheat)
    with pytest.raises(BlindnessViolation, match=f"cheat on toy/three tried {event}"):
        predict("cheat", VIEWS[0], seed=0)
    assert got_through == []


@pytest.mark.parametrize("event", sorted(CHEATS))
def test_catching_the_refusal_does_not_get_a_predictor_through(
    monkeypatch: pytest.MonkeyPatch, gold_file: Path, event: str
) -> None:
    def hides_it(view: View, *, seed: int) -> Answers:
        with contextlib.suppress(BaseException):
            CHEATS[event](gold_file)
        return {item_id: () for item_id in view.item_ids}

    monkeypatch.setitem(REGISTRY, "hides_it", hides_it)
    with pytest.raises(BlindnessViolation, match=f"hides_it on toy/three tried {event}"):
        predict("hides_it", VIEWS[0], seed=0)


def test_a_guard_of_its_own_does_not_get_a_predictor_through(
    monkeypatch: pytest.MonkeyPatch, gold_file: Path
) -> None:
    def hides_it_deeper(view: View, *, seed: int) -> Answers:
        with contextlib.suppress(BaseException), guarded("decoy"):
            gold_file.read_text()
        return {item_id: () for item_id in view.item_ids}

    monkeypatch.setitem(REGISTRY, "hides_it_deeper", hides_it_deeper)
    with pytest.raises(BlindnessViolation, match="hides_it_deeper on toy/three tried open"):
        predict("hides_it_deeper", VIEWS[0], seed=0)


def test_an_import_inside_the_standard_library_is_let_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `most_common` imports heapq when it is called. pytest has already
    # imported heapq, so it is removed here to make the import really happen.
    monkeypatch.delitem(sys.modules, "heapq")

    def commonest(view: View, *, seed: int) -> Answers:
        assert view.train_label_counts is not None
        top = Counter(view.train_label_counts).most_common(1)[0][0]
        return {item_id: (top,) for item_id in view.item_ids}

    monkeypatch.setitem(REGISTRY, "commonest", commonest)
    yes = ("yes",)
    assert predict("commonest", VIEWS[0], seed=0) == {"q0": yes, "q1": yes, "q2": yes}
    assert "heapq" in sys.modules


def test_formatting_a_traceback_is_let_through(monkeypatch: pytest.MonkeyPatch) -> None:
    # linecache would otherwise answer from what it already read for pytest.
    linecache.clearcache()
    quoted: list[str] = []

    def explains(view: View, *, seed: int) -> Answers:
        try:
            raise ValueError("tie between two labels")
        except ValueError:
            quoted.append(traceback.format_exc())
        return {item_id: () for item_id in view.item_ids}

    monkeypatch.setitem(REGISTRY, "explains", explains)
    predict("explains", VIEWS[0], seed=0)
    assert 'raise ValueError("tie between two labels")' in quoted[0]


def test_a_module_that_reads_gold_when_imported_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, gold_file: Path
) -> None:
    (tmp_path / "loads_gold.py").write_text(f"GOLD = open({str(gold_file)!r}).read()\n")
    monkeypatch.syspath_prepend(tmp_path)

    def imports_it(view: View, *, seed: int) -> Answers:
        importlib.import_module("loads_gold")
        return {item_id: () for item_id in view.item_ids}

    monkeypatch.setitem(REGISTRY, "imports_it", imports_it)
    with pytest.raises(BlindnessViolation, match="imports_it on toy/three tried open .*dev.json"):
        predict("imports_it", VIEWS[0], seed=0)


# Called with no Python frame in between, so the import system is the direct
# caller. The process and socket events are raised from subprocess.py and
# socket.py, which can never look like the import system, so they are not here.
DIRECT: dict[str, Callable[[Path], object]] = {"open": open, "sqlite3.connect": sqlite3.connect}


@pytest.mark.parametrize("event", sorted(DIRECT))
def test_a_cheat_made_through_the_import_system_is_refused(
    monkeypatch: pytest.MonkeyPatch, gold_file: Path, event: str
) -> None:
    def borrows_the_importer(view: View, *, seed: int) -> Answers:
        importlib._bootstrap._call_with_frames_removed(DIRECT[event], gold_file)
        return {item_id: () for item_id in view.item_ids}

    monkeypatch.setitem(REGISTRY, "borrows", borrows_the_importer)
    with pytest.raises(BlindnessViolation, match=f"borrows on toy/three tried {event}"):
        predict("borrows", VIEWS[0], seed=0)


def test_an_import_that_would_write_bytecode_is_let_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # pytest may run with bytecode writing off, which would hide the .pyc write.
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    (tmp_path / "never_imported.py").write_text("LABEL = 'yes'\n")
    monkeypatch.syspath_prepend(tmp_path)

    def imports_late(view: View, *, seed: int) -> Answers:
        label = importlib.import_module("never_imported").LABEL
        return {item_id: (label,) for item_id in view.item_ids}

    monkeypatch.setitem(REGISTRY, "imports_late", imports_late)
    assert predict("imports_late", VIEWS[1], seed=0) == {"q0": ("yes",), "q1": ("yes",)}
    assert sys.dont_write_bytecode is False
    monkeypatch.delitem(sys.modules, "never_imported")


def test_another_threads_file_access_is_not_blamed_on_the_predictor(
    monkeypatch: pytest.MonkeyPatch, gold_file: Path
) -> None:
    predictor_running = threading.Event()
    read_done = threading.Event()
    read: list[str] = []

    def bystander() -> None:
        predictor_running.wait(5)
        read.append(gold_file.read_text())
        read_done.set()

    def waits(view: View, *, seed: int) -> Answers:
        predictor_running.set()
        read_done.wait(5)
        return {item_id: () for item_id in view.item_ids}

    monkeypatch.setitem(REGISTRY, "waits", waits)
    thread = threading.Thread(target=bystander)
    thread.start()
    predict("waits", VIEWS[0], seed=0)
    thread.join(5)
    assert read == ['{"q0": "yes"}']


def test_the_guard_is_off_once_predict_returns(
    monkeypatch: pytest.MonkeyPatch, gold_file: Path
) -> None:
    predict("silent", VIEWS[0], seed=0)
    assert not blindness.is_guarded()
    monkeypatch.setitem(REGISTRY, "peeks", lambda view, *, seed: gold_file.read_text())
    with pytest.raises(BlindnessViolation):
        predict("peeks", VIEWS[0], seed=0)
    assert not blindness.is_guarded()
    assert gold_file.read_text() == '{"q0": "yes"}'
