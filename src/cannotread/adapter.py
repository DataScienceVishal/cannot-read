"""What a benchmark's adapter has to provide, and the check on the views it builds.

An adapter is the only code here that reads a benchmark. It lists the benchmark's
tasks, builds a `View` for each, and scores a predictor's answers on a task
through the benchmark's own scorer. No adapter is written yet. There will be one
per benchmark in the README table, and this is what each has to provide. The
repo runs no type checker, so nothing checks the fake in tests/test_adapter.py,
or any adapter written later, against `Adapter`. The class only writes the shape
down.

Four things are the adapter's to get right and nothing in this module checks
them:
- turning a tuple of labels into what the scorer reads. The predictors module
  docstring says why the same tuple means different things to different scorers;
- choosing item ids that do not spell out the question, which `View` cannot tell;
- building a label space with no label seen only in test gold. On the five
  LegalBench tasks where the pre-registration defines `everything` by the
  training split, that means from the fetched training split alone, as
  docs/licences.md plans;
- keeping fetched content out of module globals, because `blindness.guarded`
  cannot see a predictor reading one.

What is checked, by `checked_view`, is that the task is one the adapter lists
and that the `View` it returns names the benchmark and task it was asked for.
`prior` and `uniform` seed their draws from those names, so a mislabelled view
gets another task's draws. If the two tasks also share item ids, as tasks
numbered from 0 would, the scores filed under one task come from a predictor
shown the other's label space and counts, and nothing in this repository
fails. Whatever runs the adapters has to get views through `checked_view` and
not call `view` directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from cannotread.predictors import Answers
from cannotread.view import View


class Adapter(Protocol):
    benchmark: str

    def tasks(self) -> tuple[str, ...]: ...

    def view(self, task: str) -> View: ...

    def score(self, task: str, answers: Answers) -> Mapping[str, float]: ...


def checked_view(adapter: Adapter, task: str) -> View:
    if task not in adapter.tasks():
        raise KeyError(f"{adapter.benchmark} has no task called {task!r}")
    view = adapter.view(task)
    if (view.benchmark, view.task) != (adapter.benchmark, task):
        raise ValueError(
            f"asked {adapter.benchmark} for {task!r} and got a view of "
            f"{view.benchmark}/{view.task}"
        )
    return view
