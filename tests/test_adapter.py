from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import pytest

from cannotread.adapter import checked_view
from cannotread.predictors import REGISTRY, Answers, NotFitted, predict
from cannotread.seeds import over_seeds
from cannotread.view import View

# The gold labels stay in these dicts and only their keys, the item ids, go into
# a View. `no_train` has no training split, like LegalBench's `scalr`.
GOLD = {
    "with_train": {"q0": "yes", "q1": "yes", "q2": "no", "q3": "yes"},
    "no_train": {"q0": "a", "q1": "b"},
}
TRAIN = {"with_train": ["yes", "no", "yes"], "no_train": None}
LABELS = {"with_train": ["yes", "no"], "no_train": ["a", "b"]}


@dataclass
class FakeAdapter:
    benchmark: str = "fake"
    built: list[View] = field(default_factory=list)

    def tasks(self) -> tuple[str, ...]:
        return tuple(GOLD)

    def view(self, task: str) -> View:
        # An unlisted task gets an empty view and no error, so that only
        # `checked_view` can be what refuses it.
        view = View.build(
            self.benchmark, task, tuple(GOLD.get(task, ())), LABELS.get(task, ()), TRAIN.get(task)
        )
        self.built.append(view)
        return view

    def score(self, task: str, answers: Answers) -> dict[str, float]:
        gold = GOLD[task]
        hits = sum(answers[item_id] == (label,) for item_id, label in gold.items())
        return {"accuracy": hits / len(gold)}


def test_every_registered_predictor_runs_through_the_fake_adapter() -> None:
    adapter = FakeAdapter()
    view = checked_view(adapter, "with_train")
    assert adapter.built == [view]

    def score(answers: Answers) -> Mapping[str, float]:
        return adapter.score("with_train", answers)

    accuracy = {name: over_seeds(name, view, score).mean("accuracy") for name in REGISTRY}
    # majority answers "yes" to all four and three of them are "yes" in gold.
    # silent and everything never equal a single gold label.
    assert accuracy["majority"] == 0.75
    assert accuracy["silent"] == accuracy["everything"] == 0.0


def test_a_task_with_no_training_split_raises_not_fitted_for_majority_and_prior() -> None:
    view = checked_view(FakeAdapter(), "no_train")
    for name in ("majority", "prior"):
        with pytest.raises(NotFitted):
            predict(name, view, seed=0)
    assert predict("uniform", view, seed=0).keys() == {"q0", "q1"}


class WrongTask(FakeAdapter):
    def view(self, task: str) -> View:
        return super().view("no_train" if task == "with_train" else "with_train")


class WrongBenchmark(FakeAdapter):
    def view(self, task: str) -> View:
        return View.build("other", task, tuple(GOLD[task]), LABELS[task], TRAIN[task])


@pytest.mark.parametrize("adapter", [WrongTask(), WrongBenchmark()])
def test_a_view_of_some_other_task_is_refused(adapter: FakeAdapter) -> None:
    with pytest.raises(ValueError, match="asked fake for 'with_train'"):
        checked_view(adapter, "with_train")


def test_a_task_the_adapter_does_not_list_is_refused_before_its_view_is_built() -> None:
    adapter = FakeAdapter()
    with pytest.raises(KeyError, match="no task called 'missing'"):
        checked_view(adapter, "missing")
    assert adapter.built == []
