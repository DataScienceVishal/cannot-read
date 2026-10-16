from __future__ import annotations

import pytest

from cannotread import blindness
from cannotread.predictors import REGISTRY, Answers
from cannotread.seeds import SEEDS, OverSeeds, over_seeds
from cannotread.view import View

VIEW = View.build("toy", "seeds", [f"q{i}" for i in range(20)], ["yes", "no"], ["yes", "no"])


def share_of_yes(answers: Answers) -> dict[str, float]:
    return {"yes": sum(answer == ("yes",) for answer in answers.values()) / len(answers)}


def test_a_predictor_that_draws_at_random_is_scored_once_per_seed() -> None:
    calls: list[Answers] = []

    def score(answers: Answers) -> dict[str, float]:
        calls.append(answers)
        return share_of_yes(answers)

    uniform_over_seeds = over_seeds("uniform", VIEW, score)
    assert len(SEEDS) == 100
    assert tuple(uniform_over_seeds.by_seed) == SEEDS
    assert len(calls) == 100
    assert len({tuple(answers.values()) for answers in calls}) > 1


def test_every_seed_runs_through_predict_with_the_guard_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = REGISTRY["uniform"]
    guarded: list[bool] = []

    def watched(view: View, *, seed: int) -> Answers:
        guarded.append(blindness.is_guarded())
        return original(view, seed=seed)

    monkeypatch.setitem(REGISTRY, "uniform", watched)
    over_seeds("uniform", VIEW, share_of_yes)
    assert guarded == [True] * len(SEEDS)


def test_a_predictor_that_ignores_the_seed_is_scored_once_at_the_first_seed() -> None:
    majority_over_seeds = over_seeds("majority", VIEW, share_of_yes)
    assert majority_over_seeds.by_seed == {0: {"yes": 0.0}}
    assert majority_over_seeds.mean("yes") == majority_over_seeds.best("yes")[1] == 0.0


def test_the_mean_and_the_best_seed_come_from_every_seed_scored() -> None:
    toy = OverSeeds("toy", {0: {"f1": 0.2}, 1: {"f1": 0.6}, 2: {"f1": 0.6}, 3: {"f1": 0.4}})
    assert toy.mean("f1") == pytest.approx(0.45)
    assert toy.best("f1") == (1, 0.6)


def test_each_metric_has_its_own_best_seed() -> None:
    toy = OverSeeds("toy", {0: {"em": 0.5, "f1": 0.1}, 1: {"em": 0.1, "f1": 0.5}})
    assert toy.best("em") == (0, 0.5)
    assert toy.best("f1") == (1, 0.5)
