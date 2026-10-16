from __future__ import annotations

import pytest

from cannotread import predictors
from cannotread.predictors import REGISTRY, NotFitted, commonest_labels, predict, register
from cannotread.view import View

# "maybe" has no training row and `everything` still answers with it, because it
# reads the label space. The predictors module docstring says when that agrees
# with the pre-registration on LegalBench.
TOY = View.build(
    "toy", "toy_task", ["q0", "q1", "q2"], ["yes", "no", "maybe"], ["yes", "no", "yes"]
)


def test_silent_gives_every_item_the_empty_answer() -> None:
    assert predict("silent", TOY, seed=0) == {"q0": (), "q1": (), "q2": ()}


def test_everything_gives_every_item_the_whole_label_space() -> None:
    every = ("maybe", "no", "yes")
    assert predict("everything", TOY, seed=0) == {"q0": every, "q1": every, "q2": every}


def test_predictors_do_not_need_a_training_split() -> None:
    no_train = View.build("toy", "no_train", ["q0"], ["a", "b"], None)
    assert predict("silent", no_train, seed=0) == {"q0": ()}
    assert predict("everything", no_train, seed=0) == {"q0": ("a", "b")}


def test_majority_gives_every_item_the_commonest_training_label() -> None:
    assert predict("majority", TOY, seed=0) == {"q0": ("yes",), "q1": ("yes",), "q2": ("yes",)}


def test_a_tie_goes_to_the_first_label_in_sorted_order_whatever_order_it_was_counted_in() -> None:
    for train in (["a", "a", "b", "b", "c"], ["b", "b", "a", "a", "c"]):
        tied = View.build("toy", "tied", ["q0"], ["a", "b", "c"], train)
        assert commonest_labels(tied) == ("a", "b")
        assert predict("majority", tied, seed=0) == {"q0": ("a",)}


def test_prior_draws_in_proportion_to_the_training_counts() -> None:
    ids = [f"q{i}" for i in range(3000)]
    view = View.build("toy", "many", ids, ["yes", "no", "maybe"], ["yes", "no", "yes"])
    drawn = [label for (label,) in predict("prior", view, seed=0).values()]
    assert drawn.count("yes") / len(drawn) == pytest.approx(2 / 3, abs=0.03)
    # "maybe" is in the label space and has no training row.
    assert set(drawn) == {"yes", "no"}


def test_prior_depends_on_the_seed_and_not_on_the_order_labels_were_counted_in() -> None:
    ids = [f"q{i}" for i in range(50)]
    yes_first = View.build("toy", "order", ids, ["yes", "no"], ["yes", "no", "no"])
    no_first = View.build("toy", "order", ids, ["yes", "no"], ["no", "yes", "no"])
    assert predict("prior", yes_first, seed=1) == predict("prior", no_first, seed=1)
    assert predict("prior", yes_first, seed=1) != predict("prior", yes_first, seed=2)


def test_uniform_draws_every_label_in_the_label_space_equally_often() -> None:
    ids = [f"q{i}" for i in range(3000)]
    view = View.build("toy", "many", ids, ["yes", "no", "maybe"], ["yes", "no", "yes"])
    drawn = [label for (label,) in predict("uniform", view, seed=0).values()]
    for label in ("yes", "no", "maybe"):
        assert drawn.count(label) / len(drawn) == pytest.approx(1 / 3, abs=0.03)


def test_uniform_answers_without_a_training_split() -> None:
    no_train = View.build("toy", "no_train", ["q0", "q1"], ["a", "b"], None)
    answers = predict("uniform", no_train, seed=0)
    assert set(answers) == {"q0", "q1"}
    assert all(answer in {("a",), ("b",)} for answer in answers.values())


def test_uniform_refuses_an_empty_label_space() -> None:
    view = View.build("toy", "no_labels", ["q0"], [], None)
    with pytest.raises(ValueError, match="label space is empty"):
        predict("uniform", view, seed=0)


@pytest.mark.parametrize("name", ["prior", "uniform"])
def test_two_tasks_of_the_same_shape_get_different_draws_from_the_same_seed(name: str) -> None:
    ids = [f"q{i}" for i in range(50)]
    first = View.build("toy", "first", ids, ["yes", "no"], ["yes", "no"])
    second = View.build("toy", "second", ids, ["yes", "no"], ["yes", "no"])
    assert predict(name, first, seed=1) == predict(name, first, seed=1)
    assert predict(name, first, seed=1) != predict(name, second, seed=1)


def test_draws_at_random_lists_exactly_the_predictors_whose_answers_change_with_the_seed() -> None:
    ids = [f"q{i}" for i in range(50)]
    view = View.build("toy", "seeds", ids, ["yes", "no"], ["yes", "no", "no"])
    changes = {n for n in REGISTRY if predict(n, view, seed=0) != predict(n, view, seed=1)}
    assert changes == predictors.DRAWS_AT_RANDOM == {"prior", "uniform"}


@pytest.mark.parametrize("name", ["majority", "prior"])
@pytest.mark.parametrize(
    ("counts", "reason"),
    [(None, "no training split"), ({}, "has no labels"), ({"a": 0, "b": 0}, "has no labels")],
    ids=["no training split", "empty training split", "all counts zero"],
)
def test_majority_and_prior_refuse_to_fit_on_the_label_space_alone(
    name: str, counts: dict[str, int] | None, reason: str
) -> None:
    # The label space may hold labels only test gold has, so it is not a fallback.
    view = View("toy", "unfitted", ("q0",), ("a", "b"), counts)
    with pytest.raises(NotFitted, match=reason):
        predict(name, view, seed=0)


def test_every_predictor_is_registered_under_its_name() -> None:
    assert REGISTRY["silent"] is predictors.silent
    assert REGISTRY["everything"] is predictors.everything
    assert REGISTRY["majority"] is predictors.majority
    assert REGISTRY["prior"] is predictors.prior
    assert REGISTRY["uniform"] is predictors.uniform


def test_a_name_cannot_be_registered_twice() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register("silent")(lambda view, *, seed: {})
    assert REGISTRY["silent"] is predictors.silent


def test_an_unknown_name_says_what_is_registered() -> None:
    with pytest.raises(KeyError, match="everything"):
        predict("no_such_predictor", TOY, seed=0)


@pytest.mark.parametrize(
    ("answers", "error"),
    [
        ({"q0": (), "q1": ()}, ValueError),
        ({"q0": (), "q1": (), "q2": (), "q3": ()}, ValueError),
        ({"q0": (), "q1": (), "q3": ()}, ValueError),
        (["q0", "q1", "q2"], TypeError),
        ({"q0": (), "q1": (), "q2": "yes"}, TypeError),
        ({"q0": (), "q1": (), "q2": ["yes"]}, TypeError),
        ({"q0": (), "q1": (), "q2": (1,)}, TypeError),
    ],
    ids=[
        "missing item",
        "extra item",
        "wrong item same count",
        "not a dict",
        "bare str",
        "list",
        "non-str label",
    ],
)
def test_predict_refuses_a_malformed_answer_set(
    monkeypatch: pytest.MonkeyPatch, answers: object, error: type[Exception]
) -> None:
    monkeypatch.setitem(REGISTRY, "broken", lambda view, *, seed: answers)
    with pytest.raises(error):
        predict("broken", TOY, seed=0)
