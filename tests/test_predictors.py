from __future__ import annotations

import pytest

from cannotread import predictors
from cannotread.predictors import REGISTRY, predict, register
from cannotread.view import View

# "maybe" has no training row and `everything` still answers with it, because it
# reads the label space. The predictors module docstring says when that agrees
# with the pre-registration on LegalBench.
TOY = View.build(
    "toy", "toy_task", ["q0", "q1", "q2"], ["yes", "no", "maybe"], ["yes", "no", "yes"]
)


def test_silent_gives_every_item_the_empty_answer() -> None:
    assert predict("silent", TOY) == {"q0": (), "q1": (), "q2": ()}


def test_everything_gives_every_item_the_whole_label_space() -> None:
    every = ("maybe", "no", "yes")
    assert predict("everything", TOY) == {"q0": every, "q1": every, "q2": every}


def test_predictors_do_not_need_a_training_split() -> None:
    no_train = View.build("toy", "no_train", ["q0"], ["a", "b"], None)
    assert predict("silent", no_train) == {"q0": ()}
    assert predict("everything", no_train) == {"q0": ("a", "b")}


def test_both_predictors_are_registered_under_their_names() -> None:
    assert REGISTRY["silent"] is predictors.silent
    assert REGISTRY["everything"] is predictors.everything


def test_a_name_cannot_be_registered_twice() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register("silent")(lambda view: {})
    assert REGISTRY["silent"] is predictors.silent


def test_an_unknown_name_says_what_is_registered() -> None:
    with pytest.raises(KeyError, match="everything"):
        predict("majority", TOY)


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
    monkeypatch.setitem(REGISTRY, "broken", lambda view: answers)
    with pytest.raises(error):
        predict("broken", TOY)
