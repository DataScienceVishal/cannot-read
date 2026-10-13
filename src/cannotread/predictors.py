"""Blind predictors, and the registry that lists them.

A predictor is given a `View` and answers every item in it. An answer is a tuple
of labels: `silent` gives the empty tuple and `everything` the whole label space.
What a scorer reads is the adapter's to produce, because the same tuple means
different things to different scorers. For SQuAD an empty tuple has to become an
empty string and for HumanEval an empty completion, and on the five LegalBench
tasks that docs/pre-registration.md says fit the claim the labels get joined
with commas.

On those five tasks the pre-registration defines `everything` as every label the
training split shows, not every label in the label space. The two are the same
set only if the LegalBench adapter, not written yet, builds the label space from
the fetched training split alone, as docs/licences.md plans. Nothing here checks
that. If the adapter took labels from test gold as well, `everything` would hand
them out.

Whatever runs every predictor should loop over `REGISTRY`, so that none is
skipped by hand, and should call each one through `predict`, which runs it
inside `blindness.guarded`. The blindness module docstring says what that
refuses and what it cannot see.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from cannotread.blindness import guarded
from cannotread.view import View

Answers = dict[str, tuple[str, ...]]


class Predictor(Protocol):
    def __call__(self, view: View) -> Answers: ...


REGISTRY: dict[str, Predictor] = {}


def register(name: str) -> Callable[[Predictor], Predictor]:
    def add(predictor: Predictor) -> Predictor:
        if name in REGISTRY:
            raise ValueError(f"a predictor called {name!r} is already registered")
        REGISTRY[name] = predictor
        return predictor

    return add


def predict(name: str, view: View) -> Answers:
    if name not in REGISTRY:
        raise KeyError(f"no predictor called {name!r}, registered: {sorted(REGISTRY)}")
    where = f"{name} on {view.benchmark}/{view.task}"
    with guarded(where):
        answers = REGISTRY[name](view)
    # A list of ids would get through the set comparison below.
    if not isinstance(answers, dict):
        raise TypeError(f"{where}: returned {type(answers).__name__}, not a dict")
    missing = set(view.item_ids) - set(answers)
    extra = set(answers) - set(view.item_ids)
    # A missing item would be scored however the scorer treats a missing
    # prediction, which is not the same as the predictor choosing to say nothing.
    if missing or extra:
        raise ValueError(f"{where}: missing {sorted(missing)}, not in the view {sorted(extra)}")
    for item_id, answer in answers.items():
        # A bare str would pass a looser check and be read as one label per letter.
        if not isinstance(answer, tuple) or not all(isinstance(a, str) for a in answer):
            raise TypeError(f"{where}: answer for {item_id!r} is not a tuple of str")
    return answers


@register("silent")
def silent(view: View) -> Answers:
    return {item_id: () for item_id in view.item_ids}


@register("everything")
def everything(view: View) -> Answers:
    return {item_id: view.label_space for item_id in view.item_ids}
