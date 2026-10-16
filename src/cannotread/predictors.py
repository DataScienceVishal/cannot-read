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

`majority` and `prior` are fitted on the training label counts and nothing
else, so a label in the label space with no training row is never given by
either. Where there is no training split, or one with no labels, they raise
`NotFitted` and do not fall back to the label space, which is where a label
seen only in test gold could come in. `uniform` draws from the label space by
definition, so like `everything` it is only as blind as the label space the
adapter builds. The pre-registration has `scalr`
reported as not fitted; nothing catches `NotFitted` and reports it yet.

Every predictor takes a seed, including the ones that ignore it, so that a run
has to say which seed it used. `prior` and `uniform` draw at random and are
registered in `DRAWS_AT_RANDOM`, which is how `seeds.over_seeds` knows to run
them once per seed and the others once. Both seed their generator with the
benchmark and task names as well as the seed, so that two tasks with the same
number of items and labels do not get the same draws.

Whatever runs every predictor should loop over `REGISTRY`, so that none is
skipped by hand, and should call each one through `predict`, which runs it
inside `blindness.guarded`. The blindness module docstring says what that
refuses and what it cannot see.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from typing import Protocol

from cannotread.blindness import guarded
from cannotread.view import View

Answers = dict[str, tuple[str, ...]]


class Predictor(Protocol):
    def __call__(self, view: View, *, seed: int) -> Answers: ...


REGISTRY: dict[str, Predictor] = {}
DRAWS_AT_RANDOM: set[str] = set()


class NotFitted(ValueError):
    pass


def register(name: str, *, draws_at_random: bool = False) -> Callable[[Predictor], Predictor]:
    def add(predictor: Predictor) -> Predictor:
        if name in REGISTRY:
            raise ValueError(f"a predictor called {name!r} is already registered")
        REGISTRY[name] = predictor
        if draws_at_random:
            DRAWS_AT_RANDOM.add(name)
        return predictor

    return add


def predict(name: str, view: View, *, seed: int) -> Answers:
    if name not in REGISTRY:
        raise KeyError(f"no predictor called {name!r}, registered: {sorted(REGISTRY)}")
    where = f"{name} on {view.benchmark}/{view.task}"
    with guarded(where):
        answers = REGISTRY[name](view, seed=seed)
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
def silent(view: View, *, seed: int) -> Answers:
    return {item_id: () for item_id in view.item_ids}


@register("everything")
def everything(view: View, *, seed: int) -> Answers:
    return {item_id: view.label_space for item_id in view.item_ids}


def _generator(view: View, seed: int) -> random.Random:
    # A str seed goes through SHA-512 and never through hash(), so the draws do not
    # depend on PYTHONHASHSEED.
    return random.Random(f"{view.benchmark}/{view.task}/{seed}")


def _training_counts(view: View) -> Mapping[str, int]:
    where = f"{view.benchmark}/{view.task}"
    if view.train_label_counts is None:
        raise NotFitted(f"{where}: no training split")
    if not any(view.train_label_counts.values()):
        raise NotFitted(f"{where}: the training split has no labels")
    return view.train_label_counts


def commonest_labels(view: View) -> tuple[str, ...]:
    """Every label tied for the highest training count, in sorted order.

    `majority` answers with the first. The pre-registration also has the others
    scored and recorded, which is for whatever writes the results to do.
    """
    counts = _training_counts(view)
    top = max(counts.values())
    return tuple(label for label in view.label_space if counts.get(label, 0) == top)


@register("majority")
def majority(view: View, *, seed: int) -> Answers:
    label = commonest_labels(view)[0]
    return {item_id: (label,) for item_id in view.item_ids}


@register("prior", draws_at_random=True)
def prior(view: View, *, seed: int) -> Answers:
    counts = _training_counts(view)
    # Read in label-space order, so the draws do not depend on the order the
    # adapter happened to count the training rows in.
    labels = [label for label in view.label_space if counts.get(label, 0) > 0]
    draws = _generator(view, seed).choices(
        labels, weights=[counts[label] for label in labels], k=len(view.item_ids)
    )
    return {item_id: (label,) for item_id, label in zip(view.item_ids, draws, strict=True)}


@register("uniform", draws_at_random=True)
def uniform(view: View, *, seed: int) -> Answers:
    # Needs no training split, so unlike `prior` it answers on `scalr`, and it
    # gives labels the training split never shows.
    if not view.label_space:
        raise ValueError(f"{view.benchmark}/{view.task}: the label space is empty")
    draws = _generator(view, seed).choices(view.label_space, k=len(view.item_ids))
    return {item_id: (label,) for item_id, label in zip(view.item_ids, draws, strict=True)}
