from __future__ import annotations

import dataclasses

import pytest

from cannotread.view import View

# "exempt" is in the label space but in no training row, so it must reach the
# View as a label with no count.
TRAIN_LABELS = ["no", "yes", "yes"]
LABELS = ["yes", "no", "exempt"]


def _toy_view() -> View:
    return View.build("toy", "toy_task", ["q0", "q1"], LABELS, TRAIN_LABELS)


def test_view_carries_no_input_text() -> None:
    # View has no way to take a question, so what this can check is the shape:
    # exactly the fields the README calls blind, each holding ids, labels or
    # counts. A new field widens what "blind" means, so it has to change this
    # test and the README together.
    assert [f.name for f in dataclasses.fields(View)] == [
        "benchmark",
        "task",
        "item_ids",
        "label_space",
        "train_label_counts",
    ]
    view = _toy_view()
    assert all(isinstance(i, str) for i in view.item_ids)
    assert all(isinstance(label, str) for label in view.label_space)
    assert all(isinstance(n, int) for n in view.train_label_counts.values())


def test_training_counts_are_the_labels_passed_in() -> None:
    # Whether those labels came from the training split is the adapter's job.
    view = _toy_view()
    assert dict(view.train_label_counts) == {"no": 1, "yes": 2}
    assert "exempt" in view.label_space


def test_label_space_is_sorted_and_deduplicated() -> None:
    # Eight labels, so an unsorted set passing by luck is about one in 40,000.
    labels = ["h", "c", "f", "a", "g", "c", "b", "e", "d"]
    view = View.build("toy", "t", ["a"], labels, None)
    assert view.label_space == ("a", "b", "c", "d", "e", "f", "g", "h")


def test_no_training_split_is_none_and_not_an_empty_count() -> None:
    assert View.build("toy", "t", ["a"], ["x"], None).train_label_counts is None
    assert dict(View.build("toy", "t", ["a"], ["x"], []).train_label_counts) == {}


def test_a_predictor_cannot_change_what_the_next_one_sees() -> None:
    view = _toy_view()
    with pytest.raises(dataclasses.FrozenInstanceError):
        view.item_ids = ("q9",)
    with pytest.raises(TypeError):
        view.train_label_counts["yes"] = 100


def test_a_list_of_ids_is_stored_as_a_tuple() -> None:
    ids = ["a", "b"]
    view = View("toy", "t", ids, ["x"], None)
    ids.append("c")
    assert view.item_ids == ("a", "b")


def test_the_callers_dict_is_copied() -> None:
    counts = {"x": 1}
    view = View("toy", "t", ["a"], ["x"], counts)
    counts["x"] = 50
    assert view.train_label_counts["x"] == 1


def test_a_bare_string_of_ids_or_labels_is_refused() -> None:
    with pytest.raises(TypeError, match="item_ids must be"):
        View.build("toy", "t", "abc", ["x"], None)
    with pytest.raises(TypeError, match="label_space must be"):
        View.build("toy", "t", ["a"], "yes", None)


def test_duplicate_item_ids_are_refused() -> None:
    with pytest.raises(ValueError, match="duplicate item ids"):
        View.build("toy", "t", ["a", "a"], ["x"], None)


def test_a_training_label_outside_the_label_space_is_refused() -> None:
    with pytest.raises(ValueError, match="outside the label space"):
        View.build("toy", "t", ["a"], ["x"], ["x", "y"])


def test_a_negative_count_is_refused() -> None:
    # build() counts with a Counter and cannot get here; direct construction can.
    with pytest.raises(ValueError, match="negative label count"):
        View("toy", "t", ["a"], ["x"], {"x": -1})
