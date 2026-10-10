"""What a blind predictor is given: item ids, the label space, training label counts.

The README defines blind as these three things. The fields are those three, plus
the benchmark and task names so a result can say where it came from. There is no
field for a question, a document, a prompt or a gold answer, so a predictor
handed a `View` cannot read one through it.

Two openings are left, and both are the adapter's to close. Item ids could spell
out the question, and nothing here can tell. The label space is whatever the
caller passes, so a label seen only in test gold would reach the predictor as a
label it may give, without its count.

`train_label_counts` is None when the task has no training split (LegalBench's
`rule_qa` and `scalr`). That is different from an empty count: the
pre-registration says `scalr` is reported as not fitted, and an empty Counter
would let it look fitted on nothing.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType


def _as_tuple(where: str, name: str, values: Iterable[str]) -> tuple[str, ...]:
    # A bare string is an Iterable[str] too, and "abc" would become three ids.
    if isinstance(values, str):
        raise TypeError(f"{where}: {name} must be a sequence of strings, got a str")
    return tuple(values)


@dataclass(frozen=True)
class View:
    benchmark: str
    task: str
    item_ids: tuple[str, ...]
    label_space: tuple[str, ...]
    train_label_counts: Mapping[str, int] | None

    def __post_init__(self) -> None:
        where = f"{self.benchmark}/{self.task}"
        item_ids = _as_tuple(where, "item_ids", self.item_ids)
        if len(set(item_ids)) != len(item_ids):
            raise ValueError(f"{where}: duplicate item ids")
        object.__setattr__(self, "item_ids", item_ids)
        labels = _as_tuple(where, "label_space", self.label_space)
        # Sorted so that "first label in sorted order", the pre-registered
        # tie-break for `majority`, is the first tied label met in this order.
        object.__setattr__(self, "label_space", tuple(sorted(set(labels))))
        if self.train_label_counts is None:
            return
        unknown = set(self.train_label_counts) - set(self.label_space)
        if unknown:
            raise ValueError(f"{where}: training labels outside the label space: {sorted(unknown)}")
        if any(n < 0 for n in self.train_label_counts.values()):
            raise ValueError(f"{where}: negative label count")
        # A plain dict would leave the frozen dataclass mutable underneath, and a
        # predictor could change the counts the next predictor sees.
        object.__setattr__(
            self, "train_label_counts", MappingProxyType(dict(self.train_label_counts))
        )

    @classmethod
    def build(
        cls,
        benchmark: str,
        task: str,
        item_ids: Iterable[str],
        label_space: Iterable[str],
        train_labels: Iterable[str] | None,
    ) -> View:
        counts = None if train_labels is None else Counter(train_labels)
        return cls(benchmark, task, item_ids, label_space, counts)
