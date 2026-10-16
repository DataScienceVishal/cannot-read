"""Which seeds a run uses, and which of the scores across them is the headline.

One run of `prior` or `uniform` is one sample of what it scores, so each is run
once for every seed in `SEEDS` and every score is kept. A predictor that does
not draw at random gives the same answers whatever the seed, and is run once,
at the first seed.

The mean over seeds is the headline. It estimates what the predictor scores in
expectation, and that is the number a blind floor is about. `best` returns the
maximum and its seed, for whatever writes the results to put beside the mean,
labelled as the best of however many seeds ran. Nothing writes results yet. The
maximum is not a property of the predictor. It is picked after every seed has
been scored against test gold, and it can only go up as more seeds are tried,
so it measures the search as much as the benchmark. It shows how far a floor
could be pushed up by trying seeds until one scores well.

Nothing works out 100 as the right count. With 100 seeds the standard error of
the mean is a tenth of the spread of a single seed's score.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from statistics import fmean

from cannotread.predictors import DRAWS_AT_RANDOM, Answers, predict
from cannotread.view import View

SEEDS = tuple(range(100))


@dataclass(frozen=True)
class OverSeeds:
    predictor: str
    by_seed: Mapping[int, Mapping[str, float]]

    def mean(self, metric: str) -> float:
        return fmean(scores[metric] for scores in self.by_seed.values())

    def best(self, metric: str) -> tuple[int, float]:
        """The highest score and the seed that got it, the lowest seed on a tie."""
        seed = max(sorted(self.by_seed), key=lambda s: self.by_seed[s][metric])
        return seed, self.by_seed[seed][metric]


def over_seeds(
    name: str, view: View, score: Callable[[Answers], Mapping[str, float]]
) -> OverSeeds:
    seeds = SEEDS if name in DRAWS_AT_RANDOM else SEEDS[:1]
    return OverSeeds(name, {seed: dict(score(predict(name, view, seed=seed))) for seed in seeds})
