# cannot-read

[![ci](https://github.com/DataScienceVishal/cannot-read/actions/workflows/ci.yml/badge.svg)](https://github.com/DataScienceVishal/cannot-read/actions/workflows/ci.yml)

Some benchmark headline metrics can be scored well by a program that never looks
at the input. I wrote that down here before any code existed, so the claim
cannot move to fit whatever the numbers turn out to be.

The claim this repository has to earn: **recall-only and set-overlap headline
metrics have a floor, and the papers for the two live benchmarks below do not
report it.** For four named benchmarks I want to put that floor next to the
published best, measured through each benchmark's own unmodified scorer, by
predictors that are never shown the input text.

Blind here means a predictor gets item ids, the label space, and label counts
from the training split. It never gets a question, a document, a prompt or a
gold answer. A predictor that always picks the commonest training label is
blind. One that picks by the length of the question is not, and if I use
anything like it, it gets reported in a separate tier so the two cannot mix.

Nothing below has been measured in this repository yet. When a measured number
appears in this file it will come from a committed artifact, and until then
there are none.

## Where the idea came from

In [trail-scorer-audit](https://github.com/DataScienceVishal/trail-scorer-audit)
I found that a program which flags every span with every error category scores
above every model in the TRAIL paper's results table on both headline metrics,
because both divide by the number of errors in the answer key and never by the
number predicted. That was one benchmark and one scorer. This repository asks
whether the same shape of mistake exists in benchmarks people actually cite, and
keeps the claim narrow enough that it can be wrong.

## The four benchmarks, and why each one is here

| benchmark | role | what I expect |
|---|---|---|
| SQuAD v2.0 | calibration | A predictor that answers nothing should score around half, because roughly half of the dev questions have no answer (Rajpurkar, Jia and Liang 2018). This floor is already well known. It is here to show the instrument reads true on a case where the answer is known in advance. |
| LegalBench | live | Open. Each task is scored with its own metric, and the tasks scored with balanced accuracy are already defended against a predictor that always picks the commonest label. The claim can only bite on the others. |
| BFCL v4 | live | Open. Its irrelevance categories reward declining to call a function, so a predictor that always declines should take those categories in full. What I do not know is how much of the averaged headline that buys. |
| HumanEval | control | Zero. A predictor that never reads the prompt cannot write code that passes the unit tests. If it scores above zero, I check my own code before I believe the result. |

The four headline metrics do not all fit the claim, and the claim should not
pretend they do. SQuAD's F1 is token overlap between the prediction and the gold
answer, counted as a multiset, which is set overlap in all but name, and an
empty prediction on an unanswerable question counts as a perfect match.
LegalBench's balanced accuracy is the mean of per-class recall, so it is
recall-only, but averaging over classes is what defends it against a constant
predictor. BFCL's overall score is neither: it is a weighted average of
per-category accuracies, and it is here because averaging lets a category that
rewards doing nothing count towards the headline like any other. HumanEval's
pass@1 is neither as well, which is why it is the control. That leaves the
claim, as worded, with less to stand on in the live benchmarks than the first
paragraph suggests. What it does have on LegalBench is small. Each of
LegalBench's 162 tasks is scored with its own metric, so there the claim is
tested task by task. 155 use balanced accuracy, one is graded by hand, and of
the six left, five fit the claim: two count an answer correct when one part of
it matches the gold, with nothing charged for the rest, and three score F1
between the gold and the generated answers split on commas. The task names and
how each is scored are in [docs/pre-registration.md](docs/pre-registration.md).
BFCL's way in is the average. If a floor turns up through class balance or
averaging rather than through those five tasks, the claim gets reworded to say
so.

Four is the plan and three is what I expect to finish. A repository with three
benchmarks and an honest table is better than a framework with no numbers.

## What would make this README embarrassing

If LegalBench and BFCL both come out with floors far below their published
numbers, then the only floor in this repository is the SQuAD one, which
everybody already knew about. The claim above would then reduce to a
replication, and this file will say so in those words rather than being
rewritten to sound as though that was the plan.

A low floor also proves less than it looks like. Feng, Wallace and Boyd-Graber
(ACL 2019, "Misleading Failures of Partial-input Baselines") showed that when a
partial-input baseline fails, that does not mean the dataset is free of
artifacts, because some only show up in the full input. So a blind predictor
scoring near zero says nothing about whether a benchmark is sound. Only the
opposite direction is evidence: a blind predictor scoring high.

## What this is not

It is not a general tool that audits any benchmark you point it at. That has
been done recently and at a much larger scale. BenchJack (Wang et al., arXiv
2605.12673, May 2026) built an automated system for finding reward hacks in
agent benchmarks, and Automated Benchmark Auditing (arXiv 2605.26079, May 2026)
used an agentic framework to look across 168 benchmarks for broken tasks, such
as ones with incomplete environment specifications or limited grading logic.
This repository looks at one narrower thing, the arithmetic of the headline
metric, and only on four benchmarks, through scorers it does not modify.

Part of BenchJack overlaps with this repository directly. Item L3 of its
Agent-Eval Checklist (Appendix C.5) tells benchmark designers to run a null
agent that returns empty responses and to treat any task scoring above zero as
a flaw, and its example of weak string matching sends one fixed string, the
same for every task, through WebArena's unmodified scorer. Both are input-blind
predictors. Neither is set against the published best, which is the comparison
this repository is meant to make. The L3 rule also assumes that doing nothing
should never score. On SQuAD v2.0's unanswerable questions an empty answer is
the correct answer, and BFCL's irrelevance categories reward declining to call
a function, so on both the question is how much of the headline doing nothing
buys. The details are in
[docs/related-work.md](docs/related-work.md).

It also makes no model calls. No predictor here has a model behind it, so the
whole thing will run on a laptop CPU at no cost.

## Running the checks

Nothing runs against a benchmark yet. With [uv](https://docs.astral.sh/uv/)
installed:

```
uv sync --all-extras
scripts/install-hooks.sh
uv run pytest
uv run ruff check .
uv run python scripts/check_fingerprint.py
```

An adapter is the code that reads one benchmark, and none exists yet.
`src/cannotread/adapter.py` says what each will have to provide: the benchmark's
name, its task names, a `View` per task, and a `score` that takes a predictor's
answers on a task and returns that task's metrics. `checked_view` refuses a task
the adapter does not list, and a `View` that names a different benchmark or task
from the one asked for, because a mislabelled view would file one task's scores
under another's. Nothing calls it outside the tests until the first adapter
exists. `tests/test_adapter.py` runs every predictor through a fake adapter
whose gold answers stay in the test file and never go into a `View`.

The tests check that the suite cannot open a network socket, and that
`View`, the object a predictor is given, has fields only for the benchmark and
task names, item ids, the label space and training label counts. Checking that
an id does not spell out its question is left to each benchmark's adapter.
`tests/test_predictors.py` runs the five predictors written so far on hand-built
`View`s with no scorer behind them. `silent` answers nothing, `everything` gives
every label in the label space, `majority` gives the commonest training label, `prior`
draws a label per item in proportion to the training counts, and `uniform`
draws one per item evenly from the label space. The tests also check that a
predictor is refused if it skips an item, answers one that is not in the view,
or gives an answer that is not a tuple of strings. `majority` and
`prior` never give a label the training split does not show. Where a task has
no training split, or an empty one, they raise `NotFitted` and do not fall back
to the label space. Each benchmark's adapter will build the label space, and if
it took labels from test gold as well, falling back would hand those out.
`everything` and `uniform` read the label space by definition, so they are only
as blind as the label space the adapter builds.

`prior` and `uniform` draw at random, so each will be run on 100 seeds and
reported as its mean over them, with the best seed beside it, labelled as the
best of 100. The mean is the one that counts, because the best seed is only
known after scoring every seed against the gold answers. The dated entry in
[docs/pre-registration.md](docs/pre-registration.md) fixes this before either
has been run on a benchmark.

`predict` runs every predictor with a Python audit hook switched on
(`sys.addaudithook`, PEP 578). The hook refuses a predictor if it opens a file,
lists a directory, connects to a SQLite database, starts a process or opens a
socket, and still refuses it if it catches the error. Reading Python source and
compiled files is let through when the import system, or the tokenize module
that tracebacks read source through, asks for it, because the standard library
imports some modules lazily and a traceback reads the source it quotes. The
tests run each registered predictor with the hook on, and a test-only predictor
tries each of those cheats in turn. Inside the test suite pytest-socket stops an
internet socket before the hook sees it, so the socket case is tested with a
Unix socket. The hook is meant to catch a mistake in this repository's own
predictors, and code written to get round it can. It cannot see a predictor
reading content that a module already holds in memory, or reading through a C
extension such as DuckDB, and nothing checks yet that adapters keep fetched data
out of module globals. A longer list of what it misses is in
[src/cannotread/blindness.py](src/cannotread/blindness.py).

`src/cannotread/fetch.py` refuses a downloaded file whose SHA-256 is not the
one pinned, before anything is written to disk. A pin is a URL, the commit or
CodaLab bundle id it is at, and the digest, and a branch name such as `main` is
refused as a revision. `fetch` and `verified_bytes` hand back the bytes they
hashed and not a path, so a stored file edited after fetching is refused the
next time it is asked for. The tests fetch from `file://` URLs, since the suite
cannot open an internet socket. No adapter pins a real file yet.

Checking a scorer's digest and then importing it by path is not enough. Python
can run a cached `.pyc` in place of the source whenever the cache's recorded
mtime and size match the source file's, and then the digest describes a file
that did not run. trail-scorer-audit's first loader had this bug. So `load_scorer`, in
`src/cannotread/scorer.py`, compiles the bytes `verified_bytes` hashed and
executes them as a new module, and a stored scorer with one byte changed is
refused before any of it runs. The module is never named `__main__`, so the
scorer's command-line block does not run. This only works for a scorer that is
one file, which BFCL's and HumanEval's are not.

`scripts/install-hooks.sh` copies the pre-commit hook into `.git/hooks`, where it
runs ruff and the word list in `scripts/style-words.md` over every commit.
