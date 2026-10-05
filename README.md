# cannot-read

Some benchmark headline metrics can be scored well by a program that never looks
at the input. I am writing that down here before any code exists, so the claim
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
used an agentic framework to look for broken tasks across 168 benchmarks, such
as incomplete environment specifications and limited grading logic. This
repository looks at one narrower thing, the arithmetic of the headline metric,
and only on four benchmarks, through scorers it does not modify.

It also makes no model calls. No predictor here has a model behind it, so the
whole thing will run on a laptop CPU at no cost.
