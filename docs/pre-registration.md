# Pre-registration

What each floor has to satisfy, fixed before any adapter, predictor or scorer
run exists in this repository. The commit that adds this file is the timestamp.

## What was looked at before writing this

Four sources were read before the first draft, none of them task content, and
they are why some rows below give an exact value:

- LegalBench's `evaluation.py` at commit `10be1b6` (github.com/HazyResearch/legalbench).
- BFCL's `berkeley-function-call-leaderboard/bfcl_eval/eval_checker/eval_runner_helper.py`
  at commit `58f57e9` (github.com/ShishirPatil/gorilla).
- The official SQuAD v2.0 `evaluate-v2.0.py`, CodaLab bundle
  `0x6b567e1cf2e041ec80d7098f031c5c9e`, which is where the SQuAD site's
  "Evaluation Script v2.0" link goes.
- Row counts per split for `nguha/legalbench`, from the Hugging Face
  datasets-server `size` endpoint, queried 2026-10-05 when the dataset was at
  revision `daec823`.

One run happened before this file was committed, and it should not have. While
the draft was being reviewed, one of the review passes downloaded SQuAD's
`dev-v2.0.json` into a scratch directory and ran `silent` through the official
script. That run tests I1 and S1 below. Both were written before it, and
neither has been changed since. Its output is not recorded here. The number
goes in the README only once the SQuAD adapter reproduces it as a committed
artifact. No other benchmark data was downloaded and no other predictor was run.

Reading a scorer's code is enough to predict some floors exactly. Where that is
the case the property says so, and what pre-registering it buys is a check that
the instrument agrees with arithmetic done in advance.

## The predictors

All of these are blind in the README's sense: they see item ids, the label
space, and label counts from a training split, and nothing else.

- `silent` gives the empty answer to every item.
- `everything` gives every label in the label space at once, where the scorer
  accepts more than one.
- `majority` gives the commonest training label to every item. Ties go to the
  first label in sorted order, and the tied alternatives are scored as well and
  recorded, so the tie-break cannot be picked after seeing which label wins.
- `prior` samples a label per item in proportion to the training counts, seeded.
- `uniform` samples a label per item uniformly from the label space, seeded.
- `decline` is BFCL's version of `silent`: it never calls a function.

Predictors that look at the shape of the input, such as question length, are
the separate tier the README mentions. Nothing in this file covers them. Their
properties get added here, with a date, before any of them is run.

## Two kinds of property

**Instrument checks** have an answer fixed by the scorer's code. If one fails,
the bug is in this repository until shown otherwise, and no number from that
benchmark goes in the README until the failure is explained.

**Benchmark properties** are things a sound headline metric should have. A
violation is the result, and a clean sweep would mean the claim in the README
has nothing behind it.

## The properties

| property, as written before any code | how it is measured | what a failure means |
|---|---|---|
| **I1** On SQuAD v2.0 dev, `silent` gets EM and F1 from `evaluate-v2.0.py` both equal to 100 × u / n, where n is the number of questions in `dev-v2.0.json` and u the number whose gold answers are all empty after the script's own `normalize_answer` | count u and n from the file, run `silent` through the unmodified script, compare to two decimal places | the prediction writer is wrong, or the reading of the script above is. `compute_f1` returns `int(gold_toks == pred_toks)` when either side is empty, and `get_raw_scores` replaces an empty gold list with `['']`, so the value follows from the code |
| **S1** Every SQuAD v2.0 dev question marked `is_impossible: false` has at least one gold answer that survives `normalize_answer` | the same count as I1, split by `is_impossible` | `silent` scores above the unanswerable share, because an answerable question whose gold normalises to nothing (an answer that is only "the", say) is scored as unanswerable. Each such question gets named. Direction open |
| **I2** On every LegalBench task scored by `balanced_accuracy_score`, a constant predictor scores exactly 1/k when its label is one of the k distinct normalised gold labels in test, and 0 otherwise | `majority` and every constant label through the unmodified `evaluate`, against k counted from the test split | this repository's label normalisation differs from LegalBench's. This is the defence the README already concedes, written as a number so it can be checked |
| **L1** On the LegalBench tasks not scored by balanced accuracy, no blind predictor reaches the best published score for that task | blind predictors through `evaluate`, against the published per-task numbers once they are transcribed with citations | a floor exists on LegalBench, on those tasks. Direction open |
| **B1** On BFCL v4, `decline` scores 100% on `irrelevance` and `live_irrelevance` and 0% on every other category that enters the overall score | `decline` through the unmodified checker, per category | below 100 on irrelevance means the checker wants something other than "no call" and the category was misread here. Above 0 elsewhere means declining earns credit in a category meant to need a call, which would be a finding of its own. Expected from the category definitions, not yet checked against the checker code |
| **I4** If B1 holds, `decline`'s overall BFCL score is exactly 10 out of 100 | the overall from `eval_runner_helper.py`, recomputed from the per-category results | the per-category results were combined wrongly here. At `58f57e9` the overall is a weighted average of non-live, live, irrelevance, multi-turn and agentic with weights 10, 10, 10, 30 and 40, the irrelevance part is the mean of the two irrelevance categories, and `live_relevance` does not enter it. The weights carry an upstream TODO saying they may change, which is why the commit is pinned. Whether 10 is a large share of the gap between a weak and a strong model is not a property, and stays open until the leaderboard numbers are transcribed |
| **I3** On HumanEval, `silent` scores 0.0 pass@1 | empty completions through the unmodified `human-eval` execution | an empty completion leaves a function whose body is its docstring, so it returns `None`. A problem passes only if every assert in its test accepts that. If one does, the runner here gets checked first, as the README already says |
| **H1** On HumanEval, no completion from this fixed list passes any problem: `pass`, `return None`, `return 0`, `return 1`, `return ""`, `return []`, `return True`, `return False` | each as a completion for all problems, pass@1 per completion, and the ids of any problem that passes | a test suite accepts a constant, which makes that problem's pass meaningless whatever a model writes. The list is fixed here so it cannot grow until something passes. I expect it to hold, less firmly than I3 |

## Decisions fixed now, so the numbers cannot choose them

LegalBench leaves the claim little room. 155 of the 162 task configs the
datasets-server lists are in `EXACT_MATCH_BALANCED_ACC_TASKS`, which I2 covers.
Of the seven left, `rule_qa` is marked for manual evaluation and is out, which
leaves six for L1: `sara_numeric`, `citation_prediction_open`,
`definition_extraction` and the three `ssla_` tasks. These counts, and the
training split sizes given below, were taken from the two sources before any
index of tasks existed in this repository. The committed index of tasks and
metrics has to reproduce them, and if they disagree the index wins and this
file gets a dated correction under the paragraph.

Five of the six fit the claim. Two are recall-only by construction.
`definition_extraction` splits a generation on commas and counts it correct if
any piece matches, with nothing charged for the other pieces.
`citation_prediction_open` counts it correct if the gold case name appears
anywhere in the generation. The three `ssla_` tasks use `evaluate_ssla`, which
splits gold and generation on commas and counts a gold piece as found when some
generated piece contains it. It then takes one F1 over the whole task rather
than one per item. After each gold piece it adds every generated piece not yet
matched to the false positives, including correct pieces that a later gold piece
will match. Traced by hand through the code, not run: gold `a,b` against the
generation `a,b` gives two true positives, one false positive and an F1 of 0.8
for an exact answer. That matters for `everything` below, and for comparing any
floor here with a published number. `sara_numeric` is the odd one: it
takes the first integer in the generation and counts it correct within 10% of
the gold. The README as first committed said a floor on the remaining tasks
would come "through class balance and averaging rather than through recall". Reading the scorer showed that was
wrong for these five, and the README was corrected in the commit that added this
file. `everything` on these five means every label the training split shows,
joined into one generation with commas.

Among the tasks that have a training split, the split runs from 1 to 10 rows,
and 27 tasks, all `maud_` ones, have a single row. So `everything` has very
little to join, and on those 27 tasks `majority` and `prior` both mean "the
label of that one row". That is still blind and still goes in the table, and
the README has to say it. `rule_qa` and `scalr` have no training split at all.
`scalr` is one of the 155, so I2 still covers it through constant labels, but
`majority` and `prior` cannot be fitted on it and it is reported as not fitted.

BFCL's overall may not be reproducible offline. The agentic categories carry 40
of the 100 weight, and some categories may need a live API. Any category that
cannot run is not estimated. The README then reports the blind contribution as
"at least this much of 100, from the categories that ran", and does not set it
beside a published overall as if the two were the same quantity.

The README claim keeps its current wording until the numbers are in. It already
says how it gets reworded: if a floor turns up only through class balance or
averaging, it is described as coming through those. With the correction above,
the five LegalBench tasks are the one place in the live benchmarks where a floor
would fit the claim as worded. Changing the wording now, before anything is
measured, would be fitting the claim to a guess about the result.

## Things noticed in the scorers that are not properties

At `10be1b6`, `successor_liability` is in `EXACT_MATCH_BALANCED_ACC_TASKS`, and
`evaluate` checks that list first, so the task is scored by exact-match balanced
accuracy and the dedicated `evaluate_successor_liability` function, whose
docstring says F1 over the predicted exceptions, is never called. Which of the
two the LegalBench paper's numbers used has to be settled before its published
score for that task is compared with anything.

The same list contains `intra_rule_distinguishing`, which is not one of the
dataset configs. That is why 155 above counts configs in the list and not the
list's length.
