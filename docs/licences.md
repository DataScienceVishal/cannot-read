# Licences

Two questions for each benchmark, settled before any adapter exists: can its
gold answers be committed to this repository, and can its scorer be committed,
or does it have to be fetched at a pinned revision. A benchmark would be
dropped here if its terms ruled out fetching its data, running its own scorer
on it, and publishing the scores. None is dropped, though one question about
LegalBench is left open at the end.

The licence files, dataset cards and task READMEs below were read on
2026-10-06. No benchmark data was downloaded to do it. The one download that
has happened, during the review of the pre-registration, is disclosed in
[pre-registration.md](pre-registration.md).

## The short answer

| benchmark | data licence | scorer licence | gold committed | scorer committed |
|---|---|---|---|---|
| SQuAD v2.0 | CC BY-SA 4.0 | none stated | no | no, fetched |
| LegalBench | set per task, see below | none stated | no | no, fetched |
| BFCL v4 | Apache 2.0 | Apache 2.0 | no | no, fetched |
| HumanEval | MIT | MIT | no | no, fetched |

For BFCL and HumanEval the licence would allow committing both, and I am not
doing it. I am fixing the rule here: each scorer is fetched at a pinned
revision, its SHA-256 is checked, and it is run from those same bytes, never
from a copy kept in this repository. That is the rule
[trail-scorer-audit](https://github.com/DataScienceVishal/trail-scorer-audit)
follows in its `upstream.py`, which hashes the TRAIL scorer and compiles the
bytes it hashed. Gold stays out for a related
reason: if it lived in the repository, a predictor would be one import away
from it. So the last two columns are the same in every row.

What does get committed is item ids, label counts, hashes of the fetched files,
and result artifacts. A `silent` prediction file for SQuAD, for example, is
question ids mapped to the empty string. Label strings are committed only where
the label space is a small closed set, such as `Yes` and `No`. On the six
LegalBench tasks below that property L1 covers, the labels are case names,
defined terms, party names and integers, which is to say training answers, so
their label space is built at run time from the fetched training split and
never written to the repository. For those six only totals are committed,
because counts keyed by label would write the labels out anyway.

## SQuAD v2.0

The SQuAD site (rajpurkar.github.io/SQuAD-explorer) says the dataset is
"distributed under the CC BY-SA 4.0 license", and the Hugging Face copy,
`rajpurkar/squad_v2`, carries the same tag. Committing `dev-v2.0.json` would be
allowed with attribution, but any file holding it would then have to be under
the same licence.

The scorer is `evaluate-v2.0.py` from CodaLab bundle
`0x6b567e1cf2e041ec80d7098f031c5c9e`. The bundle's licence field is empty, and
the file's docstring does not mention one. With no licence stated it is not
mine to redistribute, so it is fetched at run time and checked against a hash.

## LegalBench

Here the dataset-level licence is not the answer. The repository
(github.com/HazyResearch/legalbench) has no licence file, and its README says
LegalBench "is a mix of created and transformed datasets" and asks users to
follow each dataset creator's licence. The Hugging Face card for
`nguha/legalbench` is tagged `cc-by-4.0`, but its licensing section says the
tasks "are subject to different licenses" and points to the paper.

Each task folder at commit `10be1b6` has a README with a `**License**:` line.
The table counts that line over the 162 configs the datasets-server listed on
2026-10-05, at revision `daec823`. It was produced by a short script run while
this file was written, which is not committed. The script ignored case and the
link target, so `CC By 4.0` and `CC by 4.0` both count as CC BY 4.0, and gave
the nine READMEs that say "Creative Commons Attribution-NonCommercial License"
with no version a row of their own. Six licences plus that one unversioned
spelling:

| licence | configs |
|---|---|
| CC BY 4.0 | 126 |
| CC BY-NC-SA 4.0 | 16, all `learned_hands_` |
| CC BY-NC, no version given | 9, all `opp115_` |
| CC BY-NC 4.0 | 4: `canada_tax_court_outcomes`, `consumer_contracts_qa`, `textualism_tool_dictionaries`, `textualism_tool_plain` |
| MIT | 4: `nys_judicial_ethics`, `privacy_policy_qa`, `sara_entailment`, `sara_numeric` |
| CC BY-SA 4.0 | 2: `definition_classification`, `definition_extraction` |
| CC BY-NC 3.0 | 1: `privacy_policy_entailment` |

The committed index of tasks, once it exists, should record each task's
licence from the same line. If it disagrees with this table, the index wins and
this file gets a dated correction under the table.

Two things about the upstream folders will matter for the fetcher. Seven
configs, `cuad_anti-assignment`, `cuad_exclusivity`, `cuad_insurance`,
`cuad_non-compete`, `cuad_non-disparagement`, `cuad_rofr-rofo-rofn` and
`cuad_unlimited-all-you-can-eat-license`, have their README in a folder spelled
`CUAD_` with capitals, and for the last one a lowercase folder also exists
holding only a prompt file. On a case-insensitive disk like the default macOS
one, the two names collide. And `abercrombie` and `rule_qa` name the file
`README.MD`.

The six tasks that [pre-registration.md](pre-registration.md) leaves for
property L1 are `sara_numeric` (MIT), `citation_prediction_open` (CC BY 4.0),
`definition_extraction` (CC BY-SA 4.0) and the three `ssla_` tasks (CC BY 4.0).
None of them is non-commercial.

`evaluation.py`, the scorer, has no licence stated anywhere in the repository,
so it is fetched at `10be1b6` and hashed, like SQuAD's.

The task READMEs are written by the task contributors, and nothing checks
them. This table records what they say. A task's actual terms could differ.

## BFCL v4

The gorilla repository has an Apache 2.0 licence file at `58f57e9`, the commit
the pre-registration pins. The package inside it, `bfcl_eval`, declares Apache
2.0 in its `pyproject.toml`, and the data folder's README and the Hugging Face
copy (`gorilla-llm/Berkeley-Function-Calling-Leaderboard`) are both tagged
`apache-2.0`. The data sits in that package (`bfcl_eval/data/`, with the gold
calls under `possible_answer/`), so one licence covers the scorer and the gold.
Committing either would be allowed with the licence and notice kept.

The data folder's README describes the V2 release, which added the live
categories, as "employing enterprise and OSS-contributed live data". The
licence tag covers them as published, and I have no way to check the
contributors' terms beyond that.

## HumanEval

github.com/openai/human-eval is under MIT, checked at commit `6d43fb9`, the
newest on its default branch on 2026-10-06. The problems ship inside it as
`data/HumanEval.jsonl.gz`, and the Hugging Face copy `openai/openai_humaneval`
is tagged `mit`.

The commit fetched matters for more than the licence. In `execution.py` at
`463c980` (2021) the line `exec(check_program, exec_globals)` is commented out,
so that copy marks every completion as passed. Commit `37c4dd6` ("fix broken
eval", 2025-01-17) made it live, and it is live at `6d43fb9`. The HumanEval
adapter has to pin a commit at or after `37c4dd6`.

## Left open

Thirty LegalBench configs carry a non-commercial licence. Not committing their
data does not settle whether fetching it, scoring it and publishing the scores
from a public portfolio repository is non-commercial use. I have to settle
that before the first LegalBench run. It does not decide whether LegalBench
stays in, because none of the 30 is one of the six L1 tasks. If the answer is
no, the 30 are left out of the run, and since all 30 are among the 155
balanced-accuracy tasks that property I2 covers, I2 in the pre-registration
gets a dated correction saying it was checked on 125.
