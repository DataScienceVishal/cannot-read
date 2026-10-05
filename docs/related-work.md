# Related work

The two papers this repository has to get past before it is worth building. What
follows is taken from the arXiv HTML versions named under each heading, plus one
search of the ABA code repository, which is marked where it is used. Every other
figure is the paper's own.

## BenchJack

Wang, Li, Mang, Cheung, Sen and Song, "Do Androids Dream of Breaking the Game?
Systematically Auditing AI Agent Benchmarks with BenchJack", arXiv 2605.12673,
v1, 12 May 2026.

BenchJack drives a coding agent to read a benchmark's code and build exploits
that score without doing the task. It was run on ten agent benchmarks (SWE-bench
Verified and Pro, FrontierSWE, MLE-Bench, SkillsBench, Terminal-Bench, OSWorld,
WebArena, NetArena, AgentBench) and the paper reports 219 distinct flaws, sorted
into eight classes V1 to V8. Six of the eight are about the evaluation harness
rather than the metric: a planted `conftest.py`, a reward file written before
the verifier reads it, answers left readable inside the container, `eval()` on
agent output. The other two, V5 weak string matching and V6 evaluation logic
gaps, are the ones the paper groups under test validity.

Two parts come close to what I want to do. The V5 example on WebArena is a
single fixed string, the integers 0 to 999 followed by yes, no, true, false and
n/a, sent as the answer to every task, which passes the substring matcher
whenever the gold answer is a small number or a yes/no. That is an input-blind
predictor going through an unmodified scorer. And item L3 of their Agent-Eval
Checklist (Appendix C.5) tells benchmark designers to run a null agent that
returns empty responses on every task, and to treat any task scoring above zero
as a V6 flaw. So the predictor that answers nothing, which the README plans for
SQuAD, is already in their checklist. What BenchJack does not do is set either
of these against the published best. The L3 rule is a flag on tasks, and a
near-identical string, with a few more words added, is reported as a count of
tasks won (52 of 812 on their patched WebArena, Appendix F.10). The checklist
rule also assumes that doing nothing should never score. The paper treats that
as true for the agent benchmarks it audits. Of the four here it holds for
HumanEval but not for SQuAD v2.0 or BFCL. In SQuAD v2.0 an empty answer on an
unanswerable question is the correct answer, and BFCL's irrelevance categories
reward declining to call a function. Applied to those, the L3 rule would flag
every unanswerable or irrelevant item, so the useful question there is how much
of the headline the empty answer buys, not whether it scores at all. The paper's
own limitations section says the taxonomy is aimed at agent benchmarks and may
not cover other evaluation patterns.

## Automated Benchmark Auditing (ABA)

Junlin Wang, Bianchi, Zhu, Nie, Kwon, Dhingra and Zou, "Automated Benchmark
Auditing for AI Agents and Large Language Models", arXiv 2605.26079, v2, 26 May
2026.

ABA audits one task at a time. A collector gathers each task's instruction,
reference solution, environment and, in one mode, an agent trajectory. An LLM
auditor then grades the task against a rubric on three axes: instruction
(missing or ambiguous information), environment (the container contradicts the
prompt), and evaluation (tests too narrow, too broad, or a wrong gold answer).
Across 168 benchmarks and 34,285 tasks the paper reports major issues in 25.7%
of tasks, and removing the flagged tasks raises the mean per-model score on
every one of the five leaderboards where it could recompute scores (Figure 7).
The "too broad" case in the evaluation axis is the nearest thing to my question,
since a test that accepts trivially wrong output is a test a blind predictor
might pass. But it is judged per task by reading the task, and nothing in the
paper looks at the headline metric as a formula: what it divides by, how
categories are averaged, or what a constant or label-prior predictor scores on
it. None of my four benchmarks is in their list. The search: the 168 entries in
`benchmarks/multi-domain/benchmark_categorization.json` in their repository
(github.com/IsThatYou/auto-bench-audit, commit `f743419`) contain no match for
SQuAD, LegalBench, BFCL, Gorilla or HumanEval. HumanEval appears in the paper
only in the related-work section. Their limitations say the auditor is itself an
LLM, only a sample of the audited tasks was checked by humans, and the severity
rubric is a matter of judgement.

## What is left

Neither paper puts the headline score that blind predictors reach through a
benchmark's official scorer next to the published best, so that is what this
repository will measure.
