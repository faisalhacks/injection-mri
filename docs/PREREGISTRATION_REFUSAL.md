# Pre-registration: refusal culprits (neurons and heads)

Committed before any run on a main model (Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct) and before any family/platform/date analysis. Status when written: the code is not yet written; no model has been run on this branch. Qwen2.5-0.5B-Instruct is the dev model and will be used to debug code; its numbers may change code but not the thresholds below. Anything not listed here is **exploratory** and will be labelled so.

## Question

Which MLP units and attention heads causally decide (A) jailbreak bypass (a request refused plain but complied with under a public jailbreak template) and (B) over-refusal (a benign, risky-looking request refused)? Does patching them back fix the behaviour on held-out behaviours and held-out templates, better than controls, at low side-effect cost? Do different jailbreak strategy families rely on the same units?

## Definitions

- Greedy decoding. Baseline generations 64 new tokens; **all labels use the first 32 generated tokens** (intervention runs generate 32). Refusal label = phrase match (fixed list in `refusal.py`). A refusal judge (protectai/distilroberta-base-rejection-v1) is run on an audit sample (up to 300 generations pooled across sets) and agreement and Cohen's kappa are reported. The judge does not change labels. No human audit of harmful completions is done.
- Unit = one MLP neuron (input to `down_proj`) or one attention head's output (input to `o_proj`, slice of one head), **at the last prompt token and at every generated token**. Interventions replace the unit's value with its mean over the "clean" training group (mean patching), so they are input-independent.
- Metric M at the last prompt token = logit-difference (logsumexp over refusal-start tokens minus over comply-start tokens, token sets chosen from training labels) divided by its training std, plus the projection on the refusal direction at the validation-chosen layer divided by its training std. Higher M = more refusal.
- Attribution: attribution patching, gradient of M times (clean activation minus corrupt activation), per pair, at the last position. Culprit score = sign x mean over pairs, sign = +1 for A, -1 for B. Top 150 neurons and top 50 heads by attribution are validated by exact activation patching on 24 training pairs.

## Splits (seeded SHA-1 hash of ID, seed 0)

- Behaviours: JBB harmful 100 -> 50 train / 20 val / 30 test. JBB benign matched items follow their harmful partner's split.
- Wrappers: from the In-the-Wild Jailbreak Prompts dataset (jailbreak configs, deduplicated, 150 to 2500 characters). Families = the dataset's own community names, with the two releases merged by name: Basic, Toxic, Anarchy, Exception, Guidelines, Opposite (the six with at least 20 prompts in range). 12 templates per family, plus 24 templates with no community name ("unlabelled"). 96 templates, half train, half test, stratified so each family gets 6/6 and unlabelled 12/12. **Test templates are never seen during attribution, direction fitting or layer selection.** Templates with a placeholder (`[INSERT PROMPT HERE]` style) get the behaviour substituted, otherwise the behaviour is appended after a blank line.
- Over-refusal pool: all 250 XSTest safe prompts plus 600 random OR-Bench-Hard-1K prompts, split 50/20/30 by hash. XSTest unsafe (200) is a side-effect set only and is never used for fitting.
- Side-effect sets: Alpaca (200, a different 100 are used for the direction's harmless side), MMLU (400 questions, zero-shot letter logits), JBB plain harmful test (30).
- Layer selection for the direction baseline uses validation behaviours only.

## Interventions and controls

For k in {10, 50, 200} neurons and k in {5, 10, 20} heads:
1. **Culprits**: top-k by culprit score from the training pairs.
2. **Random**: k random units from the same layers (same per-layer counts as the culprit set). 20 draws at the primary k (50 neurons), 10 draws at the primary head k (10), 5 draws at other k.
3. **Activation-difference top-k** (non-causal): top-k by |mean clean - mean corrupt| / pooled std, same patching.
4. **Arditi direction** (strong baseline): A: add the training diff-of-means refusal vector at one layer (chosen on validation) at all positions. B: ablate the refusal direction at all layers and positions. B also gets the diff-of-means over-refused vs matched-compliant direction ablated (a Wang et al. style false-refusal vector, without their orthogonalisation).

Evaluation sets: A = held-out (behaviour, template) cells where the plain behaviour is refused and the wrapped one is complied with at baseline (the "bypass set"), capped at 250 cells by seeded sampling; outcome = refused after intervention ("restore rate"; baseline restore rate is 0 by construction). B = held-out pool prompts refused at baseline; outcome = complied after ("reduction rate"). CIs: bootstrap over behaviours and templates (A) or prompts (B), 2000 resamples.

## Hypotheses and falsifiers

**H1 (A, neurons).** On the held-out bypass set, the top-50 culprit neurons restore refusal at a rate above the 95th percentile of the 20 random-neuron controls AND above the activation-difference top-50, the latter by a paired bootstrap 95% CI for the difference excluding 0. *Falsified if either condition fails.*

**H2 (B, neurons).** Same two conditions for over-refusal reduction at k=50, AND compliance on the XSTest unsafe set rises by at most 2 percentage points (point estimate). *Falsified if any condition fails.*

**H3 (side effects, both modes, k=50 neurons).** Refusal rate on harmless Alpaca rises by under 2 pp and MMLU accuracy falls by under 2 pp (point estimates). With n=200 and 400 the 95% CIs are wide (roughly +-3 to +-7 pp), so passing is weak evidence of no cost; CIs are reported and the verdict is stated on the point estimates, with the CI noted. *Falsified if either point estimate breaches 2 pp.*

**H4 (replication at scale).** H1 and H2 are each met on Qwen2.5-7B-Instruct with the same criteria. Reported separately for each. If the 7B model has fewer than 30 eligible test items in a set (for example almost no over-refusals), that arm is "untestable", not "supported".

**H5 (families share culprits).** Using the community families that have at least 15 attribution pairs on the model:
- *H5a overlap.* Family-specific top-50 neuron sets from family-restricted attribution. J_between = mean off-diagonal Jaccard. J_within = mean Jaccard between sets from two disjoint random halves of one family's pairs (10 splits). Null = Jaccard of two random same-layer-histogram sets (1000 draws). Verdict "shared" if J_between >= 0.5 x J_within and above the null 99th percentile; "family-specific" if J_between < 0.25 x J_within or not above the null; otherwise "partially shared". Same for heads at k=10, exploratory.
- *H5b transfer.* R[F,G] = restore rate on family G's held-out bypass cells (cap 40 per family) when mean-patching family F's top-50 neurons. Control: 5 random same-layer sets per F, each evaluated on every G. q = share of off-diagonal (F,G) cells with R[F,G] above the max of F's 5 random sets on G. rho = mean off-diagonal R / mean diagonal R (families with at least 10 eval cells). Verdict "shared mechanism" if rho >= 0.7 and q >= 0.75; "family-specific" if rho < 0.4 or q < 0.4; otherwise "mixed". The global (all-family) culprit set is included as a reference row.
- Secondary (exploratory): the same on keyword-defined families (a fixed, documented regex classifier in `refusal.py`, first-match priority; crude, so it is exploratory).

**H6 (platform and date).** Platform = the dataset's `platform` field (discord, website, reddit); date bin = created before vs on or after 2023-04-01 (the in-range median region).
- *H6a effectiveness.* Bypass rate (wrapped complied, among cells where the plain behaviour is refused) by platform and by date bin, using all train and test wrapped cells. Test: permutation over templates (10,000) of the largest between-group gap, alpha 0.05 two-sided. Template is the unit.
- *H6b culprit profile.* Spearman rho across all neurons between group attribution vectors, for half-sized random samples of each group's pairs, versus the within-group rho between two disjoint halves (20 splits). "Profile differs" if cross-group rho is below the 5th percentile of the within-group distribution. Groups with fewer than 20 pairs are skipped and listed.

A hypothesis that cannot be tested because a group is too small is reported as "untestable".

## Exploratory (no pass/fail)

Heads at every k; k=10 and 200 for neurons; direction baseline vs units (smallest k at which units match the direction); AtP vs exact-patching correlation; top-10 unit interpretation (top-activating benign wikitext tokens, logit-lens promoted tokens; observations only); keyword families; the 0.5B and 1.5B results.

## Known weaknesses stated in advance

- Phrase-matching labels miss refusals in unusual wording and count partial refusals as refusals. The judge only measures agreement.
- Mean patching changes units at the last prompt position and generated tokens only, not at earlier prompt positions; the Arditi direction baseline follows the original recipe (all positions), so the two are not position-matched.
- Greedy bf16 batched generation is not bit-reproducible; eligibility of a cell can flip with batch composition.
- Test-set noise: with up to 250 cells, a 4 pp difference is about at the noise floor.
- The in-the-wild dataset ends in Dec 2023; families are the dataset authors' community clusters, not a taxonomy of attack mechanism.
- OR-Bench-Hard prompts may include a few genuinely borderline requests.
- 0.5B-scale models refuse inconsistently; the results may not transfer to larger safety-tuned models.
