# Data sources for the refusal module

Checked 2026-11-26 by opening each dataset card (HF) and, where noted, loading it with `datasets`. Licences are as the card states; where a card is ambiguous I say so. **No harmful request text and no model completion is committed to this repo.** The harmful prompts and jailbreak templates are fetched from the public sources at run time and kept in memory; committed files contain only IDs, labels and scores.

| Role | Source | Access | Licence (as stated) | Notes |
|---|---|---|---|---|
| Harmful behaviours (100) + matched benign (100) | `JailbreakBench/JBB-Behaviors` (config `behaviors`, splits `harmful`, `benign`), paper arXiv:2404.01318 | open, loaded OK | MIT | Built from AdvBench, TDC/HarmBench and original behaviours (card). We use column `Goal` only. |
| Jailbreak wrappers | `TrustAIRLab/in-the-wild-jailbreak-prompts`, configs `jailbreak_2023_05_07` (666 rows) and `jailbreak_2023_12_25` (1,405 rows), paper arXiv:2308.03825, repo verazuo/jailbreak_llms | open, loaded OK | MIT (card and repo). README: "research purposes only, any misuse is strictly prohibited"; contains harmful language | Columns used: `prompt`, `platform`, `source`, `date`, community label. The two releases name the label column differently (`community_name` in 05_07, `community` in 12_25); only ~8 to 11 of ~74 communities carry a human-given name, the rest are unnamed (`None`). Dataset-level counts after dedup: 1,558 unique jailbreak prompts, of which 962 are 150..2500 characters. We **only use existing prompts**; none are written or optimised by us. Community names are the dataset authors' own; I do not know how they were assigned beyond the card. |
| (not used) | `jackhhao/jailbreak-classification` | open | Apache-2.0 | Its card says its jailbreak prompts come from verazuo/jailbreak_llms, i.e. the same source as the row above, so it adds nothing but duplicates. Dropped after the in-the-wild dataset was added. |
| Over-refusal, safe-but-risky-looking | XSTest via the HF mirror `Paul/XSTest` (250 safe, 200 unsafe), paper arXiv:2308.01263 | open, loaded OK | cc-by-4.0 | This is a third-party mirror, not the authors' GitHub CSV (I could not fetch the GitHub file in this environment). Counts match the paper's 250/200. The `unsafe` set is used only for side-effect checks. |
| Over-refusal, harder | `bench-llm/or-bench`, config `or-bench-hard-1k` (1,319 rows), paper arXiv:2405.20947 | open, loaded OK | cc-by-4.0 | "Seemingly toxic but safe" per the benchmark's design; the card gives no toxicity caveat. Some prompts may be borderline; the XSTest-unsafe compliance check is the guard. A random subset is used. |
| Harmless instructions, direction baseline and side effects | `yahma/alpaca-cleaned` | open, loaded OK | Card is inconsistent: metadata tag `cc-by-4.0`, text says CC BY-NC 4.0. Treated as the stricter CC BY-NC 4.0 (research use only here). | Rows without an `input` field and under 200 characters. |
| Capability check | `cais/mmlu` (config `all`, test) | open, loaded OK | MIT | Seeded random sample. ARC not used. |
| Neuron interpretation text | `Salesforce/wikitext` (`wikitext-2-raw-v1`) | open | CC BY-SA 4.0 per the card (tags also list cc-by-sa-3.0, gfdl) | Benign public text for top-activating-token views. Only fragments appear in reports. Attribution: the wikitext fragments quoted in `report_*.html` and `summary_*.json` (top-activating tokens and contexts) are excerpts of Wikipedia text (Merity et al., WikiText) and are shared under CC BY-SA 4.0 (see the card above for the other tags). Logit-lens token lists are raw vocabulary tokens, not curated, and can include offensive or NSFW tokens. |
| Refusal judge (audit) | `protectai/distilroberta-base-rejection-v1` | open | Apache-2.0 | Classifies whether a response is a refusal (label 1). Used only to measure agreement with phrase matching. Llama-Guard-3-1B is gated by Meta, not requested. |

## Considered and not used

- **Gray Swan Arena / ART benchmark** (arXiv:2507.20526). The abstract says the ART benchmark is released, but the abstract page names no location or licence, and a web-search summary says the authors plan a private leaderboard. `GraySwanAI` on HF lists "None public yet" for datasets (opened 2026-11-26). Not used.
- **IPI Arena attacks** (`sureheremarv/ipi_arena_attacks`, MIT, 95 rows, arXiv:2603.15714). Public and licensed, but these are Qwen-generated *indirect prompt-injection* strings for agent tool outputs, not wrappers for a direct harmful request, and not shared by people on social platforms. Different threat model; not used for this module. (It could be a family for `injection-mri` proper.) I did not verify that the `sureheremarv` account is Gray Swan's; the paper cites it.
- **HarmBench, WildJailbreak, walledai/XSTest**: gated on HF, no access requested.
- **Scraping X, Reddit or Discord**: not done. The in-the-wild dataset is the authors' own released collection; "social-platform" coverage here means what that dataset contains (Reddit, Discord, websites, up to Dec 2023). It does not contain 2024-2026 jailbreaks.

## Safety handling

- Raw generations: `results_refusal/_private/` (gitignored). Written only so labels can be audited; I did not read harmful completions myself, the audit is phrase-match vs the judge classifier.
- Committed: IDs, refusal labels, scores, neuron/head indices, aggregate rates.
- Reports and CLI output never print a model completion. `mri.py fix` prints REFUSE/COMPLY and a redaction marker.
- Interpretation sections show benign wikitext fragments and logit-lens token strings only.
