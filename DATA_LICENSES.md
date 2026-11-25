# Data licences

**Code** in this repository is under the MIT licence (see `LICENSE`).

**Data is not part of this repository.** `fetch_v2_data.py` downloads third-party text from its original sources at pinned
revisions and builds `data_v2/hosts.jsonl` and `data_v2/real_attacks.jsonl` locally. Data fetched or derived this way stays under
each source's own licence and terms; the MIT licence of the code does not apply to it. If you redistribute any file built by
these scripts (including `results_v2/*/prompts.jsonl` and `generations.jsonl`, which contain it), you take on those terms:

| Source (pinned revision in `fetch_v2_data.py`) | Licence / terms as found |
|---|---|
| BIPIA e-mails (from OpenAI Evals) | MIT; BIPIA README: code "should only be used for research on indirect prompt injection attacks"; the e-mails contain personal names, an address and partial card digits |
| BIPIA tables (WikiTableQuestions) | CC BY-SA 4.0, attribution and share-alike |
| BIPIA code questions (Stack Overflow / Stack Exchange) | CC BY-SA 4.0, attribution and share-alike |
| BIPIA text attacks | MIT, same research-use note |
| Wikitext-2 (Wikipedia) | CC BY-SA 3.0, attribution and share-alike |
| Amazon polarity reviews | dataset card Apache-2.0; review text (c) the reviewers/Amazon, research use only |
| LLMail-Inject phase 1 | MIT (dataset card); free text typed by challenge participants may contain personal data |

Files committed here that are mine: the 12 synthetic attack families, the 16 hand-written foreign-language paragraphs and the
checkers (`data_v2.py`), the analysis code, `results.json`, `labels.jsonl`, `audit_sample.csv` (ids and labels only), probes and
plots (MIT). Details and verification notes: `docs/DATA_SOURCES.md`. I am not a lawyer; this is a record of what I found, not
legal advice.
