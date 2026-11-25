# v2 data sources and licences

**This repository does not ship third-party text.** The host documents and the real attack strings are downloaded from their
original sources at pinned revisions by `fetch_v2_data.py` (which runs `fetch_v2_hosts.py` and `fetch_v2_real_attacks.py`) and
checked against `data_v2/CHECKSUMS.sha256`. Data fetched this way stays under each source's own licence, see `DATA_LICENSES.md`.
The 16 hand-written foreign-language hosts and all 12 synthetic attack families are in `data_v2.py` (MIT, written by the repo author).
Licences below were read from the repository or dataset card I actually opened. I am not a lawyer and did not verify
upstream-of-upstream terms beyond what is stated. Everything was first fetched on 2026-11-25 / 2026-11-26 (revisions in parentheses).

Host documents total 312 = 296 fetched (56 + 60 + 70 + 70 + 40 in the table) + 16 hand-written. `hosts.jsonl` written by the fetch
script has the 296 fetched ones; `data_v2.load_hosts()` appends the 16 hand-written ones.

## Host documents (real text, fetched, not committed)

| Source | What I used | Licence as found | Count | Notes |
|---|---|---|---|---|
| BIPIA `benchmark/email` (github.com/microsoft/BIPIA, commit a004b69ec0dd) | email contexts | repo `LICENSE`: MIT, "except for the datasets specified below", which lists "Invoices data in OpenAI Evals" (MIT). The BIPIA README says the Email QA data comes from OpenAI Evals, so the e-mails are MIT through OpenAI Evals' licence. The BIPIA README also says the code "should only be used for research on indirect prompt injection attacks"; I use it for that only. | 56 | 100 available; 44 dropped because the `SUBJECT\|EMAIL_FROM\|DATE\|CONTENT` header could not be parsed. Body truncated to 1,300 chars; e-mail addresses redacted to `[address]`. **Personal names, a street address and partial card digits from the upstream data are kept** (names are not redacted; e.g. the sender names that the `sender` task asks for), which is why this text is fetched rather than shipped. Tasks: sender, date, summarise. |
| BIPIA `benchmark/table` | table contexts with BIPIA's question and gold answer | MIT (repo) except WikiTableQuestions, which is CC BY-SA 4.0 (**verified in the BIPIA `LICENSE`, read 2026-11-26**) | 60 | Only tables < 1,300 chars and 5-12 rows. Task: table QA. |
| BIPIA `benchmark/code` | code snippet + last 3 lines of the traceback | MIT (repo) except Stack Exchange / Stack Overflow content, which is CC BY-SA 4.0 (**verified in the BIPIA `LICENSE`**; the licence also says the dataset includes question URLs and author information) | 70 | Task: explain the error. The StackOverflow answer text in BIPIA was NOT used. |
| `Salesforce/wikitext` `wikitext-2-raw-v1` train (HF commit b08601e04326) | paragraphs of 450-1,000 chars with >= 3 sentences | CC BY-SA 3.0 (dataset card) | 70 | Tokenisation artefacts (` @-@ `, spaces before punctuation) cleaned by regex. Tasks: summarise, translate first sentence to French, first year. |
| `fancyzhx/amazon_polarity` test (HF commit 9d9c45c18f8c) | reviews of 150-500 chars, 20 positive + 20 negative | dataset card says Apache-2.0; the review text itself is (c) the reviewers/Amazon, so treat as research-use only | 40 | Task: sentiment. Redistribution of review text may be restricted, which is the second reason the text is fetched, not shipped. |
| Hand-written foreign-language paragraphs (es/fr/de/pt) | 16 paragraphs of three sentences, in `data_v2.py` | MIT (this repo, written by the repo author) | 16 | Used only as hard negatives and as hosts of a few injections. |

Total 312 host documents, 6 task families (summarise, extraction of sender/date/year, table QA, code-error explanation, translation, sentiment).

Sources I considered and did not use: BIPIA `qa` (NewsQA) and `abstract` (XSum) contexts must be rebuilt by the user from
datasets whose licences restrict redistribution (the BIPIA README says so), and I did not rebuild them. The AgentDojo and
InjecAgent host documents were not fetched.

## Attack phrasings

The 12 attack families are my own wording. They follow public attack *categories*, and I did **not** copy text from any of
these datasets; a family name is therefore not a claim of a faithful reproduction:

| Family | Category it is patterned on |
|---|---|
| plain command | "ignore previous instructions" (deepset/prompt-injections, Apache-2.0 / CC-BY-4.0 on its card; HackAPrompt; BIPIA text attacks, MIT) |
| fake system/admin notice | Tensor Trust and HackAPrompt style role spoofing; LLMail-Inject style "system" claims |
| role-play / persona | "DAN"-type persona prompts (jailbreak literature) |
| delimiter / format escape | closing quotes and fake "END OF DOCUMENT" markers (HackAPrompt, Tensor Trust) |
| fake chat turns | fake `User:` / `Assistant:` turns |
| completion attack | pre-written fake answer, then a new task |
| authority / urgency | social-engineering framing |
| hidden in comments / footnote | HTML and markdown comments, hidden spans (InjecAgent and LLMail-Inject style) |
| multilingual | Spanish / German / French versions of the plain command (deepset/prompt-injections contains German) |
| encoded / obfuscated | base64, leetspeak, spaced letters |
| payload splitting | the instruction split over two sentences in two places |
| polite / indirect | "it would be helpful if the assistant ..." |

For the 12 synthetic families I opened only `deepset/prompt-injections` (card: both `cc-by-4.0` and `apache-2.0` appear in its
metadata, which is inconsistent), and no text from any dataset was copied into them. LLMail-Inject, the BIPIA text attacks,
HackAPrompt, Tensor Trust and Gray Swan were examined later, for Amendment A1; see the table below. (An earlier version of
this file said LLMail-Inject and Tensor Trust "were not opened"; that sentence was stale and contradicted the A1 section.)

## Models and tools

- protectai/deberta-v3-base-prompt-injection-v2 (Apache-2.0 on its card, opened 2026-11-25): used as an off-the-shelf baseline.
- Qwen2.5-0.5B-Instruct and Qwen2.5-7B-Instruct (as in v1).

## Amendment A1 (2026-11-26): real-world attack text (see PREREGISTRATION_V2.md, Amendment A1)

| Source | Status | Licence as found | Use |
|---|---|---|---|
| LLMail-Inject (Microsoft), `microsoft/llmail-inject-challenge` on Hugging Face, commit 1063bdf01ec8, file `data/labelled_unique_submissions_phase1.json` | **used** (fetched, not committed) | MIT (dataset card metadata; I read the first 100k characters of the card and saw no redistribution restriction). Participants' free text can contain personal data, which is one reason it is not shipped. | 150 attack e-mails that triggered the send-email API, verbatim; e-mail addresses redacted to `[address]` by regex. Family `llmail_real`. |
| BIPIA text attacks (`benchmark/text_attack_{train,test}.json`, commit a004b69ec0dd) | **used** (fetched, not committed) | MIT (repo `LICENSE`; research-use note in the README, see above) | all 150 attack requests verbatim, family `bipia_real`. One of them contains an e-mail address; the redaction regex is now applied to these strings too (the models saw it unredacted, see README). The code attacks were not used (they need a code judge). |
| HackAPrompt, `hackaprompt/hackaprompt-dataset` | **not used** | card says MIT, but the files are gated (HF login + acceptance of conditions + sharing HF profile info); I did not accept conditions on the user's behalf | none. If the user obtains access, phrasings can be added as a further held-out family. |
| Tensor Trust data, github.com/HumanCompatibleAI/tensor-trust-data (commit 747a75e) | **cloned to look, not used** | no LICENSE file, README states none, GitHub shows none | I cloned it to look, did not use any text. Needs the maintainers' permission or a stated licence. |
| Gray Swan / UK AISI Agent Red Teaming benchmark (arXiv 2507.20526) | **not used** | the arXiv abstract page states no licence for the benchmark and gives no hosting location; `GraySwanAI/agent_red_teaming` and `gray-swan/agent_red_teaming` do not exist on Hugging Face (repository not found) | none; I did not find a public, research-licensed release. |

No scraping of social platforms was done. The 12 synthetic families (above) remain the main set; the two real families are a separate supplementary set evaluated out-of-family and out-of-goal.

## What is and is not in git

Committed: code, the pre-registration, `data_v2/CHECKSUMS.sha256`, per-model `results.json`, `labels.jsonl` (ids and labels, no
model output), `audit_sample.csv` (ids and labels, no text), probes (`probe_v2.npz`), plots, `results_v2/prompts_sha256.json`.
Not committed (gitignored): `data_v2/hosts.jsonl`, `data_v2/real_attacks.jsonl`, `results_v2/*/prompts.jsonl`,
`results_v2/*/real/prompts.jsonl`, `results_v2/*/generations.jsonl`, `results_v2/*/real/generations.jsonl`,
`results_v2/*/audit_sample_full.csv`, the `gen/` activation arrays. How to rebuild them is in the README.
