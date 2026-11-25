# v2 data sources and licences

Everything below was fetched on 2026-11-25 (commits in parentheses) and recorded by `fetch_v2_hosts.py`, which writes
`data_v2/hosts.jsonl` (296 hand-checked-by-script host documents, each with its source and licence) so a fresh clone
needs no network access. Licences are copied from the repository / dataset card I actually opened; I am not a lawyer and
did not verify upstream-of-upstream terms beyond what is stated.

## Host documents (real text)

| Source | What I used | Licence as found | Count | Notes |
|---|---|---|---|---|
| BIPIA `benchmark/email` (github.com/microsoft/BIPIA, commit a004b69) | email contexts | MIT (repo `LICENSE`) | 56 | 100 available; 44 dropped because the `SUBJECT\|EMAIL_FROM\|DATE\|CONTENT` header could not be parsed. Body truncated to 1,300 chars; e-mail addresses redacted to `[address]`. Tasks: sender, date, summarise. |
| BIPIA `benchmark/table` | table contexts with BIPIA's question and gold answer | MIT (repo); the tables come from WikiTableQuestions (CC BY-SA 4.0, stated by me from memory of the BIPIA paper, not re-verified) | 60 | Only tables < 1,300 chars and 5-12 rows. Task: table QA. |
| BIPIA `benchmark/code` | code snippet + last 3 lines of the traceback | MIT (repo); questions come from StackOverflow (CC BY-SA 4.0, not re-verified) | 70 | Task: explain the error. The StackOverflow answer text in BIPIA was NOT used. |
| `Salesforce/wikitext` `wikitext-2-raw-v1` train (HF commit b08601e) | paragraphs of 450-1,000 chars with >= 3 sentences | CC BY-SA 3.0 (dataset card) | 70 | Tokenisation artefacts (` @-@ `, spaces before punctuation) cleaned by regex. Tasks: summarise, translate first sentence to French, first year. |
| `fancyzhx/amazon_polarity` test (HF commit 9d9c45c) | reviews of 150-500 chars, 20 positive + 20 negative | dataset card says Apache-2.0; the review text itself is (c) the reviewers/Amazon, so treat as research-use | 40 | Task: sentiment. Risk: redistribution of review text may be restricted; if that matters, regenerate with `fetch_v2_hosts.py` instead of shipping `hosts.jsonl`. |
| Hand-written foreign-language paragraphs (es/fr/de/pt) | 16 paragraphs of three sentences | MIT (this repo, written by the repo author) | 16 | Used only as hard negatives and as hosts of a few injections. |

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

I only opened `deepset/prompt-injections` (card: both `cc-by-4.0` and `apache-2.0` appear in its metadata, which is inconsistent);
HackAPrompt, LLMail-Inject, Tensor Trust, InjecAgent and AgentDojo texts were not opened, so no licence is claimed for them.

## Models and tools

- protectai/deberta-v3-base-prompt-injection-v2 (Apache-2.0 on its card, opened 2026-11-25): used as an off-the-shelf baseline.
- Qwen2.5-0.5B-Instruct and Qwen2.5-7B-Instruct (as in v1).

## Amendment A1 (2026-11-26): real-world attack text (see PREREGISTRATION_V2.md, Amendment A1)

| Source | Status | Licence as found | Use |
|---|---|---|---|
| LLMail-Inject (Microsoft), `microsoft/llmail-inject-challenge` on Hugging Face, commit 1063bdf, file `data/labelled_unique_submissions_phase1.json` | **used** | MIT (dataset card metadata) | 150 attack e-mails that triggered the send-email API, verbatim; e-mail addresses redacted to `[address]`. Family `llmail_real`. |
| BIPIA text attacks (`benchmark/text_attack_{train,test}.json`, commit a004b69) | **used** | MIT (repo) | all 150 attack requests verbatim, family `bipia_real`. The code attacks were not used (they need a code judge). |
| HackAPrompt, `hackaprompt/hackaprompt-dataset` | **not used** | card says MIT, but the files are gated (HF login + acceptance of conditions + sharing HF profile info); I did not accept conditions on the user's behalf | none. If the user obtains access, phrasings can be added as a further held-out family. |
| Tensor Trust data, github.com/HumanCompatibleAI/tensor-trust-data (commit 747a75e) | **not used** | no LICENSE file, README states none, GitHub shows none | I cloned it to look, did not use any text. Needs the maintainers' permission or a stated licence. |
| Gray Swan / UK AISI Agent Red Teaming benchmark (arXiv 2507.20526) | **not used** | the arXiv abstract page states no licence for the benchmark and gives no hosting location; `GraySwanAI/agent_red_teaming` and `gray-swan/agent_red_teaming` do not exist on Hugging Face (repository not found) | none; I did not find a public, research-licensed release. |

No scraping of social platforms was done. The 12 synthetic families (above) remain the main set; the two real families are a separate supplementary set evaluated out-of-family and out-of-goal.
