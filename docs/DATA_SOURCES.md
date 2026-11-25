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
