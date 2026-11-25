injection-mri v2 (goal-agnostic this time), short and honest.

v1 problem: my probe only beat bag-of-words, and the one thing that beat it needs the attacker's target word. v2 drops that: 312 real host docs (BIPIA, Wikipedia, reviews), 12 attack families x 9 goals (word, URL, language switch, secret leak, fake tool call, refuse, false fact, ALL CAPS, ask for password), 7 hard-negative types, pre-registered before any run.

What held up (Qwen2.5-7B only):
- A residual probe predicts "the model will follow an instruction from the data" on held-out goals (AUROC 0.906) and held-out attack families (0.929), +0.055 / +0.070 over the best goal-agnostic baseline (a simple Attention Tracker, 0.851 / 0.860). The H1 margin barely clears the +0.05 bar I set.

What did not:
- 0.5B follows only 9.5% of injections here, so the held-out tests were untestable on it.
- Injected vs hard-negative documents: a TF-IDF classifier beats the probe (FPR 0.0 vs 0.006 at TPR 0.9). Rejected on both models. Partly my templated attacks.
- Blame map (delete each sentence, watch the probe): finds the planted sentence 95% (7B) but plain attention finds it 100%. Rejected. Deleting the top-blamed sentence does stop the following 96% of the time on 7B, so it is useful, just not better than attention.
- Real LLMail-Inject / BIPIA attacks: only 18 of 299 were followed by 7B; probe 0.80 vs 0.72, CI on the difference touches zero. Weak evidence.
- 7B-only support, one model family, heuristic checkers, probe works late in the network.

Code, data sources and licences, pre-registration and every number: branch v2 of injection-mri (not pushed yet). Built with an AI coding agent.
