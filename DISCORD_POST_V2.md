injection-mri v2 (goal-agnostic this time), short and honest.

v1 problem: my probe only beat bag-of-words, and the one thing that beat it needs the attacker's target word. v2 drops that: 296 real host docs (BIPIA, Wikipedia, reviews) + 16 hand-written foreign-language ones, 12 synthetic attack families x 9 goals (word, URL, language switch, secret leak, fake tool call, refuse, false fact, ALL CAPS, ask for password), 7 hard-negative types, pre-registered before any run.

What held up (Qwen2.5-7B only, my own synthetic attacks, not real ones):
- A residual probe read at a late layer predicts "the model will follow an instruction from the data" on held-out attack families (AUROC 0.929) and held-out goals (0.906). That beats the best pre-registered baseline (my re-implementation of Attention Tracker, 0.860 / 0.851) by +0.070 / +0.055. The attack-family result is the firmer one. The held-out-goal result clears my +0.05 bar by 0.005, only 4 of 9 goal folds reach +0.05, and dropping either of two folds (caps, creds) puts it below the bar. I hand-checked 70 follow labels (an AI agent, not a human): 4 of 13 positives in the creds fold were not follows, and with those fixed it still passes, by 0.003.
- Caveat that matters: a logistic regression on all the attention-head features, which I added after seeing the results, gets 0.898 / 0.908, only +0.008 / +0.022 behind the probe. Most of the signal is already in attention patterns.

What did not:
- 0.5B follows only 9.5% of injections here. The pre-registration was ambiguous on when to call a test untestable ("< 10% follow rate" vs "fewer than 3 valid folds"). I labelled H1/H2 untestable; under the fold-count reading they would be rejected (the probe is significantly worse than Attention Tracker on 0.5B, paired CI [-0.194, -0.008]). I added that base-rate check after the 0.5B results existed.
- Injected vs hard-negative documents: a TF-IDF classifier beats the probe (FPR 0.0 vs 0.006 at TPR 0.9). Rejected on both models. Partly my templated attacks.
- Blame map (delete each sentence, watch the probe): finds the planted sentence 95% (7B) but plain attention finds it 100%. Rejected. Deleting the top-blamed sentence does stop the following 96% of the time on 7B, so it is useful, just not better than attention.
- Real LLMail-Inject / BIPIA attacks: only 18 of 299 were followed by 7B; probe 0.80 vs 0.72. That technically passes my pre-registered rule (by 0.003) but the CI on the difference includes zero and 15 of the 18 positives use my weakest checker. Not evidence either way.
- One model family, heuristic checkers (audited on 70 of 920 outputs), probe works late in the network, and the repo ships code and results only: the third-party text is fetched from the original sources at pinned revisions (licences in DATA_LICENSES.md).

Code, pre-registration, deviations list and every number: branch v2 of injection-mri (local so far; the rewritten history has not been pushed). Built with an AI coding agent.
