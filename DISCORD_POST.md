Small replication of CounterSteer-style prompt-injection directions, built with an AI coding agent. Qwen2.5 0.5B + 7B, synthetic templated data, 403 held-out injected prompts.

- A probe on the last prompt token predicts "will follow the injection" at AUROC 0.92 (0.5B) / 0.93 (7B) on the 236/237 word-goal test prompts, but the model's own P(canary) gets 0.99, so it doesn't beat that. On 0.5B shuffled-label directions match it.
- Ablating the direction: 0.5B follow 35.5% -> 26.6% with no clean-task cost, but one of 20 control directions (a variance-matched top-PC direction) cut it more (-12.9 pp), so it fails the control test. 7B: no effect (34.7 -> 34.0).
- Adding it to 7B raised following on 258 resisted test prompts to 21.3% vs 1.2-10.1% for 10 norm-matched controls (one run, no significance test, not in the falsifier; fails on 0.5B).

Mostly a negative result. Caveats: two small models from one family, synthetic templated data, one injection family, labels spot-audited by the AI agent rather than a human, bf16 GPU runs not bit-reproducible. Code, raw outputs and a per-layer scan tool (with an HTML heatmap report) are in the repo.
