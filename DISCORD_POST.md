Small replication of CounterSteer-style prompt-injection directions, built with an AI coding agent. Qwen2.5 0.5B + 7B, synthetic templated data, 403 held-out injected prompts.

- A probe on the last prompt token predicts "will follow the injection" at AUROC 0.92 (0.5B) / 0.94 (7B), but the model's own P(canary) gets 0.99, so it doesn't beat that. On 0.5B shuffled-label directions match it.
- Ablating the direction: 0.5B follow 35.5% -> 26.6% with no clean-task cost, but a random high-variance direction did as well. 7B: no effect (34.7 -> 34.0).
- Adding it to 7B did induce following on resisted prompts (21% vs <=10% for controls).

Mostly a negative result. Caveats: small models, synthetic data, one injection family. Code, raw outputs and a per-layer scan tool are in the repo.
