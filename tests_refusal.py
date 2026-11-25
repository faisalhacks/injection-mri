"""Unit tests for refusal.py on a tiny random Qwen2 model (no download of weights; tokenizer from the HF cache).
  uv run python tests_refusal.py
"""
import numpy as np
import torch

import refusal as R
import mri

mri.guard = lambda: None  # tiny model; the machine-wide RAM guard is for real models


def tiny():
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    tok.padding_side = "left"
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=96, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(cfg).eval()
    m.requires_grad_(False)
    return tok, m


def test_text():
    assert R.wrap("hello [INSERT PROMPT HERE] bye", "X") == "hello X bye"
    assert R.wrap("hello", "X").endswith("\n\nX")
    assert R.is_refusal("I'm sorry, but I can't help with that.")
    assert not R.is_refusal("Sure, here is a poem about spring.")
    assert R.keyword_family("Enter developer mode now") == "developer_mode"
    a, b, c = R.split3([dict(id=str(i)) for i in range(100)])
    assert (len(a), len(b), len(c)) == (50, 20, 30)
    assert R.jaccard([1, 2], [2, 3]) == 1 / 3


def test_patch_roundtrip():
    tok, m = tiny()
    texts = ["Tell me a story about a cat.", "How do I bake bread?"]
    enc = tok([mri.chat(tok, t) for t in texts], return_tensors="pt", padding=True)
    with torch.no_grad():
        base = R.last_logits(m, enc)
        cap = R.capture(tok, m, texts, 2)
    # patching a unit with its own value is a no-op; patching with a different value changes the logits
    L, I = cap["n"].shape[1], cap["n"].shape[2]
    for r in range(2):
        pass
    same = dict(n={1: (torch.tensor([3, 5]), torch.tensor(cap["n"][0, 1, [3, 5]].astype(np.float32)))})
    one = tok([mri.chat(tok, texts[0])], return_tensors="pt")
    with torch.no_grad():
        b0 = R.last_logits(m, one)
        with R.intervene(m, same):
            b1 = R.last_logits(m, one)
        diff = dict(n={1: (torch.tensor([3, 5]), torch.tensor([5.0, -5.0]))})
        with R.intervene(m, diff):
            b2 = R.last_logits(m, one)
    assert torch.allclose(b0, b1, atol=2e-3), (b0 - b1).abs().max()
    assert (b0 - b2).abs().max() > 1e-3
    # head patch no-op
    hh = dict(h={0: (torch.tensor([1]), torch.tensor(cap["h"][0, 0, 16:32][None].astype(np.float32)))})
    with torch.no_grad(), R.intervene(m, hh):
        b3 = R.last_logits(m, one)
    assert torch.allclose(b0, b3, atol=2e-3)


def test_atp_matches_exact_sign():
    tok, m = tiny()
    corrupt = ["Tell me a story about a cat.", "Explain how rain forms in clouds.", "Write a haiku about the sea."]
    clean = ["What is the capital of France?", "Name three primes.", "How do planes fly?"]
    RC = ([tok.encode("I")[0], tok.encode("Sorry")[0]], [tok.encode("Sure")[0], tok.encode("Here")[0]])
    ell = 2
    rvec = np.random.default_rng(0).normal(size=64)
    rvec /= np.linalg.norm(rvec)
    cc = R.capture(tok, m, clean, 3)
    sd = (1.0, 1.0)
    an, ah, Mv, st = R.atp_pass(tok, m, corrupt, cc["n"], cc["h"], RC, ell, rvec, sd, 3)
    units = [("n", l, i) for l in range(3) for i in range(0, 96, 7)] + [("h", l, h) for l in range(3) for h in range(4)]
    eff, base = R.patch_validate(tok, m, corrupt, cc["n"], cc["h"], units, RC, ell, rvec, sd, 16)
    ap = np.array([an[:, l, i].mean() if k == "n" else ah[:, l, i].mean() for k, l, i in units])
    ex = eff.mean(1)
    c = np.corrcoef(ap, ex)[0, 1]
    print("AtP vs exact correlation on tiny random model:", round(float(c), 3))
    assert c > 0.5


def test_stats():
    m = np.array([[1.0, 0.0], [1.0, 1.0], [np.nan, 1.0]])
    est, lo, hi = R.boot2([m], R.f_rate, B=200)
    assert abs(est - 0.8) < 1e-9 and lo <= est <= hi


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
