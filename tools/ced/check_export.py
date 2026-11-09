"""Verify the exported .bin reproduces the GPU-side projected K/V exactly.

Guards the orientation/scale assumption the board loader makes: the blob stores
K as [kv_dim, hidden] C-order (ne[0] = hidden), and the board computes
ggml_mul_mat(P, h) = P @ h (both [hidden, n]). If the export were transposed or
mis-scaled, the board would silently produce garbage; this catches it.
"""
import json, struct, sys
import numpy as np, torch
from transformers import AutoModelForCausalLM

MP = "/home/gfx/pop-bridge/models/Qwen3.5-2B-Base"
CI = "/home/gfx/pop-bridge/data/corpus/index.json"
BIN = sys.argv[1] if len(sys.argv) > 1 else "/home/gfx/pop-bridge/data/ced6/main.bin"
T, S, TAIL = 512, 14, 256
P = T - TAIL

d = open(BIN, "rb").read()
n_cal, hidden, kv, split, blob_tail = struct.unpack_from("<IIIII", d, 8)
ids = list(struct.unpack_from(f"<{n_cal}I", d, 28))
off = 28 + 4 * n_cal
mat = kv * hidden * 2
W = {}
for i, l in enumerate(ids):
    W[(l, "k")] = np.frombuffer(d, "<f2", kv * hidden, off + i * 2 * mat).reshape(kv, hidden)
    W[(l, "v")] = np.frombuffer(d, "<f2", kv * hidden, off + i * 2 * mat + mat).reshape(kv, hidden)
print(f"blob: n_cal={n_cal} hidden={hidden} kv={kv} split={split} ids={ids}")

meta = json.load(open(CI))
ev = np.load([s["path"] for s in meta["shards"]][-1])
ids_w = torch.from_numpy(ev).long()[:T].reshape(1, T).cuda()

m = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.bfloat16, device_map="cuda:0",
                                         low_cpu_mem_usage=True).eval()
for p in m.parameters():
    p.requires_grad_(False)
cfg = m.config.get_text_config() if hasattr(m.config, "get_text_config") else m.config
DEV = next(m.parameters()).device
full = [l for l in range(S, cfg.num_hidden_layers) if cfg.layer_types[l] == "full_attention"]

kv_true = {}
def mk(l, w):
    def f(mod, a, kw, o):
        kv_true[(l, w)] = (o[0] if isinstance(o, tuple) else o).detach()
    return f
hs = [getattr(m.model.layers[l].self_attn, w).register_forward_hook(mk(l, w), with_kwargs=True)
      for l in full for w in ("k_norm", "v_proj")]
with torch.no_grad():
    m(input_ids=ids_w)
for h in hs:
    h.remove()

with torch.no_grad():
    pos = torch.arange(T, device=DEV).unsqueeze(0)
    h = m.model.embed_tokens(ids_w)
    cos, sin = m.model.rotary_emb(h, pos)
    for l in range(S):
        o = m.model.layers[l](h, position_embeddings=(cos, sin), attention_mask=None, position_ids=pos)
        h = o[0] if isinstance(o, tuple) else o
    hpref = h[0, :P].float()            # [P, hidden]

print(f"{'layer':>6} {'arm':>2} {'true rms':>9} {'pred rms':>9} {'corr':>7} {'rel L2':>8}")
ok = True
for l in full:
    for arm, tgt in (("k", kv_true[(l, "k_norm")]), ("v", kv_true[(l, "v_proj")])):
        Y = tgt[0, :P].reshape(P, -1).float()
        pred_ok = hpref.cpu().numpy().astype(np.float32) @ W[(l, arm)].T.astype(np.float32)
        rel = np.linalg.norm(pred_ok - Y.cpu().numpy()) / np.linalg.norm(Y.cpu().numpy())
        a = pred_ok.ravel(); b = Y.cpu().numpy().ravel()
        corr = float(np.corrcoef(a, b)[0, 1])
        print(f"{l:>6} {arm:>2} {np.sqrt((b**2).mean()):9.4f} {np.sqrt((a**2).mean()):9.4f} "
              f"{corr:7.4f} {rel:8.4f}")
        if not (corr > 0.5 and rel < 0.6):
            ok = False
    # contrast: the transposed/mis-shaped reading must be clearly worse
    Yc = kv_true[(l, "k_norm")][0, :P].reshape(P, -1).float().cpu().numpy()
    xb = hpref.cpu().numpy().astype(np.float32)
    Wk = W[(l, "k")].astype(np.float32)
    pred_bad = (Wk @ xb.T).T if Wk.shape == (kv, hidden) and xb.shape[1] == hidden else None
    if pred_bad is not None:
        rel_bad = np.linalg.norm(pred_bad - Yc) / np.linalg.norm(Yc)
        cb = float(np.corrcoef(pred_bad.ravel(), Yc.ravel())[0, 1])
        print(f"{l:>6}  k read as [hidden,kv] instead (wrong way): rel L2 = {rel_bad:.4f} corr = {cb:.4f}")
print("\nORIENTATION CHECK:", "PASS" if ok else "FAIL")
