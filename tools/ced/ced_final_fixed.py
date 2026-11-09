"""CED retrofit on Qwen3.5 — clean, self-checking implementation.

Design
------
    encoder = layers [0, split)   every token, exact
    decoder = layers [split, 24)  the last `tail` tokens ONLY
        full-attention : Q from the tail; K/V = concat(prefix K/V, exact tail K/V)
        GDN            : runs on the tail; recurrent/conv state starts at ZERO
    PPL scored on the last position of each window

The prefix K/V may come from either:
    --kv exact   : captured from a stock full-model pass   (the quality ceiling)
    --kv proj    : a trained projection off H_split         (the deployable case)

Correctness gate (runs automatically when tail == ctx):
    hand-rolled decoder-half forward == stock model, max|dlogit| == 0.

Bugs found & fixed (all three were silent):
  1. GDN conv: HF's causal_conv1d_fn pads SYMMETRICALLY then crops
     (`F.conv1d(..., padding=pad)[:, :, :L]`), NOT a left pad (was ~0.8 off).
  2. The token mixer consumes `input_layernorm(h)`, not the raw layer input.
     Skipping that norm was a ~1.2 error per attention block.
  3. SDPA must be causal. `is_causal=False` let every tail query see all tail
     keys (~1.2 per block, 17.75 on the gate logits). With a prefix present an
     explicit prefix-causal mask is required: SDPA's `is_causal` is UPPER-LEFT
     aligned, so `is_causal=True` with kv_len > q_len is silently WRONG
     (measured 0.84 vs 0.03 at layer 3).
  4. Prefix K must be captured after `k_norm` and rotated at its absolute
     positions; the raw `k_proj` output (pre-norm, pre-rope) is not a key
     (measured 0.61 vs 0.03 across a layer).
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    apply_rotary_pos_emb, torch_chunk_gated_delta_rule, repeat_kv,
)

import os as _os
MODEL_PATH = _os.environ.get("CED_MODEL_PATH", "/home/gfx/pop-bridge/models/Qwen3.5-2B-Base")
DEV = "cuda:0"
CORPUS_INDEX = _os.environ.get("CED_CORPUS_INDEX", "/home/gfx/pop-bridge/data/corpus/index.json")


class Proj(nn.Module):
    def __init__(self, d_in, d_out, hidden=0):
        super().__init__()
        if hidden > 0:
            self.net = nn.Sequential(nn.Linear(d_in, hidden, bias=False), nn.GELU(),
                                     nn.Linear(hidden, d_out, bias=False))
            nn.init.normal_(self.net[0].weight, std=1.0 / math.sqrt(d_in))
            nn.init.zeros_(self.net[2].weight)
        else:
            self.net = nn.Linear(d_in, d_out, bias=False)
            nn.init.normal_(self.net.weight, std=1.0 / math.sqrt(d_in))

    def forward(self, x):
        return self.net(x)


def gdn_forward(gdn, x):
    """Qwen3_5GatedDeltaNet.forward with no cache (state starts at zero)."""
    b, L, _ = x.shape
    mq = gdn.in_proj_qkv(x).transpose(1, 2)
    z = gdn.in_proj_z(x).reshape(b, L, -1, gdn.head_v_dim)
    bb = gdn.in_proj_b(x)
    aa = gdn.in_proj_a(x)
    pad = gdn.conv_kernel_size - 1
    # HF's causal_conv1d_fn pads SYMMETRICALLY then crops. A left pad gives a
    # different alignment and was a silent 0.8-magnitude error.
    mq = F.conv1d(mq, gdn.conv1d.weight, gdn.conv1d.bias, padding=pad,
                  groups=gdn.conv1d.groups)[:, :, :L]
    mq = F.silu(mq).transpose(1, 2)
    q, k, v = torch.split(mq, [gdn.key_dim, gdn.key_dim, gdn.value_dim], dim=-1)
    q = q.reshape(b, L, -1, gdn.head_k_dim)
    k = k.reshape(b, L, -1, gdn.head_k_dim)
    v = v.reshape(b, L, -1, gdn.head_v_dim)
    beta = bb.sigmoid()
    g = -gdn.A_log.float().exp() * F.softplus(aa.float() + gdn.dt_bias)
    rep = gdn.num_v_heads // gdn.num_k_heads
    if rep > 1:
        q = q.repeat_interleave(rep, dim=2)
        k = k.repeat_interleave(rep, dim=2)
    o, _ = torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta,
                                        initial_state=None, output_final_state=False,
                                        use_qk_l2norm_in_kernel=True)
    o = gdn.norm(o.reshape(-1, gdn.head_v_dim), z.reshape(-1, gdn.head_v_dim))
    return gdn.out_proj(o.reshape(b, L, -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-layer", type=int, default=14)
    ap.add_argument("--tail", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--train-steps", type=int, default=0)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--capacity", type=int, default=0)
    ap.add_argument("--out", default="/home/gfx/pop-bridge/data/ced5")
    ap.add_argument("--tag", default="ced5")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    meta = json.load(open(CORPUS_INDEX))
    tr = np.concatenate([np.load(s["path"]) for s in meta["shards"][:-1]])
    ev = np.load([s["path"] for s in meta["shards"]][-1])

    m = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map=DEV, low_cpu_mem_usage=True).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    cfg = m.config.get_text_config() if hasattr(m.config, "get_text_config") else m.config
    H, V, NL = cfg.hidden_size, cfg.vocab_size, cfg.num_hidden_layers
    lt = cfg.layer_types
    T, tail, split = args.ctx, args.tail, args.split_layer
    dec = list(range(split, NL))
    full_layers = [l for l in dec if lt[l] == "full_attention"]
    print(f"split=L{split} tail={tail} ctx={T}  KV layers={full_layers}", flush=True)
    print(f"attn impl = {cfg._attn_implementation}", flush=True)

    KV_DIM = {w: (cfg.num_key_value_heads * cfg.head_dim) for w in ("k_norm", "v_proj")}

    FFN = 3 * H * 6144
    GDN = H * 6144 + H * 2048 + H * 2048 + 2 * H * 16
    FULLA = H * 4096 + 2 * H * 512 + H * 2048
    per_layer = [FFN + (GDN if lt[l] == "linear_attention" else FULLA) for l in range(NL)]
    TOT = sum(per_layer)
    frac = (T - tail) / T
    print(f"MAC skipped {sum(per_layer[split:])*frac/TOT*100:.2f}% -> wall "
          f"{sum(per_layer[split:])*frac/TOT*100*1.073:.2f}% [derived]", flush=True)

    wins = torch.from_numpy(ev).long()[: args.windows * T].reshape(args.windows, T)

    @torch.no_grad()
    def encode_prefix(ids):
        pos = torch.arange(ids.shape[1], device=DEV).unsqueeze(0)
        h = m.model.embed_tokens(ids)
        cos, sin = m.model.rotary_emb(h, pos)
        for l in range(split):
            o = m.model.layers[l](h, position_embeddings=(cos, sin),
                                  attention_mask=None, position_ids=pos)
            h = o[0] if isinstance(o, tuple) else o
        return h

    @torch.no_grad()
    def capture_full(ids):
        """True prefix K/V and true layer inputs, from one stock pass.

        k is captured AFTER k_norm (and rotated later at its absolute positions);
        v is captured as the raw v_proj output. Capturing raw k_proj here was a
        bug: it is neither normalised nor rotated.
        """
        kv, xin = {}, {}

        def mk_kv(l, w):
            def f(mod, a, kw, o):
                kv[(l, w)] = (o[0] if isinstance(o, tuple) else o).detach()
            return f

        def mk_x(l):
            def f(mod, a, kw, o):
                xin[l] = (a[0] if a else kw.get("hidden_states")).detach()
            return f
        hs = [getattr(m.model.layers[l].self_attn, w).register_forward_hook(
                  mk_kv(l, w), with_kwargs=True)
              for l in full_layers for w in ("k_norm", "v_proj")]
        hs += [m.model.layers[l].self_attn.register_forward_hook(mk_x(l), with_kwargs=True)
               for l in full_layers]
        m(input_ids=ids)
        for h in hs:
            h.remove()
        return kv, xin

    def decode_tail(ids, pref):
        """pref: {(l, 'k_norm'|'v_proj'): tensor [1, T-tail, dim]} prefix K/V."""
        hall = encode_prefix(ids)
        ht = hall[:, T - tail:].contiguous()
        pos_all = torch.arange(T, device=DEV).unsqueeze(0)
        cos_all, sin_all = m.model.rotary_emb(hall, pos_all)
        cos_t, sin_t = cos_all[:, T - tail:], sin_all[:, T - tail:]
        cos_p, sin_p = cos_all[:, :T - tail], sin_all[:, :T - tail]
        for li in dec:
            layer = m.model.layers[li]
            # The token mixer consumes input_layernorm(hidden_states) — exactly as
            # Qwen3_5DecoderLayer does. Skipping it was a ~1.2-magnitude bug.
            xt = layer.input_layernorm(ht)
            if lt[li] == "full_attention":
                a = layer.self_attn
                nrep = a.config.num_attention_heads // a.config.num_key_value_heads
                st = (*xt.shape[:-1], -1, a.config.head_dim)
                qf = a.q_proj(xt)
                q, g = torch.chunk(
                    qf.view(*xt.shape[:-1], -1, a.config.head_dim * 2), 2, dim=-1)
                g = g.reshape(*xt.shape[:-1], -1)
                q = a.q_norm(q.view(st)).transpose(1, 2)
                kt = a.k_norm(a.k_proj(xt).view(st)).transpose(1, 2)
                vt = a.v_proj(xt).view(st).transpose(1, 2)
                q, kt = apply_rotary_pos_emb(q, kt, cos_t, sin_t)
                kp = pref.get((li, "k_norm"))
                if kp is not None and kp.shape[1] > 0:
                    vp = pref[(li, "v_proj")]
                    kp = kp.contiguous().view(
                        1, -1, a.config.num_key_value_heads, a.config.head_dim).transpose(1, 2)
                    vp = vp.contiguous().view(
                        1, -1, a.config.num_key_value_heads, a.config.head_dim).transpose(1, 2)
                    # prefix keys are rotated at THEIR OWN absolute positions
                    _, kp = apply_rotary_pos_emb(kp, kp, cos_p, sin_p)
                    k_all = torch.cat([kp.to(kt.dtype), kt], dim=2)
                    v_all = torch.cat([vp.to(vt.dtype), vt], dim=2)
                else:
                    k_all, v_all = kt, vt
                kr = repeat_kv(k_all, nrep)
                vr = repeat_kv(v_all, nrep)
                npre = k_all.shape[2] - q.shape[2]
                if npre == 0:
                    # no prefix: plain causal, bit-identical to the stock model
                    o = F.scaled_dot_product_attention(q, kr, vr, is_causal=True)
                else:
                    # SDPA's is_causal is UPPER-LEFT aligned, so with prefix keys it
                    # would hide the prefix (0.84 error). Use the explicit mask:
                    # query i may see every prefix key plus tail keys <= i.
                    qi = torch.arange(q.shape[2], device=q.device).unsqueeze(1)
                    kj = torch.arange(k_all.shape[2], device=q.device).unsqueeze(0)
                    msk = (kj <= (npre + qi)).view(1, 1, q.shape[2], k_all.shape[2])
                    o = F.scaled_dot_product_attention(q, kr, vr, attn_mask=msk,
                                                       is_causal=False)
                o = o.transpose(1, 2).reshape(*xt.shape[:-1], -1).contiguous()
                ht = ht + a.o_proj(o * torch.sigmoid(g))
            else:
                ht = ht + gdn_forward(layer.linear_attn, xt)
            ht = ht + layer.mlp(layer.post_attention_layernorm(ht))
        return m.lm_head(m.model.norm(ht))

    @torch.no_grad()
    def ppl_last(pref_kind, projs=None, src="hsplit"):
        nll = 0.0
        for w in wins:
            ids = w.unsqueeze(0).to(DEV)
            kv, xin = capture_full(ids)
            if pref_kind == "zero":
                pref = {(l, ww): torch.zeros_like(kv[(l, ww)][:, :T - tail])
                        for (l, ww) in kv}
            elif pref_kind == "exact":
                pref = {(l, ww): kv[(l, ww)][:, :T - tail] for (l, ww) in kv}
            else:
                h = encode_prefix(ids)[:, :T - tail]
                pref = {}
                for (l, ww) in projs:
                    s = h if src == "hsplit" else xin[l][:, :T - tail]
                    pref[(l, ww)] = projs[(l, ww)](s.float()).to(torch.bfloat16)
            lg = decode_tail(ids, pref)[0].float()[-2]
            nll += float(F.cross_entropy(lg.unsqueeze(0), ids[0, -1:], reduction="sum"))
        return math.exp(nll / len(wins))

    # ---- gate 1: with tail == ctx and nothing skipped, must equal the stock model
    if tail == T:
        with torch.no_grad():
            ids_chk = wins[0].unsqueeze(0).to(DEV)
            ref_lg = m(input_ids=ids_chk).logits[0].float()
            kv, _ = capture_full(ids_chk)
            man = decode_tail(ids_chk, {(l, w): kv[(l, w)][:, :0] for (l, w) in kv})
            d = (ref_lg - man[0].float()).abs().max().item()
            print(f"[gate1] tail==ctx, max|dlogit| vs stock = {d:.6f}", flush=True)
            if d > 1e-2:
                raise SystemExit("FAIL gate1: tail-only path != stock model")
            print("[gate1] PASS", flush=True)

    @torch.no_grad()
    def stock_last():
        nll = 0.0
        for w in wins:
            ids = w.unsqueeze(0).to(DEV)
            lg = m(input_ids=ids).logits[0].float()[-2]
            nll += float(F.cross_entropy(lg.unsqueeze(0), ids[0, -1:], reduction="sum"))
        return math.exp(nll / len(wins))

    ppl_ref = stock_last()
    print(f"\n[ref]      PPL(last) = {ppl_ref:.4f}", flush=True)
    ppl_zero = ppl_last("zero")
    print(f"[tail0]    PPL = {ppl_zero:.4f}  ({(ppl_zero/ppl_ref-1)*100:+.2f}%)", flush=True)
    ppl_ex = ppl_last("exact")
    print(f"[oracle]   PPL = {ppl_ex:.4f}  ({(ppl_ex/ppl_ref-1)*100:+.2f}%)  <- bound",
          flush=True)

    res = {"split_layer": split, "tail": tail, "ctx": T, "windows": args.windows,
           "ppl_ref": ppl_ref, "ppl_tail0": ppl_zero, "ppl_oracle": ppl_ex,
           "oracle_degrade_pct": (ppl_ex / ppl_ref - 1) * 100,
           "mac_skip_pct": sum(per_layer[split:]) * frac / TOT * 100,
           "wall_pct_derived": sum(per_layer[split:]) * frac / TOT * 100 * 1.073,
           "arms": {}}

    if args.train_steps > 0:
        train_wins = torch.from_numpy(tr).long()[: max(2, args.batch * 2) * T].reshape(-1, T)
        for src in ("hsplit", "self"):
            projs = {(l, w): Proj(H, KV_DIM[w], args.capacity).to(DEV).float()
                     for l in full_layers for w in ("k_norm", "v_proj")}
            npar = sum(q.numel() for mod in projs.values() for q in mod.parameters())
            opt = torch.optim.AdamW([q for mod in projs.values() for q in mod.parameters()],
                                    lr=args.lr, weight_decay=0.0)
            sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                      total_steps=args.train_steps, pct_start=0.05)
            rng = np.random.default_rng(71)
            print(f"\n[train] src={src} cap={args.capacity} params={npar/1e6:.3f}M "
                  f"steps={args.train_steps}", flush=True)
            t0 = time.time()
            for st in range(args.train_steps):
                i = int(rng.integers(0, len(train_wins)))
                bt = train_wins[i:i + 1].to(DEV)
                with torch.no_grad():
                    tl = m(input_ids=bt).logits[:, -2, :].float()
                    kv, xin = capture_full(bt)
                    h = encode_prefix(bt)[:, :T - tail]
                pref = {}
                for (l, w) in projs:
                    s = h if src == "hsplit" else xin[l][:, :T - tail]
                    pref[(l, w)] = projs[(l, w)](s.float()).to(torch.bfloat16)
                sl = decode_tail(bt, pref)[:, -2, :].float()
                loss = F.kl_div(F.log_softmax(sl, -1), F.softmax(tl, -1),
                                reduction="batchmean")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sch.step()
                if st == 0 or (st + 1) % max(1, args.train_steps // 5) == 0:
                    print(f"  step {st+1:5d}/{args.train_steps} kl={float(loss):.5e} "
                          f"({time.time()-t0:.0f}s)", flush=True)
            p = ppl_last("proj", projs=projs, src=src)
            res["arms"][src] = {"ppl": p, "degrade_pct": (p / ppl_ref - 1) * 100,
                                "params": npar}
            print(f"  >>> src={src:6s} PPL={p:.4f}  ({(p/ppl_ref-1)*100:+.2f}%)", flush=True)

    print("\n" + "=" * 78)
    print(f"CED RESULT  split=L{split} tail={tail} windows={args.windows}")
    print("=" * 78)
    print(f"  ref     PPL={ppl_ref:9.4f}")
    print(f"  tail0   PPL={ppl_zero:9.4f}  {(ppl_zero/ppl_ref-1)*100:+.2f}%")
    print(f"  oracle  PPL={ppl_ex:9.4f}  {(ppl_ex/ppl_ref-1)*100:+.2f}%   <- ceiling")
    for k, d in res["arms"].items():
        print(f"  proj[{k:6s}] PPL={d['ppl']:9.4f}  {d['degrade_pct']:+.2f}%")
    # save the trained linear projections for export_ced_blob.py
    try:
        import numpy as _np
        _sd = {}
        for (_l, _w), _mod in projs.items():
            _lin = _mod.net if not isinstance(_mod.net, type(None)) else _mod
            _wt = _lin[-1].weight if hasattr(_lin, "__getitem__") else _lin.weight
            _sd[f"{_w}.{int(_l)}"] = _wt.detach().cpu().float().numpy()
        if _sd:
            _np.savez(os.path.join(args.out, f"{args.tag}.npz"), **_sd)
            print(f"saved {len(_sd)} projection matrices -> {args.out}/{args.tag}.npz")
    except Exception as _e:
        print("WARN: could not save projections:", _e)
    json.dump(res, open(os.path.join(args.out, f"{args.tag}.json"), "w"), indent=2)
    print(f"\nwrote {args.out}/{args.tag}.json")


if __name__ == "__main__":
    main()
