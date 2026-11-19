"""Stateless one-step Embla graph (prefill chunk or decode token) with the cache as explicit inputs/outputs."""
import torch
import export_patches as P
import embla_ref.modeling_embla as M

def cache_spec(cfg):
    """[(name, kind, layer, which, feature_dims)] for every cache tensor, in a fixed order."""
    D, d = cfg.head_dim, cfg.hidden_size
    spec = []
    for i, t in enumerate(cfg.layer_types):
        if t == "sliding_attention":
            _, hkv, *_ = cfg.attn_spec(i)
            spec += [(f"sw{i}_k", "sliding", i, 0, (hkv, D)), (f"sw{i}_v", "sliding", i, 1, (hkv, D))]
        elif t == "full_attention":
            hkv = cfg.num_key_value_heads
            spec += [(f"fl{i}_k", "full", i, 0, (hkv, D)), (f"fl{i}_kn", "full", i, 1, (hkv, D)), (f"fl{i}_v", "full", i, 2, (hkv, D))]
    pad = (cfg.engram_conv - 1) * max(cfg.engram_orders)
    spec.append(("conv", "conv", -1, 0, (d,)))
    return spec, pad

def expert_spec(cfg):
    """[(input_name, layer, which(0 fc/1 proj), part(q4/scales/biases))] for every routed-expert tensor."""
    return [(f"e{i}_{'fc' if w == 0 else 'proj'}_{part}", i, w, pi)
            for i, t in enumerate(cfg.mlp_layer_types) if t == "sparse"
            for w in (0, 1) for pi, part in enumerate(("q4", "sc", "c"))]

class Step(torch.nn.Module):
    def __init__(self, model, experts_as_inputs=False):
        super().__init__(); self.m = model
        self.spec, self.pad = cache_spec(model.config)
        self.espec = expert_spec(model.config) if experts_as_inputs else []
        for i, b in enumerate(model.transformer.h): b.mlp.lk = i
    def forward(self, ids, engram_rows, *rest):
        import q4_patches
        past, ex = rest[:len(self.spec)], rest[len(self.spec):]
        q4_patches.EXPERT_IN.clear()
        for (nm, i, w, pi), t in zip(self.espec, ex):
            q4_patches.EXPERT_IN.setdefault(i, [[None] * 3, [None] * 3])[w][pi] = t
        cache = M.EmblaCache(); tmp = {}
        for (name, kind, i, w, _), t in zip(self.spec, past):
            if kind == "conv": cache.engram_conv = t
            else: tmp.setdefault((kind, i), [None] * (3 if kind == "full" else 2))[w] = t
        for (kind, i), v in tmp.items(): (cache.sliding if kind == "sliding" else cache.full)[i] = tuple(v)
        full0 = next(i for i, t in enumerate(self.m.config.layer_types) if t == "full_attention")
        cache.seen = cache.full[full0][0].shape[1]             # positions already seen == past length of a full layer
        P._ENGRAM_E["e"] = engram_rows
        logits = self.m(ids, past_key_values=cache, logits_to_keep=1).logits
        out = [logits]
        for (name, kind, i, w, _) in self.spec:
            out.append(cache.engram_conv if kind == "conv" else (cache.sliding if kind == "sliding" else cache.full)[i][w])
        return tuple(out)

def empty_past(cfg, B=1, dtype=torch.float32):
    spec, pad = cache_spec(cfg)
    return [torch.zeros((B, pad) + f, dtype=dtype) if k == "conv" else torch.zeros((B, 0) + f, dtype=dtype) for _, k, _, _, f in spec]
