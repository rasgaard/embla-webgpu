"""Make the reference's packed-4-bit path memory-sane and export-friendly (works for eager fp32 and fp16 graphs)."""
import torch
import embla_ref.modeling_embla as M

EXPERT_IN = {}
FUSED = False

def _unpack_words(w):
    """int32 words (8 nibbles, element 0 in the lowest nibble) -> float32 codes [..., 8*n]; exact float math, no bit ops."""
    hi = torch.div(w, 65536, rounding_mode="floor")
    lo = w - hi * 65536                                   # [0, 65535]
    hi = torch.where(hi < 0, hi + 65536, hi)              # unsigned high half
    f = torch.stack([lo, hi], -1).float()                 # [..., n, 2]  (< 2^16: exact in fp32)
    nib = []
    for j in range(4):
        a = torch.floor(f / float(16 ** j)); nib.append(a - 16.0 * torch.floor(a / 16.0))
    c = torch.stack(nib, -1)                              # [..., n, 2, 4]
    return c.reshape(*w.shape[:-1], w.shape[-1] * 8)

def dequant4(q4, scales, biases, group=64):
    """Nibble unpack in float math; q4 is int32 words (MLX layout) or uint8 bytes. Output dtype == scales dtype."""
    dt = scales.dtype
    if q4.dtype == torch.int32:
        codes = _unpack_words(q4).to(dt)
        g = codes.reshape(*codes.shape[:-1], -1, group)
        return (g * scales.unsqueeze(-1) + biases.unsqueeze(-1)).reshape(codes.shape)
    qf = q4.to(dt)
    hi = torch.floor(qf / 16); lo = qf - 16 * hi
    codes = torch.stack([lo, hi], -1).reshape(*q4.shape[:-1], q4.shape[-1] * 2)
    g = codes.reshape(*codes.shape[:-1], -1, group)
    return (g * scales.unsqueeze(-1) + biases.unsqueeze(-1)).reshape(codes.shape)

def _weight(self):                      # no dequantized-weight cache (it would be tens of GB for the experts)
    return dequant4(self.q4, self.scales, self.biases, self.group)

def moe_forward(self, x):               # gather the chosen experts' packed bytes FIRST, dequantize only those
    B, T, d = x.shape
    x2 = x.reshape(-1, d)
    shared = self.shared_proj(M.relu2(self.shared_fc(x2)))
    idx, gates = self.route(x2)
    def stack(w, which):
        if isinstance(w, torch.Tensor): return w[idx]
        flat = idx.reshape(-1)
        def take(t): return t.index_select(0, flat).reshape(*idx.shape, *t.shape[1:])   # flat Gather: stays on the WebGPU EP
        if getattr(self, "lk", None) in EXPERT_IN:                       # packed tensors supplied as graph inputs (streamed to the GPU by JS)
            q4, sc, bi = EXPERT_IN[self.lk][which]
            if q4.dtype == torch.uint8 and FUSED:                        # (q4 bytes, scales, c = b + 8s): fused GatherBlockQuantized
                import fused
                wq = fused.gather_dequant(q4, flat, sc, bi, q4.shape[1], q4.shape[2] * 2)
                return wq.reshape(*idx.shape, q4.shape[1], q4.shape[2] * 2)
            return dequant4(take(q4), take(sc), take(bi), w.group)
        return dequant4(take(w.q4), take(w.scales), take(w.biases), w.group)
    h = M.relu2(torch.einsum("nd,nkfd->nkf", x2, stack(self.experts_fc, 0).to(x2.dtype)))
    y = torch.einsum("nkf,nkdf->nkd", h, stack(self.experts_proj, 1).to(h.dtype))
    return (shared + (y * gates.unsqueeze(-1).to(y.dtype)).sum(1)).reshape(B, T, d)

def _emb_forward(self, ids):
    flat = ids.reshape(-1)
    r = lambda t: t.index_select(0, flat).reshape(*ids.shape, *t.shape[1:])
    return dequant4(r(self.q4), r(self.scales), r(self.biases), self.group).to(self._dtype())

def apply():
    M.Q4Embedding.forward = _emb_forward
    M.dequant4 = dequant4
    M._Q4.weight = property(_weight)
    M._Q4._dtype = lambda self: self.scales.dtype
    M.MoE.forward = moe_forward
