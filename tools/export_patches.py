"""Export-friendly replacements for the four constructs that torch.export / ONNX cannot take as written."""
import math, torch, torch.nn.functional as F
import embla_ref.modeling_embla as M

def moe_forward(self, x):                      # gather top-k experts: no data-dependent loop / argsort / index_add
    B, T, d = x.shape
    x2 = x.reshape(-1, d)
    shared = self.shared_proj(M.relu2(self.shared_fc(x2)))
    idx, gates = self.route(x2)                                   # [N,k]
    w1, w2 = M._stack(self.experts_fc), M._stack(self.experts_proj)
    h = M.relu2(torch.einsum("nd,nkfd->nkf", x2, w1[idx]))
    y = torch.einsum("nkf,nkdf->nkd", h, w2[idx])
    return (shared + (y * gates.unsqueeze(-1).to(y.dtype)).sum(1)).reshape(B, T, d)

def select_keys(qn, kn, topk, recent, chunk=512, position_ids=None):
    """Single-chunk, fixed-k version. Scores padded with -inf up to `topk` columns so k never depends on Tk."""
    B, Tq, H, D = qn.shape
    Tk, Hkv = kn.shape[1], kn.shape[2]
    G = H // Hkv
    qv = qn.view(B, Tq, Hkv, G, D).permute(0, 2, 3, 1, 4)
    kv = kn.permute(0, 2, 1, 3).unsqueeze(2)
    j = torch.arange(Tk)
    tq = Tk - Tq + torch.arange(Tq)
    s = torch.matmul(qv, kv.transpose(-1, -2)).amax(dim=2)        # [B,Hkv,Tq,Tk]
    s = s.masked_fill(~(j[None, :] <= (tq[:, None] - recent)), float("-inf"))
    s = F.pad(s, (0, topk), value=float("-inf"))                  # guarantees >= topk columns
    vals, ind = s.topk(topk, dim=-1, sorted=False)
    ind = torch.where(torch.isinf(vals), -1, ind)
    rec = tq[:, None] - recent + 1 + torch.arange(recent)[None, :]
    rec = torch.where(rec >= 0, rec, -1).expand(B, Hkv, Tq, recent)
    return torch.cat([rec, ind], -1).to(torch.int32)

def sparse_attention(q, k, v, idx, chunk=1024):
    B, Tq, H, D = q.shape
    Hkv = k.shape[2]; G = H // Hkv
    valid = idx >= 0
    ii = idx.clamp_min(0).long()
    bi = torch.arange(B)[:, None, None, None]; gi = torch.arange(Hkv)[None, :, None, None]
    kg = k.permute(0, 2, 1, 3)[bi, gi, ii].float(); vg = v.permute(0, 2, 1, 3)[bi, gi, ii].float()
    qg = q.view(B, Tq, Hkv, G, D).permute(0, 2, 1, 3, 4).float()
    s = torch.einsum("bgthd,bgtkd->bgthk", qg, kg) / math.sqrt(D)
    p = torch.softmax(s.masked_fill(~valid[:, :, :, None, :], float("-inf")), -1)
    return torch.einsum("bgthk,bgtkd->bgthd", p, vg).permute(0, 2, 1, 3, 4).reshape(B, Tq, H, D).to(q.dtype)

_ENGRAM_E = {}
def engram_lookup(self, ids, dtype, position_ids=None, cache=None):
    """Rows come in as a graph input (hashing + table fetch happen on the host / in JS)."""
    e = _ENGRAM_E["e"]; return e.to(dtype)

def apply():
    M.MoE.forward = moe_forward
    M.select_keys = select_keys
    M.sparse_attention = sparse_attention
    M.Engram.lookup = engram_lookup
