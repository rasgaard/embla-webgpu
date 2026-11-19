"""Export-time replacements that emit ORT's fused 4-bit contrib ops (exact for MLX affine: w = s*(q-8) + c, c = b + 8s)."""
import torch
G = 64
_sym = torch.onnx.ops.symbolic

def linear_fused(x, qb, sc, ct, K, N):
    """y = MatMulNBits(x; q, s)  +  groupsum(x) @ C^T"""
    y1 = _sym("com.microsoft::MatMulNBits", (x, qb, sc), {"K": K, "N": N, "bits": 4, "block_size": G},
              dtype=x.dtype, shape=(*x.shape[:-1], N), version=1)
    xs = x.reshape(*x.shape[:-1], K // G, G).sum(-1)
    return y1 + xs @ ct

def gather_dequant(q, idx, sc, c, rows, cols):
    """Gather experts `idx` [M] from packed uint8 q [E, rows, cols/2], dequantize fused, add the c offset.  -> [M, rows, cols]"""
    w = _sym("com.microsoft::GatherBlockQuantized", (q, idx, sc),
             {"bits": 4, "block_size": G, "gather_axis": 0, "quantize_axis": 2},
             dtype=sc.dtype, shape=(idx.shape[0], rows, cols), version=1)
    cg = c.index_select(0, idx)                                   # [M, rows, cols/G]
    return (w.reshape(idx.shape[0], rows, cols // G, G) + cg.unsqueeze(-1)).reshape(idx.shape[0], rows, cols)

def prepare_linear(mod):
    """Give a Q4Linear the tensors the fused path needs (initializers), and drop what it no longer reads."""
    N, K = mod.out_features, mod.in_features
    mod.qb = mod.q4.view(torch.uint8).reshape(N, K // G, G // 2).contiguous()
    mod.ct = (mod.biases + 8.0 * mod.scales).t().contiguous()
    mod.sc2 = mod.scales.contiguous()
    mod.q4 = torch.zeros(0, dtype=torch.int32); mod.biases = torch.zeros(0); mod.scales = torch.zeros(0)

_ones = {}
def rms_norm_fused(x, normalized_shape=None, weight=None, eps=None):
    """F.rms_norm over the last dim -> one SimplifiedLayerNormalization node (WebGPU EP has it)."""
    d = x.shape[-1]
    if weight is None:
        key = (int(d), x.dtype)
        if key not in _ones: _ones[key] = torch.ones(int(d), dtype=x.dtype)
        weight = _ones[key]
    e = float(eps) if eps is not None else float(torch.finfo(x.dtype).eps)
    return _sym("ai.onnx::RMSNormalization", (x, weight), {"epsilon": e, "axis": -1, "stash_type": 1},
                dtype=x.dtype, shape=x.shape, version=23)
