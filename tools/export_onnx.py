"""Export Embla1-4bit as a stateless one-step ONNX graph (prefill chunk or decode token, cache as explicit inputs / outputs).

fp32 activations; dense layers as com.microsoft MatMulNBits + an offset GEMM (exact for MLX affine 4-bit, see fused.py); the routed
experts are graph *inputs* (streamed to the GPU by src/experts.mjs) so 3.5 GB never pass through the WASM heap. Opset 18, ~1.5 min.

usage: uv run python tools/export_onnx.py build/export/embla_step.onnx
"""
import torch, sys, time
import q4_patches; q4_patches.apply()
import export_patches as P
import fused
import embla_ref.modeling_embla as M
M.select_keys = P.select_keys; M.sparse_attention = P.sparse_attention; M.Engram.lookup = P.engram_lookup
q4_patches.FUSED = True
from loader import load
from graph import Step, empty_past, expert_spec

out = sys.argv[1] if len(sys.argv) > 1 else "build/export/embla_step.onnx"
t0 = time.time()
m, cfg, _, _ = load(dtype=torch.float32); m.eval()
for mod in m.modules():                                  # dense projections -> fused MatMulNBits path
    if isinstance(mod, M.Q4Linear): fused.prepare_linear(mod)
M.Q4Linear.forward = lambda self, x: fused.linear_fused(x, self.qb, self.sc2, self.ct, self.in_features, self.out_features)
step = Step(m, experts_as_inputs=True)
ex = []
for nm, i, w, pi in step.espec:
    mod = m.transformer.h[i].mlp.experts_fc if w == 0 else m.transformer.h[i].mlp.experts_proj
    t = [mod.q4, mod.scales, mod.biases][pi]
    shape, dt = ((*t.shape[:-1], t.shape[-1] * 4), torch.uint8) if pi == 0 else (t.shape, t.dtype)
    ex.append(torch.empty(shape, dtype=dt))                 # shape-only placeholder for tracing
    if pi == 2: mod.q4 = torch.zeros(0, dtype=torch.uint8); mod.scales = torch.zeros(0); mod.biases = torch.zeros(0)
past = empty_past(cfg)
T0 = 4
ids = torch.randint(3, 60000, (1, T0)); rows = torch.randn(1, T0, 16, 128)
args = (ids, rows, *past, *ex)
names = ["ids", "engram_rows"] + [s[0] + "_in" for s in step.spec] + [e[0] for e in step.espec]
onames = ["logits"] + [s[0] + "_out" for s in step.spec]
Tq = torch.export.Dim("T", min=1, max=64)
ds = [{1: Tq}, {1: Tq}]
for (n, k, *_) in step.spec:
    ds.append({} if k == "conv" else {1: torch.export.Dim(f"{n}_len", min=0, max=8192 if k == "full" else 127)})
ds = (ds[0], ds[1], tuple(ds[2:]) + tuple({} for _ in ex))
import pathlib; pathlib.Path(out).parent.mkdir(exist_ok=True, parents=True)
with torch.no_grad():
    torch.onnx.export(step, args, out, dynamo=True, opset_version=18, input_names=names, output_names=onames,
                      dynamic_shapes=ds, external_data=True)
print("EXPORT OK in", round(time.time() - t0), "s")
