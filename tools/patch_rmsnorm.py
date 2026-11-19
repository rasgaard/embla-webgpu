"""Fuse the exported RMSNorms (Pow -> ReduceMean -> Add eps -> Sqrt -> Reciprocal -> Mul [-> Mul weight]) into one node each.

Once the GPU->CPU syncs are gone (patch_syncs.py), decode time is dominated by per-node cost: ORT's CPU-side dispatch and a GPU
dispatch per node. 185 norms x 6-7 nodes -> 185 nodes.
  --op sln  (default) ORT's SimplifiedLayerNormalization (onnx domain, opset 1 in ORT's registry)
  --op rms  ONNX RMSNormalization (needs the model's default opset raised to 23)

usage: uv run python tools/patch_rmsnorm.py in/embla_step.onnx out/embla_step.onnx [--op sln|rms]
"""
import collections, os, sys
import numpy as np
import onnx
from onnx import helper, numpy_helper

src, dst = sys.argv[1], sys.argv[2]
op = sys.argv[sys.argv.index("--op") + 1] if "--op" in sys.argv else "sln"
m = onnx.load(src, load_external_data=False)
g = m.graph
cons = collections.defaultdict(list)
for n in g.node:
    for i in n.input: cons[i].append(n)
init = {t.name: t for t in g.initializer}
vi = {v.name: v for v in list(g.value_info) + list(g.input)}
val = lambda t: None if t not in init or init[t].data_location == 1 else numpy_helper.to_array(init[t])
def last_dim(x, axis):
    v = vi.get(x)
    if v is not None:
        d = v.type.tensor_type.shape.dim[-1]
        if d.HasField("dim_value"): return d.dim_value
    return {2: 2048, 3: 128}[axis]                    # hidden size / head dim (reduce axis 2 = [B,T,D], 3 = [B,T,H,Dh])

remove, add_before, fused = set(), collections.defaultdict(list), 0
for p in list(g.node):
    if p.op_type != "Pow" or val(p.input[1]) is None or float(val(p.input[1])) != 2.0: continue
    x = p.input[0]
    try:
        (rm,) = cons[p.output[0]]; (ad,) = cons[rm.output[0]]; (sq,) = cons[ad.output[0]]; (rc,) = cons[sq.output[0]]; (mu,) = cons[rc.output[0]]
    except ValueError: continue
    if (rm.op_type, ad.op_type, sq.op_type, rc.op_type, mu.op_type) != ("ReduceMean", "Add", "Sqrt", "Reciprocal", "Mul") or x not in mu.input: continue
    axes = val(rm.input[1]).tolist(); eps = float(val([i for i in ad.input if i != rm.output[0]][0]))
    if len(axes) != 1: continue
    chain = [p, rm, ad, sq, rc, mu]; out = mu.output[0]
    nxt = cons[out]
    scale = None
    if len(nxt) == 1 and nxt[0].op_type == "Mul":
        w = [i for i in nxt[0].input if i != out][0]
        if w in init and len(init[w].dims) == 1:
            scale = w; chain.append(nxt[0]); out = nxt[0].output[0]
    if scale is None:
        D = last_dim(x, axes[0]); scale = f"rmsnorm_ones{D}"
        if scale not in init:
            t = numpy_helper.from_array(np.ones(D, np.float32), scale); g.initializer.append(t); init[scale] = t
    if op == "sln":
        node = helper.make_node("SimplifiedLayerNormalization", [x, scale], [out], name=f"rmsnorm_{fused}", axis=-1, epsilon=eps, stash_type=1)
    else:
        node = helper.make_node("RMSNormalization", [x, scale], [out], name=f"rmsnorm_{fused}", axis=-1, epsilon=eps)
    add_before[p.name].append(node); remove.update(n.name for n in chain); fused += 1

nodes = []
for n in g.node:
    nodes += add_before.get(n.name, [])
    if n.name not in remove: nodes.append(n)
del g.node[:]; g.node.extend(nodes)
if op == "rms":
    for o in m.opset_import:
        if o.domain in ("", "ai.onnx"): o.version = max(o.version, 23)
os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
onnx.save(m, dst)
data_dst = os.path.join(os.path.dirname(dst), "embla_step.onnx.data")
if not os.path.exists(data_dst): os.link(os.path.join(os.path.dirname(src), "embla_step.onnx.data"), data_dst)
print(f"fused {fused} RMSNorms ({op}), nodes {len(nodes)} -> {dst}")
