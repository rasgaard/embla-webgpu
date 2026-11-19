"""Fuse the AttnRes gating stacks and the partial RoPE (graph surgery on a patch_rmsnorm.py output).

1. RoPE (30 sites, sliding layers' q and k; ~11 nodes each: 4 Slice, 4 Mul, 2 Add, Concat) -> one com.microsoft RotaryEmbedding.
   The model rotates the first 64 of 128 dims, half-split, as  [x1 cos + x2 sin, x2 cos - x1 sin]: ORT's non-interleaved form
   with sin negated, so the sin table is -sin. Tables are precomputed for positions 0..MAX_POS-1 exactly as the graph did
   (fp32 position * fp32 inv_freq, then cos/sin) - more accurate than WGSL cos on arguments in the thousands.
2. AttnRes logit stacks. Every mix took the same column range from each source's logits with one Slice + Unsqueeze per source
   and a Concat (2N+1 nodes, N up to ~10 sources). Now each source list is stacked once ([B, T, Q, N]) and a mix is one Slice.
3. Common-subexpression elimination: identical nodes (same op, inputs, attributes; small constants compared by value) are merged.
   Removes e.g. the per-mix re-stacking of the same block outputs and the groupsum(x) shared by q/k/v projections.
4. Dead-node removal.

usage: uv run python tools/patch_fuse_gating_rope.py in/embla_step.onnx models/graph/embla_step.onnx
"""
import collections, hashlib, os, sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

MAX_POS = 8192 + 64                                   # full-layer cache dims go to 8192, + one chunk
src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src, load_external_data=False)
g = m.graph
init = {t.name: t for t in g.initializer}
vi = {v.name: v for v in g.value_info}
uid = iter(range(10**7))
nm = lambda base: f"{base}__f{next(uid)}"
def val(t):
    if t not in init or init[t].data_location == 1: return None
    return numpy_helper.to_array(init[t])
def const(name, arr):
    if name not in init:
        t = numpy_helper.from_array(arr, name); g.initializer.append(t); init[name] = t
    return name
def graph_maps(nodes):
    prod = {o: n for n in nodes for o in n.output}
    cons = collections.defaultdict(list)
    for n in nodes:
        for i in n.input: cons[i].append(n)
    return prod, cons
stats = collections.Counter()

# ---------------------------------------------------------------- 1. RoPE -> RotaryEmbedding
prod, cons = graph_maps(g.node)
cos_n = [n for n in g.node if n.op_type == "Cos"]; sin_n = [n for n in g.node if n.op_type == "Sin"]
assert len(cos_n) == 1 and len(sin_n) == 1 and cos_n[0].input[0] == sin_n[0].input[0]
unsq7 = prod[cos_n[0].input[0]]; mul69 = prod[unsq7.input[0]]                   # freqs = pos[:, None] * inv_freq
inv_name = [i for i in mul69.input if val(i) is not None][0]; inv_freq = val(inv_name).astype(np.float32)
unsq6 = prod[[i for i in mul69.input if i != inv_name][0]]; cast = prod[unsq6.input[0]]
assert cast.op_type == "Cast" and prod[cast.input[0]].op_type == "Range"
positions = cast.input[0]                                                       # int64 [T] = arange(past, past + T)
freqs = np.arange(MAX_POS, dtype=np.float32)[:, None] * inv_freq[None, :]       # fp32 product, as in the graph
cos_t = const("rope_cos_cache", np.cos(freqs.astype(np.float64)).astype(np.float32))
sin_t = const("rope_neg_sin_cache", (-np.sin(freqs.astype(np.float64))).astype(np.float32))
rot_dims, half = 2 * inv_freq.size, inv_freq.size
pos2d = None
add_before = collections.defaultdict(list); rewire = {}
for cat in [n for n in g.node if n.op_type == "Concat" and len(n.input) == 3]:
    rest = prod.get(cat.input[2])
    if rest is None or rest.op_type != "Slice" or val(rest.input[1]) is None or val(rest.input[1]).tolist() != [rot_dims]: continue
    X = rest.input[0]
    dims = [d.dim_value or d.dim_param for d in vi[X].type.tensor_type.shape.dim]
    assert len(dims) == 4 and isinstance(dims[2], int) and dims[3] == 2 * rot_dims, dims
    H, D = dims[2], dims[3]
    if pos2d is None:
        pos2d = nm("rope_pos")
        add_before[cat.name].append(helper.make_node("Unsqueeze", [positions, const("fuse_ax0", np.array([0], np.int64))], [pos2d], name=nm("rope")))
    x3, y3 = nm("rope_in"), nm("rope_out")
    add_before[cat.name] += [
        helper.make_node("Reshape", [X, const("fuse_shape_bt_flat", np.array([0, 0, -1], np.int64))], [x3], name=nm("rope")),
        helper.make_node("RotaryEmbedding", [x3, pos2d, cos_t, sin_t], [y3], domain="com.microsoft", name=nm("rope"),
                         interleaved=0, rotary_embedding_dim=rot_dims, num_heads=H),
        helper.make_node("Reshape", [y3, const(f"fuse_shape_heads{H}", np.array([0, 0, H, D], np.int64))], [cat.output[0]], name=nm("rope")),
    ]
    cat.output[0] = nm("rope_dead")                                              # old chain becomes dead
    stats["rope"] += 1
nodes = []
for n in g.node: nodes += add_before.get(n.name, []); nodes.append(n)

# ---------------------------------------------------------------- 2. AttnRes logit stacks
prod, cons = graph_maps(nodes)
stacks = {}                                         # tuple(sources) -> stacked tensor name
add_before = collections.defaultdict(list)
def stack_of(srcs, before):
    if srcs in stacks: return stacks[srcs]
    parts = []
    for k in range(len(srcs) - 1, 0, -1):           # reuse the longest stacked prefix
        if srcs[:k] in stacks: parts = [stacks[srcs[:k]]]; rest = srcs[k:]; break
    else: rest = srcs
    for s in rest:
        u = nm("ar_u"); add_before[before].append(helper.make_node("Unsqueeze", [s, const("fuse_ax_last", np.array([-1], np.int64))], [u], name=nm("ar")))
        parts.append(u)
    out = nm("ar_stack"); add_before[before].append(helper.make_node("Concat", parts, [out], axis=-1, name=nm("ar")))
    stacks[srcs] = out; return out
for cat in [n for n in nodes if n.op_type == "Concat" and n.attribute[0].i == -1 and len(n.input) >= 2]:
    srcs, rng = [], None
    for i in cat.input:
        u = prod.get(i)
        if u is None or u.op_type != "Unsqueeze" or val(u.input[1]) is None or val(u.input[1]).tolist() != [-1]: break
        s = prod.get(u.input[0])
        if s is None or s.op_type != "Slice" or len(s.input) != 5: break
        a, b, ax, st = (val(x) for x in s.input[1:])
        if any(v is None for v in (a, b, ax, st)) or ax.tolist() != [2] or st.tolist() != [1]: break
        r = (int(a[0]), int(b[0]))
        if rng not in (None, r): break
        rng = r; srcs.append(s.input[0])
    else:
        S = stack_of(tuple(srcs), cat.name)
        add_before[cat.name].append(helper.make_node("Slice", [S, const(f"fuse_i{rng[0]}", np.array([rng[0]], np.int64)), const(f"fuse_i{rng[1]}", np.array([rng[1]], np.int64)),
                                                              const("fuse_ax2", np.array([2], np.int64))], [cat.output[0]], name=nm("ar")))
        cat.output[0] = nm("ar_dead"); stats["attnres mixes"] += 1
new = []
for n in nodes: new += add_before.get(n.name, []); new.append(n)
nodes = new
stats["attnres source stacks"] = len(stacks)

# ---------------------------------------------------------------- 3. CSE
def ckey(name):                                     # small constants by value, everything else by name
    t = init.get(name)
    if t is not None and t.data_location != 1 and len(t.raw_data or b"") + 8 * len(t.int64_data) + 4 * len(t.float_data) <= 4096:
        a = numpy_helper.to_array(t); return ("const", str(a.dtype), a.shape, hashlib.sha1(a.tobytes()).hexdigest())
    return name
alias, seen, out_nodes = {}, {}, []
for n in nodes:
    for k, x in enumerate(n.input):
        if x in alias: n.input[k] = alias[x]
    key = (n.op_type, n.domain, tuple(ckey(i) for i in n.input), tuple(sorted(a.SerializeToString() for a in n.attribute)), len(n.output))
    if key in seen and n.op_type not in ("RandomNormal", "RandomUniform", "Multinomial"):
        for o_new, o_old in zip(n.output, seen[key].output): alias[o_new] = o_old
        stats["cse merged"] += 1; continue
    seen[key] = n; out_nodes.append(n)
outs = {o.name for o in g.output}
for o in g.output: assert o.name not in alias
nodes = out_nodes

# ---------------------------------------------------------------- 4. dead nodes
while True:
    used = {i for n in nodes for i in n.input} | outs
    live = [n for n in nodes if any(o in used for o in n.output)]
    if len(live) == len(nodes): break
    stats["dead nodes removed"] += len(nodes) - len(live); nodes = live
del g.node[:]; g.node.extend(nodes)
used = {i for n in nodes for i in n.input}
dead = [t for t in g.initializer if t.name not in used]
for t in dead: g.initializer.remove(t)
if not any(o.domain == "com.microsoft" for o in m.opset_import): m.opset_import.append(helper.make_opsetid("com.microsoft", 1))

os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
onnx.save(m, dst)
data_dst = os.path.join(os.path.dirname(dst), "embla_step.onnx.data")
if not os.path.exists(data_dst): os.link(os.path.join(os.path.dirname(src), "embla_step.onnx.data"), data_dst)
print(dict(stats), "nodes", len(nodes), "->", dst)
