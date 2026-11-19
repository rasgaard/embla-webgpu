"""Load Sevren's MLX-4bit checkpoint into the reference EmblaForCausalLM (embla_q4 layout).
MLX packs 8 nibbles per little-endian uint32, element 2j in the LOW nibble of byte j: viewing the u32 as uint8
gives exactly the reference's `q4` packing, so no repacking is needed."""
import json, torch
from pathlib import Path
from safetensors import safe_open
from embla_ref.configuration_embla import EmblaConfig
from embla_ref.modeling_embla import EmblaForCausalLM

SKIP = {"architectures","auto_map","model_type","dtype","transformers_version","quantization","quantization_config"}

def load(wdir="models/Embla1-4bit", dtype=torch.float32):
    wdir = Path(wdir)
    raw = json.load(open(wdir / "config.json"))
    cfg = EmblaConfig(**{k: v for k, v in raw.items() if k not in SKIP}, embla_q4={"group_size": 64})
    with torch.device("meta"):
        m = EmblaForCausalLM(cfg)
    sd = {}
    with safe_open(wdir / "model.safetensors", "pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if t.dtype == torch.uint32 or (t.dtype == torch.int32 and k.endswith(".weight") and t.shape[-1] * 8 and False):
                pass
            sd[k] = t
    out = {}
    for k, t in sd.items():
        if k.endswith(".scales") or k.endswith(".biases"):
            out[k] = t.to(dtype)                                   # bf16 -> working dtype (fp32 exact for bf16)
        elif k.endswith(".weight") and t.dtype in (torch.uint32, torch.int32):
            out[k[:-len(".weight")] + ".q4"] = t.view(torch.int32)  # [.., cols/8] u32 words, bit-identical
        else:
            out[k] = t.to(dtype) if t.is_floating_point() else t
    q4 = {k: out.pop(k) for k in [k for k in out if k.endswith(".q4")]}
    missing, unexpected = m.load_state_dict(out, strict=False, assign=True)
    for k, t in q4.items():                       # int32-word buffers don't match the declared byte shape: set directly
        mod = m.get_submodule(k[:-len(".q4")]); mod.q4 = t
    missing = [k for k in missing if not k.endswith(".q4")]
    # buffers the checkpoint lacks must not be left on meta
    return m, cfg, missing, unexpected

def load_engram(m, wdir="models/Embla1-4bit"):
    f = safe_open(Path(wdir) / "engram_table.safetensors", "pt")
    m.engram.host_table = (f.get_tensor("weight").view(torch.uint8), f.get_tensor("scales"), f.get_tensor("biases"))
    return m
