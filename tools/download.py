"""Download what the export needs (into models/ and tools/embla_ref/, both gitignored), pinned to the revisions this was built with.

  models/Embla1-4bit/   sevren-ai/Embla1-4bit (MLX 4-bit weights, engram table, tokenizer, config), ~5.6 GB
  tools/embla_ref/      modeling_embla.py + configuration_embla.py from sevren-ai/Embla1-bf16 (the PyTorch reference; MIT)

usage: uv run python tools/download.py
"""
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

WEIGHTS = ("sevren-ai/Embla1-4bit", "4d23eb7de21fc5b55c3051021c7550af1fdce0e8")
REFERENCE = ("sevren-ai/Embla1-bf16", "347209cc1a7414f597be80ff9bab7e454d1d030a")

root = Path(__file__).resolve().parent.parent
snapshot_download(WEIGHTS[0], revision=WEIGHTS[1], local_dir=root / "models" / "Embla1-4bit")
ref = root / "tools" / "embla_ref"; ref.mkdir(exist_ok=True); (ref / "__init__.py").touch()
for f in ("modeling_embla.py", "configuration_embla.py"):
    hf_hub_download(REFERENCE[0], f, revision=REFERENCE[1], local_dir=ref)
print("ok:", root / "models" / "Embla1-4bit", ref)
