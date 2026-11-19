"""Reference vectors for the browser parity check (bench.html): a Danish chat prompt and the 16 greedy tokens the PyTorch reference
(sevren-ai's modeling_embla.py, fp32, real engram table) produces for it.  usage: uv run python tools/make_reference.py"""
import torch, json
from transformers import PreTrainedTokenizerFast
import q4_patches; q4_patches.apply()
from loader import load, load_engram
import embla_ref.modeling_embla as M
tok=PreTrainedTokenizerFast.from_pretrained("models/Embla1-4bit")
ids=tok.apply_chat_template([{"role":"user","content":"Hvad er hovedstaden i Danmark? Svar kort."}],add_generation_prompt=True,return_tensors="pt",return_dict=True)["input_ids"]
m,cfg,_,_=load(dtype=torch.float32); load_engram(m); m.eval()
c=M.EmblaCache(); cur=ids; out=[]
with torch.no_grad():
    for _ in range(16):
        n=int(m(cur,past_key_values=c,logits_to_keep=1).logits[0,-1].argmax()); out.append(n); cur=torch.tensor([[n]])
json.dump({"ids":ids[0].tolist(),"greedy":out,"text":tok.decode(out)},open("test/reference.json","w")); print(out, repr(tok.decode(out)))
