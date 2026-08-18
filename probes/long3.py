"""3단계: 긴 입력이 ATOM+ 에서 실제로 적재/실행되는가. prefill 시간과 hidden 전송량 기록."""
import torch, time, os, subprocess, json
from optimum.rbln import RBLNQwen3ForCausalLM
from transformers import AutoTokenizer
D="/home/work/npu_work/dflash_work/rbln-Qwen3-4B-h-c64-16k"
SRC="/home/work/npu_work/eagle_test/Qwen3-4B"

def devmem():
    try:
        j=json.loads(subprocess.run(["rbln-stat","--json"],capture_output=True,text=True,timeout=5).stdout)
        return j["devices"][0].get("memory_used") or j["devices"][0].get("mem_used")
    except Exception as e: return "n/a"
def rss():
    return int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0])//1024

tok=AutoTokenizer.from_pretrained(SRC)
txt=open("/home/work/npu_work/dflash_work/pg1342.txt",encoding="utf-8").read()
full=tok(txt,return_tensors="pt").input_ids
print(f"corpus tokens = {full.shape[1]}",flush=True)
print(f"rss_before_load = {rss()} MB   dev_mem = {devmem()}",flush=True)

t=time.time(); m=RBLNQwen3ForCausalLM.from_pretrained(D,export=False,rbln_device=0)
print(f"LOADED {time.time()-t:.1f}s  rss={rss()} MB  dev_mem={devmem()}",flush=True)
print(f"max_seq_len = {m.rbln_config.max_seq_len}",flush=True)

for L in [4096, 8192, 16000]:
    ids=full[:, :L].contiguous()
    pad = 1 if L % 64 == 0 else 0          # RBLN 함정 #1: chunk 배수면 크래시
    if pad: ids = torch.cat([ids, ids[:, -1:]], dim=1)
    try:
        torch_t=time.time()
        out=m(input_ids=ids,attention_mask=torch.ones_like(ids))
        dt=time.time()-torch_t
        hs=tuple(h[:, :L] for h in out.hidden_states) if pad else out.hidden_states
        nbytes=sum(h.numel()*h.element_size() for h in hs)/1e9
        print(f"L={L:6d}  prefill={dt:6.2f}s  hidden={len(hs)}x{tuple(hs[-1].shape)}  "
              f"transfer={nbytes:.2f}GB  rss={rss()}MB  dev={devmem()}",flush=True)
        del out, hs
    except Exception as e:
        print(f"L={L:6d}  FAIL {type(e).__name__}: {str(e)[:220]}",flush=True)
print("LONG3_DONE",flush=True)
