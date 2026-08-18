"""비정렬 오프셋 prefix caching 검증 (가드 제거, 클래스 내부 재정의).

가드가 지키는 코드는 전부 `if use_attention_mask:` 아래인데 우리 모델은 False 다.
super() 가 zero-arg 로 쓰이므로 반드시 클래스 본문 안에서 재정의해야 한다.
"""
import os, sys, inspect, textwrap, torch
import optimum.rbln.transformers.models.decoderonly.decoderonly_runtime_utils as RU

M = RU.RBLNRuntimeModel
src = inspect.getsource(M.prefill_forward)
lines = src.split("\n")
out, i = [], 0
removed = False
while i < len(lines):
    l = lines[i]
    if "prefix_cached_len % self.rbln_config.prefill_chunk_size != 0" in l:
        ind = len(l) - len(l.lstrip())
        i += 1
        while i < len(lines) and (not lines[i].strip() or
                                  len(lines[i]) - len(lines[i].lstrip()) > ind):
            i += 1
        removed = True
        continue
    out.append(l)
    i += 1
if not removed:
    print("GUARD_NOT_FOUND"); sys.exit(1)

body = textwrap.dedent("\n".join(out))
body = textwrap.indent(body, "    ")
code = "class _P(_Base):\n" + body + "\n"
ns = dict(RU.__dict__)
ns["_Base"] = M
exec(compile(code, "<patched>", "exec"), ns)
M.prefill_forward = ns["_P"].prefill_forward
print("guard removed", flush=True)

from optimum.rbln import RBLNQwen3ForCausalLM
from transformers import AutoTokenizer

D = "/home/work/npu_work/dflash_work"
SRC = "/home/work/npu_work/eagle_test/Qwen3-4B"
m = RBLNQwen3ForCausalLM.from_pretrained(D + "/rbln-Qwen3-4B-h-c64", export=False, rbln_device=1)
pd = m.prefill_decoder
CH = pd.rbln_config.prefill_chunk_size
BT = torch.tensor([0], dtype=torch.int16)
print("use_attention_mask=%s chunk=%d" % (m.rbln_config.use_attention_mask, CH), flush=True)

tok = AutoTokenizer.from_pretrained(SRC)
full = tok(open(D + "/pg1342.txt", encoding="utf-8").read(), return_tensors="pt").input_ids
P, NEW = 100, 16
ids = full[:, :P].contiguous()
nxt = full[:, P:P + NEW].contiguous()


def pre(seg, off):
    L = seg.shape[1]
    pad = 1 if L % CH == 0 else 0
    if pad:
        seg = torch.cat([seg, seg[:, -1:]], dim=1)
    r = pd.prefill_forward(seg, cache_position=torch.arange(off, off + seg.shape[1],
                           dtype=torch.int32).unsqueeze(0),
                           attention_mask=torch.ones(seg.shape[1], dtype=torch.int64),
                           batch_idx=0, block_tables=BT, is_external_block_tables=False)
    hs = r.hidden_states
    return tuple(h[:, :L] for h in hs) if pad else hs


ref = pre(torch.cat([ids, nxt], dim=1), 0)[-1][:, P:P + NEW].float()

pre(ids, 0)
al = (P // CH) * CH
hA = pre(torch.cat([full[:, al:P], nxt], dim=1), al)[-1][:, P - al:P - al + NEW].float()

pre(ids, 0)
ok = True
try:
    hB = pre(nxt, P)[-1][:, :NEW].float()
except Exception as e:
    ok = False
    print("UNALIGNED_FAIL %s: %s" % (type(e).__name__, str(e)[:200]), flush=True)


def cmp(name, a, b):
    c = float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
    print("%-24s cos=%.6f  %s" % (name, c, "MATCH" if c > 0.999 else "MISMATCH"), flush=True)


cmp("aligned(64) vs ref", hA, ref)
if ok:
    cmp("UNALIGNED(100) vs ref", hB, ref)
print("UNALIGN2_DONE", flush=True)
