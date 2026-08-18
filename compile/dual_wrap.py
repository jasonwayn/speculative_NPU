"""두 청크 그래프에 벤더 래퍼(RBLNRuntimeModel)를 씌워 prefill_forward 를 그대로 쓴다.

재구현 없이: 패딩·마스크·청크 루프·cache_position 처리는 전부 벤더 코드가 한다.
래퍼는 rbln_config.prefill_chunk_size 로 마스크를 만들므로 청크별 config 만 주면 된다.
"""
import os, sys, copy, time, torch, rebel
from optimum.rbln.transformers.models.decoderonly.decoderonly_runtime_utils import (
    RBLNRuntimeModel, RBLNPageTableManager)
from optimum.rbln import RBLNQwen3ForCausalLM
from transformers import AutoTokenizer

D = "/home/work/npu_work/dflash_work"
SRC = "/home/work/npu_work/eagle_test/Qwen3-4B"
DEV = int(os.environ.get("DEV", "1"))
CH = [64, 128]

# 기준 RBLN 모델을 디바이스에 올리면 메모리가 모자라므로(9.2GB x 2 > 15.7GB)
# HF 모델은 호스트에만 올려 설정/임베딩만 빌려온다.
from transformers import AutoConfig, AutoModelForCausalLM
mcfg = AutoConfig.from_pretrained(SRC)
hf = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.float32)
cls = RBLNQwen3ForCausalLM
rc, _ = cls.prepare_rbln_config(rbln_config={
    "batch_size": 1, "max_seq_len": 4096, "prefill_chunk_size": CH[0],
    "output_hidden_states": True, "create_runtimes": False})
rc.max_seq_len = 4096
rc = cls._update_rbln_config(preprocessors=None, model=hf, model_config=mcfg, rbln_config=rc)
print("config ok  chunk=%s  max_seq_len=%s" % (rc.prefill_chunk_size, rc.max_seq_len), flush=True)

ptm = RBLNPageTableManager(rc)
dec_attn_mask = torch.zeros(rc.batch_size, 1, 1, rc.max_seq_len, dtype=torch.float32)
common = dict(main_input_name="input_ids", embed_tokens=hf.model.embed_tokens,
              dec_attn_mask=dec_attn_mask, page_table_manager=ptm, config=mcfg)

dec = {}
for ch in CH:
    cmod = rebel.RBLNCompiledModel(os.path.join(D, "dual_graphs", "prefill_%d.rbln" % ch))
    rt = rebel.Runtime(cmod, tensor_type="pt", device=DEV)
    cfg = copy.deepcopy(rc)
    cfg.prefill_chunk_size = ch                 # 래퍼가 이 값으로 causal mask 를 만든다
    dec[ch] = RBLNRuntimeModel(runtime=rt, phase="prefill", batch_size=rc.batch_size,
                               rbln_config=cfg, **common)
    print("wrapped chunk=%d" % ch, flush=True)

tok = AutoTokenizer.from_pretrained(SRC)
full = tok(open(D + "/pg1342.txt", encoding="utf-8").read(), return_tensors="pt").input_ids
BT = torch.tensor([0], dtype=torch.int16)


def pf(ch, seg, off):
    L = seg.shape[1]
    return dec[ch].prefill_forward(
        seg, cache_position=torch.arange(off, off + L, dtype=torch.int32).unsqueeze(0),
        attention_mask=torch.ones(L, dtype=torch.int64),
        batch_idx=0, block_tables=BT, is_external_block_tables=False)


try:
    r = pf(64, full[:, :100].contiguous(), 0)
    print("PF_OK chunk64 L=100  hidden=%d" % len(r.hidden_states), flush=True)
    r2 = pf(128, full[:, 100:180].contiguous(), 100)
    print("PF_OK chunk128 L=80 (offset 100)", flush=True)
except Exception as e:
    print("PF_FAIL %s: %s" % (type(e).__name__, str(e)[:220]), flush=True)

# verify 크기 비용
for ch in CH:
    seg = full[:, :17].contiguous()
    try:
        for _ in range(3): pf(ch, seg, 0)
        ts = []
        for _ in range(7):
            s = time.time(); pf(ch, seg, 0); ts.append((time.time() - s) * 1000)
        ts.sort(); print("chunk%-4d L=17  %.1f ms" % (ch, ts[3]), flush=True)
    except Exception as e:
        print("chunk%-4d TIME_FAIL %s" % (ch, str(e)[:110]), flush=True)
print("WRAP_DONE", flush=True)
