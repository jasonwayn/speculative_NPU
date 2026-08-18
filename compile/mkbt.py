"""bench2.py 의 block_tables 하드코딩을 모델 설정에서 유도하도록 수정.

flash_attn 은 max_seq_len/kvcache_partition_len 만큼 블록을 쓰므로 [0] 고정이면
"block_tables (shape=(1,)) has a shape different to required shape (2,)" 로 죽는다.
"""
import io, sys

src = "/home/work/npu_work/dflash_work/bench2.py"
dst = "/home/work/npu_work/dflash_work/bench2_bt.py"
s = io.open(src, encoding="utf-8", newline="").read()

old = 'BT=torch.tensor([0],dtype=torch.int16)'
new = ('BT=torch.arange(int(getattr(m.rbln_config,"kvcache_num_blocks",1) or 1),dtype=torch.int16)  '
       '# flash_attn 은 블록이 여러 개')
if old not in s:
    print("ANCHOR_MISS"); sys.exit(1)
s = s.replace(old, new)
# BT 는 m 이 만들어진 뒤에 정의돼야 한다 -- 원본 순서 확인
i_m = s.index("m=RBLNQwen3ForCausalLM.from_pretrained")
i_bt = s.index("BT=torch.arange")
if i_bt < i_m:
    print("ORDER_BAD: BT defined before model"); sys.exit(1)
s = s.replace('print("compiled "', 'print("blocks="+str(BT.numel())+" compiled "', 1)
io.open(dst, "w", encoding="utf-8", newline="").write(s)
print("WROTE", dst)
