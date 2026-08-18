"""bench2.py -> bench2_opt.py : lm_head 경로 최적화.

① argmax 전의 .float() 제거. fp16->fp32 는 무손실·단조라 argmax 인덱스가 동일한데
   [1,16,151936] 을 통째로 변환하느라 라운드당 약 36ms 를 쓰고 있었다.
② 호스트 argmax 를 T["lmh"] 타이밍 안으로 넣는다. 지금까지 이 시간이 분해에서 빠져
   라운드 비용이 실제보다 작게 보고됐다.
"""
import io, sys

src = "/home/work/npu_work/dflash_work/bench2.py"
dst = "/home/work/npu_work/dflash_work/bench2_opt.py"
s = io.open(src, encoding="utf-8", newline="").read()

old = ('        s=time.time(); o=torch.as_tensor(lrt(x.contiguous().numpy())); T["lmh"]+=time.time()-s\n'
       '        return torch.argmax(o[:,:n].float(),dim=-1)')
new = ('        s=time.time(); o=torch.as_tensor(lrt(x.contiguous().numpy()))\n'
       '        r=torch.argmax(o[:,:n],dim=-1)          # .float() 불필요: fp16->fp32 는 argmax 결과 불변\n'
       '        T["lmh"]+=time.time()-s                 # 호스트 argmax 도 타이밍에 포함\n'
       '        return r')
if old not in s:
    print("ANCHOR_MISS"); sys.exit(1)
s = s.replace(old, new)
io.open(dst, "w", encoding="utf-8", newline="").write(s)
print("WROTE", dst)
print("float_left =", s.count(".float(),dim=-1"))
