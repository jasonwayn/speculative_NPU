"""bench_sf_rr.py -> bench_sf_rr_cmr.py : KEEP 에 레이어 35 를 추가한 그래프용.

fused 그래프 출력 순서 (KEEP=2,10,18,26,34,35)
  prefill(256) : [layer2, 10, 18, 26, 34, 35]                -> 6개
  verify(17)   : [layer2, 10, 18, 26, 34, 35, contract]      -> 7개
CMR scorer 는 레이어 35 를 쓴다. 두 경우 모두 인덱스 5 이므로 hs[-2] 대신 hs[5] 로 고정한다.
(원래 코드의 hs[-2] 는 prefill 에서 레이어 26, verify 에서 34 를 가리켜 서로 달랐다.)
"""
import io, sys

src = "/home/work/npu_work/dflash_work/bench_sf_rr.py"
dst = "/home/work/npu_work/dflash_work/bench_sf_rr_cmr.py"
t = io.open(src, encoding="utf-8").read()
n = {}

# ① CMR 차단 가드 제거
i = t.find('    raise SystemExit("slim 그래프는')
if i < 0:
    print("MISS guard"); sys.exit(1)
j = t.find("\n", t.find('"(KEEP 에 35', i))
head = t.rfind("if CMR:", 0, i)
t = t[:head] + "# (CMR 가드 제거: 레이어 35 를 내보내는 그래프를 쓴다)\n" + t[j + 1:]
n["guard"] = 1

# ② 출력 버퍼 개수 +1
old = "_wcfg = _copy.deepcopy(_mcfg); _wcfg.num_hidden_layers = 5   # 출력 6개"
n["w"] = t.count(old)
t = t.replace(old, "_wcfg = _copy.deepcopy(_mcfg); _wcfg.num_hidden_layers = 6   # 출력 7개")
old = "_gcfg = _copy.deepcopy(_mcfg); _gcfg.num_hidden_layers = 5 if _ch == CH_V else 4"
n["g"] = t.count(old)
t = t.replace(old, "_gcfg = _copy.deepcopy(_mcfg); _gcfg.num_hidden_layers = 6 if _ch == CH_V else 5")

# ③ CMR scorer 입력 레이어를 명시 인덱스로
n["hs"] = t.count("hs[-2]") + t.count("hs2[-2]")
t = t.replace("hs[-2]", "hs[5]").replace("hs2[-2]", "hs2[5]")

if n["w"] != 1 or n["g"] != 1 or n["hs"] < 3:
    print("PATCH_MISS %s" % n); sys.exit(1)
io.open(dst, "w", encoding="utf-8", newline="\n").write(t)
print("WROTE %s %s" % (dst, n))
