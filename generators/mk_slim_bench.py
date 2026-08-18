"""bench_sf_dual.py -> bench_sf_slim.py : hidden state 출력 37 -> 6 적용.

① 그래프 디렉토리 dual_17_256 -> slim_17_256
② 런타임 출력 버퍼 개수를 6개로 (config.num_hidden_layers 를 5로 속인다)
③ extract_context_feature 호출을 직접 concat 으로 대체
   - 원본은 hidden_states[layer_id + 1] 을 읽는다 (offset=1). LID=[1,9,17,25,33] 이면
     실제 인덱스는 [2,10,18,26,34].
   - slim 그래프는 그 5개를 앞에서부터 0..4 로 내보내므로 그냥 0..4 를 이어붙이면 된다.
     LID 를 [-1,0,1,2,3] 으로 두는 방법도 되지만 읽기 어려워서 concat 으로 바꾼다.
④ CMR 은 hs[-2](인덱스 35)가 필요한데 잘려나갔으므로 막는다
"""
import io, sys

src = "/home/work/npu_work/dflash_work/bench_sf_dual.py"
dst = "/home/work/npu_work/dflash_work/bench_sf_slim.py"
txt = io.open(src, encoding="utf-8").read()
n = {}

# ① 그래프 경로
n["dir"] = txt.count("dual_17_256")
txt = txt.replace("dual_17_256", "slim_17_256")

# ② 출력 버퍼 6개
old = "_common = dict("
n["cfg"] = txt.count(old)
txt = txt.replace(old,
                  "_wcfg = _copy.deepcopy(_mcfg); _wcfg.num_hidden_layers = 5   # 출력 6개\n"
                  "_common = dict(", 1)
# model_config=_mcfg 도 부분일치하므로 닫는 괄호까지 포함해서 매칭
n["cfg2"] = txt.count("config=_mcfg)")
txt = txt.replace("config=_mcfg)", "config=_wcfg)", 1)

# ③ 특징 추출 (offset 없이 앞 5개)
n["feat"] = txt.count("extract_context_feature(hs, LID)") + \
            txt.count("extract_context_feature(hs2, LID)")
txt = txt.replace("extract_context_feature(hs, LID)",
                  "torch.cat([hs[i] for i in range(5)], dim=-1)")
txt = txt.replace("extract_context_feature(hs2, LID)",
                  "torch.cat([hs2[i] for i in range(5)], dim=-1)")

# ④ CMR 차단
txt = txt.replace('CMR = int(os.environ.get("CMR", "0"))',
                  'CMR = int(os.environ.get("CMR", "0"))\n'
                  'if CMR:\n'
                  '    raise SystemExit("slim 그래프는 인덱스 35를 안 내보내므로 CMR 불가 "\n'
                  '                     "(KEEP 에 35 추가해서 재컴파일할 것)")', 1)

if n["dir"] != 1 or n["cfg"] != 1 or n["cfg2"] != 1 or n["feat"] != 2:
    print("PATCH_MISS %s" % n); sys.exit(1)

io.open(dst, "w", encoding="utf-8", newline="\n").write(txt)
print("WROTE", dst, n)
