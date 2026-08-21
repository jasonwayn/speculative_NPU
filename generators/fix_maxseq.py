"""벤치의 max_seq_len 하드코딩(4096)을 MAXC 에 맞춘다.

그래프는 MAXC 로 컴파일해 놓고 런타임 rbln_config 는 4096 이라 8192 이상 프롬프트가 거부됐다.
TARGET_MAX_SEQ 로 따로 줄 수 있게 하되 기본은 MAXC.
"""
import io, sys
p = "/home/work/npu_work/dflash_work/bench_sf_corpus.py"
t = io.open(p, encoding="utf-8").read()
n = {}

n["a"] = t.count('"batch_size": 1, "max_seq_len": 4096, "prefill_chunk_size": CH_V,')
t = t.replace('"batch_size": 1, "max_seq_len": 4096, "prefill_chunk_size": CH_V,',
              '"batch_size": 1, "max_seq_len": TARGET_MAX_SEQ, "prefill_chunk_size": CH_V,')
n["b"] = t.count("_rc.max_seq_len = 4096")
t = t.replace("_rc.max_seq_len = 4096", "_rc.max_seq_len = TARGET_MAX_SEQ")

# MAXC 정의 뒤에 TARGET_MAX_SEQ 추가
anchor = 'MAXC = int(os.environ.get("MAXC", "16384"))'
n["c"] = t.count(anchor)
t = t.replace(anchor, anchor + '\n# 타깃 그래프를 컴파일할 때 쓴 max_seq_len. 기본은 드래프터 캐시와 동일.\n'
                      'TARGET_MAX_SEQ = int(os.environ.get("TARGET_MAX_SEQ", str(MAXC)))', 1)

if n["a"] != 1 or n["b"] != 1 or n["c"] != 1:
    print("MISS %s" % n); sys.exit(1)
io.open(p, "w", encoding="utf-8", newline="\n").write(t)
print("patched %s" % n)
