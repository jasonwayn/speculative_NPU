"""bench_sf_rr_cmr.py -> bench_sf_corpus.py : 실제 코퍼스 프롬프트 입력 경로 추가.

기존 make_prompt 은 (a) gsm8k 문제 이어붙이기 또는 (b) " Background context." 반복
둘 다 합성이다. GPU 쪽 CMR 실험(spec/cmr/run_cmr.py)과 동일한 입력을 쓰기 위해
그쪽에서 뽑아 둔 토큰 id 를 그대로 읽는다.

  CORPUS=pg19|govreport   CORPUS_LEN=1024|2048|4096|8192|16384
"""
import io, sys

src = "/home/work/npu_work/dflash_work/bench_sf_rr_cmr.py"
dst = "/home/work/npu_work/dflash_work/bench_sf_corpus.py"
t = io.open(src, encoding="utf-8").read()

anchor = "def make_prompt(idx, _n=None):"
if t.count(anchor) != 1:
    print("MISS anchor"); sys.exit(1)

ins = '''CORPUS = os.environ.get("CORPUS", "")
CORPUS_LEN = os.environ.get("CORPUS_LEN", "")
_CORPUS_SAMPLES = None
if CORPUS:
    _blob = json.load(open("/home/work/npu_work/dflash_work/corpus_prompts.json"))
    if CORPUS not in _blob:
        raise SystemExit("코퍼스 없음: %s (있는 것: %s)" % (CORPUS, list(_blob)))
    if CORPUS_LEN not in _blob[CORPUS]:
        raise SystemExit("길이 없음: %s (있는 것: %s)" % (CORPUS_LEN, list(_blob[CORPUS])))
    _CORPUS_SAMPLES = _blob[CORPUS][CORPUS_LEN]
    print("corpus=%s len=%s samples=%d 토큰=%s"
          % (CORPUS, CORPUS_LEN, len(_CORPUS_SAMPLES),
             [len(s) for s in _CORPUS_SAMPLES]), flush=True)


'''
t = t.replace(anchor, ins + anchor, 1)

body_anchor = """def make_prompt(idx, _n=None):
    ex = _ds[idx % len(_ds)]"""
if t.count(body_anchor) != 1:
    print("MISS body"); sys.exit(1)
t = t.replace(body_anchor,
"""def make_prompt(idx, _n=None):
    if _CORPUS_SAMPLES is not None:
        return torch.tensor(_CORPUS_SAMPLES[idx % len(_CORPUS_SAMPLES)],
                            dtype=torch.long).unsqueeze(0)
    ex = _ds[idx % len(_ds)]""", 1)

io.open(dst, "w", encoding="utf-8", newline="\n").write(t)
print("WROTE", dst)
