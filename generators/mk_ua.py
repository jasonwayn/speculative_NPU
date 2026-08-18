"""bench_sf_paper.py -> bench_sf_ua.py : 비정렬 재개 적용 (꼬리 재계산 제거).

① prefill_forward 의 "non-multiple of prefill_chunk_size" 가드 제거
   (그 가드가 지키는 코드는 전부 use_attention_mask=True 경로인데 우리는 False)
② verify 에 꼬리를 붙이지 않고 블록만 보낸다. c0 = VL 로 두면 기존 슬라이싱(vl-c0)이
   그대로 0 이 되어 나머지 코드를 안 건드려도 된다.
"""
import io, sys

PRE = '''# --- 비정렬 오프셋 prefix caching 가드 제거 -------------------------------------
import inspect as _insp, textwrap as _tw
import optimum.rbln.transformers.models.decoderonly.decoderonly_runtime_utils as _RU
_M = _RU.RBLNRuntimeModel
_ls = _insp.getsource(_M.prefill_forward).split("\\n")
_o, _i, _rm = [], 0, False
while _i < len(_ls):
    _l = _ls[_i]
    if "prefix_cached_len % self.rbln_config.prefill_chunk_size != 0" in _l:
        _ind = len(_l) - len(_l.lstrip())
        _i += 1
        while _i < len(_ls) and (not _ls[_i].strip() or
                                 len(_ls[_i]) - len(_ls[_i].lstrip()) > _ind):
            _i += 1
        _rm = True
        continue
    _o.append(_l); _i += 1
if not _rm:
    raise RuntimeError("guard not found")
_body = _tw.dedent("\\n".join(_o)).replace("super()", "super(_Base, self)")
_ns = dict(_RU.__dict__); _ns["_Base"] = _M
exec(compile(_body, "<patched>", "exec"), _ns)
_M.prefill_forward = _ns["prefill_forward"]
# -------------------------------------------------------------------------------
'''

src = "/home/work/npu_work/dflash_work/bench_sf_paper.py"
dst = "/home/work/npu_work/dflash_work/bench_sf_ua.py"
lines = io.open(src, encoding="utf-8").read().splitlines()

# ① 프리앰블은 import 들 뒤에 넣는다
k = max(i for i, l in enumerate(lines) if l.startswith("import ") or l.startswith("from "))
lines = lines[:k + 1] + PRE.split("\n") + lines[k + 1:]

# ② 꼬리 제거
n_seg = n_c0 = 0
for i, l in enumerate(lines):
    if "seg = torch.cat([VRB[:, c0:VL], blk], dim=1)" in l:
        ind = len(l) - len(l.lstrip())
        lines[i] = " " * ind + "seg = blk          # 비정렬 재개: 꼬리 없이 블록만"
        n_seg += 1
    elif "vl = VL; c0 = cached" in l:
        ind = len(l) - len(l.lstrip())
        lines[i] = " " * ind + "vl = VL; c0 = VL   # 재개 지점 = 현재 위치 (vl-c0 = 0)"
        n_c0 += 1
if n_seg != 1 or n_c0 != 1:
    print("PATCH_MISS seg=%d c0=%d" % (n_seg, n_c0)); sys.exit(1)

io.open(dst, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
print("WROTE", dst)
