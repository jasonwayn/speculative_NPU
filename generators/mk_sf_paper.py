"""bench_sf.py 의 프롬프트 소스를 논문 데이터셋으로 교체 (줄 단위 처리).

다중행 앵커는 줄바꿈 차이에 취약하므로 줄 번호/부분문자열로 찾는다.
"""
import io, sys

src = "/home/work/npu_work/dflash_work/bench_sf.py"
dst = "/home/work/npu_work/dflash_work/bench_sf_paper.py"
lines = io.open(src, encoding="utf-8").read().splitlines()

i_full = next((i for i, l in enumerate(lines) if "_full = tok(open(" in l), None)
i_mp = next((i for i, l in enumerate(lines) if l.startswith("def make_prompt(")), None)
i_run = next((i for i, l in enumerate(lines) if l.startswith("def run(")), None)
i_call = next((i for i, l in enumerate(lines) if "ids = make_prompt(" in l), None)
if None in (i_full, i_mp, i_run, i_call):
    print("LOCATE_FAIL", i_full, i_mp, i_run, i_call); sys.exit(1)

new_head = [
    'DSET = os.environ.get("DSET", "gsm8k")',
    '_ds = [json.loads(l) for l in',
    '       open("/home/work/npu_work/dflash_work/dflash/cache/%s.jsonl" % DSET)]',
]
new_mp = [
    'def make_prompt(idx, _n=None):',
    '    ex = _ds[idx % len(_ds)]',
    '    return tok.apply_chat_template(',
    '        [{"role": "user", "content": ex["turns"][0]}],',
    '        add_generation_prompt=True, return_tensors="pt", enable_thinking=False)',
    '',
    '',
]
out = lines[:i_full] + new_head + lines[i_full + 2:i_mp] + new_mp + lines[i_run:]
out = [('        ids = make_prompt(si)' if "ids = make_prompt(" in l else l) for l in out]
out = [l.replace('r = dict(mode="stateful", cmr=CMR,',
                 'r = dict(mode="stateful", dset=DSET, cmr=CMR,') for l in out]
io.open(dst, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print("WROTE", dst, "lines", len(out))
