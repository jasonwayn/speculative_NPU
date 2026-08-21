"""Power 클래스를 다중 카드 합산으로. dev 에 int 도 list 도 받게 해서 하위 호환 유지."""
import io, sys

p = "/home/work/npu_work/dflash_work/bench_cmr_parts.py"
t = io.open(p, encoding="utf-8").read()
if "s.devs =" in t:
    print("already patched")
else:
    old = """    def __init__(s, dev, hz=5): s.dev = dev; s.hz = hz; s.on = False; s.rows = []"""
    new = """    def __init__(s, dev, hz=5):
        # dev 는 int 또는 list. TP 구성에서 카드 1장만 읽으면 에너지가 N배 과소 보고된다.
        s.devs = [dev] if isinstance(dev, int) else list(dict.fromkeys(dev))
        s.dev = s.devs[0]; s.hz = hz; s.on = False; s.rows = []"""
    if t.count(old) != 1:
        print("MISS init"); sys.exit(1)
    t = t.replace(old, new)

    old2 = """                d = j["devices"][s.dev]
                s.rows.append((time.time(),
                               float(str(d["card_power"]).replace("uW", "")) / 1e6,
                               float(str(d.get("temperature", "0C")).replace("C", ""))))"""
    new2 = """                ds = [j["devices"][i] for i in s.devs]
                s.rows.append((time.time(),
                               sum(float(str(d["card_power"]).replace("uW", "")) / 1e6
                                   for d in ds),
                               max(float(str(d.get("temperature", "0C")).replace("C", ""))
                                   for d in ds)))"""
    if t.count(old2) != 1:
        print("MISS loop"); sys.exit(1)
    t = t.replace(old2, new2)
    io.open(p, "w", encoding="utf-8", newline="\n").write(t)
    print("patched bench_cmr_parts.Power")

# 호출부: 타깃 카드 전부 + 드래프트 카드
p = "/home/work/npu_work/dflash_work/bench_sf_rr.py"
t = io.open(p, encoding="utf-8").read()
if "_PWR_DEVS" in t:
    print("bench already patched"); sys.exit(0)
n = t.count("Power(DEV_TARGET)")
if n != 2:
    print("MISS callsite n=%d" % n); sys.exit(1)
t = t.replace("with Power(DEV_TARGET) as p0:",
              "_PWR_DEVS = list(dict.fromkeys(\n"
              "    (TARGET_DEVICES if isinstance(TARGET_DEVICES, list) else [TARGET_DEVICES])\n"
              "    + [DEV_DRAFT]))\n"
              "print(\"power sampling devices: %s\" % _PWR_DEVS, flush=True)\n"
              "with Power(_PWR_DEVS) as p0:", 1)
t = t.replace("with Power(DEV_TARGET) as pw:", "with Power(_PWR_DEVS) as pw:", 1)
io.open(p, "w", encoding="utf-8", newline="\n").write(t)
print("patched bench_sf_rr callsites")
