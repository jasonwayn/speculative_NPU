#!/bin/bash
cd /home/work/npu_work/dflash_work || exit 1
NCARD=${NCARD:-4}; NSAMP=${NSAMP:-20}; DS=${DS:-gsm8k}
TAG=${TAG:-n$NCARD}; BUCK=${BUCKETS:-2048}; STG=${STAGGER:-10}
TG=${TGTDIR:-/home/work/npu_work/dflash_work/rbln-Qwen3-4B-hidden}
PER=$(( NSAMP / NCARD ))
rm -f ${TAG}_d*.log ${TAG}_pwr.csv
cat > /tmp/pwr_$$.py <<'PYS'
import sys,json,time,subprocess
while True:
    try:
        j=json.loads(subprocess.run(["rbln-stat","--json"],capture_output=True,text=True,timeout=3).stdout)
        p=[float(str(d["card_power"]).replace("uW",""))/1e6 for d in j["devices"]]
        print(str(round(time.time(),3))+","+",".join(str(round(x,2)) for x in p),flush=True)
    except Exception: pass
    time.sleep(0.25)
PYS
python3 /tmp/pwr_$$.py > ${TAG}_pwr.csv 2>/dev/null &
PW=$!
PIDS=""
for i in $(seq 0 $((NCARD-1))); do
  OFF=$(( i * PER ))
  DEV=$i NSAMP=$PER OFFSET=$OFF B=16 MAXNEW=2048 DSETS=$DS BUCKETS=$BUCK \
    NTHREADS=${NTHREADS:-2} OMP_NUM_THREADS=${NTHREADS:-2} PREC=${PREC:-float16} TGTDIR=$TG \
    python3 bench2.py > ${TAG}_d$i.log 2>&1 &
  PIDS="$PIDS $!"
  [ $i -lt $((NCARD-1)) ] && sleep $STG
done
S0=$(date +%s.%N)
k=0
for p in $PIDS; do
  wait $p; echo "EXIT worker$k rc=$?"
  k=$((k+1))
done
S1=$(date +%s.%N)
sleep 1; kill $PW 2>/dev/null; rm -f /tmp/pwr_$$.py
echo "MULTI_WALL $(python3 -c "print(round($S1-$S0,1))")"
grep -h "^BENCH2 " ${TAG}_d*.log 2>/dev/null | cut -c1-400
python3 - <<PY
import csv,statistics
try:
    rows=[r for r in csv.reader(open("${TAG}_pwr.csv")) if len(r)>=5]
    t=[float(r[0]) for r in rows]; tot=[sum(float(x) for x in r[1:5]) for r in rows]
    E=sum(0.5*(tot[i]+tot[i-1])*(t[i]-t[i-1]) for i in range(1,len(rows)))
    print("PWR4 "+str({"n":len(rows),"P_tot_mean":round(statistics.mean(tot),1),
          "P_tot_max":round(max(tot),1),"E_tot_J":round(E,1),"span_s":round(t[-1]-t[0],1)}))
except Exception as e: print("PWR4 ERR",e)
PY
