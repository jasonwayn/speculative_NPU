# speculative_NPU

DFlash(block-diffusion speculative decoding)를 **Rebellions ATOM+ NPU** 로 포팅하고
정적 그래프 제약 아래에서 최적화한 작업 기록입니다.

측정은 전부 **NPU 1장 단독** 실행입니다. prefill부터 decoding까지 target/drafter 모두
같은 카드에서 돕니다.

---

## 1. 환경

| 항목 | 값 |
|---|---|
| NPU | Rebellions ATOM+ (RBLN-CA22) × 4, 카드당 15.7 GiB |
| 소프트웨어 | `rebel` / `optimum-rbln` / `rebel-compiler` 0.10.2, torch 2.9.1+cpu, Python 3.13 |
| 타깃 | Qwen3-4B (fp16), `max_seq_len` 4096 |
| 드래프터 | DFlash 5레이어, 블록 크기 B=16 |
| 측정 실측치 | weight streaming 약 225 GB/s, dispatch 약 61 µs, idle 17–18 W / load 48–69 W |

---

## 2. 결과

`NSAMP=20`, `MAXNEW=2048`, 카드 0 단독.

### gsm8k

| | 원본 | +드래프터 KV캐시 | +비정렬 재개 | **+이중 그래프** |
|---|---|---|---|---|
| draft | 15.7 ms | 5.4 | 5.2 | **5.3** |
| verify | 63.8 | 63.4 | 47.7 | **41.2** |
| lm_head | 9.2 | 9.1 | 8.9 | 8.9 |
| **라운드** | 96.5 ms | 86.5 | 67.3 | **62.3** |
| tau | 4.628 | 5.699 | 5.699 | 5.84 |
| **tok/s** | 47.96 | 65.86 | 84.71 | **93.77** |
| J/token | 1.236 | 0.917 | 0.732 | **0.611** |

### math500

| | 원본 | +드래프터 KV캐시 | +비정렬 재개 | **+이중 그래프** |
|---|---|---|---|---|
| draft | 23.6 ms | 5.7 | 5.8 | **5.6** |
| verify | 82.8 | 63.2 | 50.3 | **42.7** |
| **라운드** | 126.1 ms | 86.2 | 73.4 | **64.2** |
| tau | 3.732 | 3.81 | 3.81 | 4.295 |
| **tok/s** | 29.59 | 44.21 | 51.93 | **66.9** |
| J/token | 1.866 | 1.389 | 1.165 | **0.855** |

**총 1.96× (gsm8k) / 2.26× (math500), 토큰당 에너지는 절반.**

> ⚠️ tok/s 는 tau 변화를 포함합니다. 하드웨어 이득만 보려면 **라운드 시간** 열을 보세요.
> 자세한 내용은 [docs/RESULTS.md](docs/RESULTS.md) 의 "tau 교란" 절 참고.

---

## 3. 최종 구조 — 3단계 정적 그래프

```
[1] target prefill      input_ids [1, 256]        샘플당 1회
[2] drafter             Append th [1,16,12800] -> 캐시 기록
                        Block  noise [1,16,2560] -> 15개 제안
[3] target verify       input_ids [1, 17]         라운드마다, 16토큰 + 패딩 1
```

- **[1] 과 [3] 은 청크 크기가 다르지만 같은 KV 캐시를 공유합니다.** (`cos 0.999982` 검증)
- 드래프터는 Append/Block 두 그래프가 자체 KV 캐시를 공유합니다.
- KV 캐시는 target `[1,8,4096,128] × 72`, drafter `× 10` 로 디바이스 상주.

---

## 4. 적용한 최적화 4가지

| | 내용 | 효과 |
|---|---|---|
| ① | **드래프터 KV 캐시 상주** — 컨텍스트를 매 라운드 재전송하지 않음 | draft 15.7 → 5.4 ms (16K 컨텍스트에선 158.9 → 13.6 ms) |
| ② | **비정렬 재개** — 청크 배수가 아닌 오프셋에서 재개, 꼬리 재계산 제거 | verify 63.4 → 47.7 ms |
| ③ | **verify 청크 64 → 17** — 패딩 48칸 → 1칸 | verify 47.7 → 41.2 ms |
| ④ | **prefill 청크 64 → 256** — ③과 캐시 공유 | prefill 호출 수 1/4 |

②③④ 는 모두 **벤더가 파이썬 레벨에 걸어둔 방어적 검사**였고, 커널은 처음부터
지원하고 있었습니다. 상세는 [docs/OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md).

---

## 5. 알아낸 벤더 제약

7가지를 [docs/VENDOR_LIMITS.md](docs/VENDOR_LIMITS.md) 에 정리했습니다. 요약:

1. prefill 입력 길이가 청크 배수여야 한다는 검사 — **파이썬 전용, 우회 가능**
2. 비정렬 prefix caching 금지 가드 — **`use_attention_mask=False` 면 무해, 우회 가능**
3. `logits_to_keep > 1` 미지원
4. argmax 컴파일 불가 / topk 실행 불가
5. `decoder_batch_sizes` 가 작은 배치에서 깨짐 (`block_tables` shape 불일치)
6. prefill 에 배치 차원이 없음
7. 양자화(int4/int8)가 **조용히** 쓰레기를 뱉음 — 캘리브레이션 API 없음

가장 위험했던 건 제약이 아니라 **틀린 결과가 에러 없이 나오는 경우**였습니다.
stateful draft 초기 구현에서 `q_len != k_len` 으로 PAGED attention 을 호출했더니
에러 없이 `cos 0.57` 이 나왔습니다.

---

## 6. 실패로 확인된 것

| 시도 | 결과 |
|---|---|
| flash attention | 5–15% **악화** (partition 8192 ≫ 우리 컨텍스트) |
| 청크 32 / 16 / 8 | 악화 — 호출당 바닥 비용 약 37 ms (가중치 스트리밍) |
| verify 청크 128 / 256 | 악화 |
| `.float()` 제거 | 차이 없음 (초기 주장 철회) |
| int4 / int8 양자화 | 출력 붕괴 |

---

## 7. 측정 규율

같은 서버의 다른 카드에서 실험을 병행하면 **결과가 오염됩니다.**
카드는 독립이지만 호스트(워커당 19.5 GiB RSS / 64 GiB 상한, 8코어, flock 직렬 로딩)를
공유하기 때문입니다. 실제로 그렇게 측정한 humaneval/mbpp 는 verify 가
63.9 → 225–245 ms 로 튀었습니다.

**측정은 카드 1장, 단독으로.**

---

## 8. 디렉토리

| 경로 | 내용 |
|---|---|
| [bench/](bench/) | 벤치 하네스 4단계 (`bench2.py` → `bench_sf_paper.py` → `bench_sf_ua.py` → `bench_sf_dual.py`) |
| [compile/](compile/) | 그래프 컴파일 (`dual_1_256.py` 가 청크 17+256 공유 캐시 컴파일) |
| [generators/](generators/) | 하네스 패치 생성기 — 단계 간 diff 가 여기 담겨 있음 |
| [checks/](checks/) | 수치 등가 검증 (`d17_check.py`, `tau_diag.py`, `verify_stateful.py` …) |
| [run/](run/) | 실행/대기 셸 스크립트 |
| [probes/](probes/) | 탐색용 일회성 프로브 (정리 안 됨, 기록용) |
| [results/](results/) | 원본 측정 결과 JSON |

## 9. 재현

컴파일된 `.rbln` 그래프와 모델 가중치는 저장소에 없습니다 (수 GB).
`compile/` 의 스크립트로 다시 만들어야 합니다.

```bash
# 1) 드래프터 stateful 그래프
python3 compile/build_stateful.py

# 2) verify 17 + prefill 256 공유 캐시 그래프
CHUNKS=17,256 python3 compile/dual_1_256.py

# 3) 최종 벤치
DSET=gsm8k NSAMP=20 MAXNEW=2048 DEV=0 python3 bench/bench_sf_dual.py
```

경로는 스크립트 상단 상수(`SRC`, `TGT`, `D`)에 하드코딩돼 있습니다.
서버 접속 정보는 연구실 내부 문서에 있고 이 저장소에는 **포함하지 않습니다.**
