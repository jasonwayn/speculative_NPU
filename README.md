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

### DFlash 논문 워크로드 5종 (최종, NSAMP=20 / MAXNEW=2048 / 카드 1장 단독)

| 데이터셋 | tau | draft | verify | lm_head | **라운드** | **tok/s** | **J/token** |
|---|---|---|---|---|---|---|---|
| gsm8k | 5.840 | 5.3 | 38.8 | 9.0 | **59.5 ms** | **98.12** | **0.593** |
| humaneval | 6.258 | 5.2 | 39.1 | 9.0 | **59.9** | **104.45** | **0.561** |
| math500 | 4.295 | 5.7 | 39.7 | 9.0 | **62.0** | **69.32** | **0.832** |
| mbpp | 5.730 | 5.4 | 39.8 | 9.2 | **62.6** | **91.51** | **0.633** |
| mt-bench | 2.287 | 5.6 | 40.1 | 9.1 | **62.6** | **36.51** | **1.570** |

**라운드 시간이 데이터셋과 무관하게 59~63 ms 로 평평합니다.** tok/s 차이는 전부 tau 차이입니다
(mt-bench 가 느린 건 하드웨어가 아니라 tau 2.287 때문).

### 최적화 단계별 (gsm8k)

| | 원본 | +드래프터 KV캐시 | +비정렬 재개 | +이중 그래프 | **+출력 슬라이싱** |
|---|---|---|---|---|---|
| draft | 15.7 ms | 5.4 | 5.2 | 5.3 | 5.3 |
| verify | 63.8 | 63.4 | 47.7 | 41.2 | **38.8** |
| lm_head | 9.2 | 9.1 | 8.9 | 8.9 | 9.0 |
| **라운드** | 96.5 ms | 86.5 | 67.3 | 62.3 | **59.5** |
| tau | 4.628 | 5.699 | 5.699 | 5.84 | 5.84 |
| **tok/s** | 47.96 | 65.86 | 84.71 | 93.77 | **98.12** |
| J/token | 1.236 | 0.917 | 0.732 | 0.611 | **0.593** |

**라운드 96.5 → 59.5 ms (1.62배), 처리량 2.05배, 토큰당 에너지 2.08배 개선.**

math500 은 라운드 126.1 → 62.0 ms (2.03배), mt-bench 는 89.3 → 62.6 ms (1.43배).

> ⚠️ tok/s 는 tau 변화를 포함합니다. 이중 그래프 단계에서 tau 가 바뀌었으므로
> 하드웨어 이득만 보려면 **라운드 시간** 열을 보세요. [docs/RESULTS.md](docs/RESULTS.md) 참고.

### GPU 대비 (RTX PRO 6000 Blackwell, gsm8k, 배치 1)

| | 처리량 | J/token | 전력 |
|---|---|---|---|
| GPU (2026-08-13, 유휴 측정) | 293.84 tok/s | 0.797 | 235 W |
| NPU 최적화 전 | 63.35 | 0.973 | 58 W |
| **NPU 현재** | **98.12** | **0.593** | **57 W** |

**처리량 격차 4.6배 → 3.0배. 토큰당 에너지는 NPU 가 GPU 를 1.34배 앞섭니다** (역전).

> GPU 수치는 2026-08-13 측정(60샘플)이고 NPU 는 20샘플이라 조건이 완전히 같지는 않습니다.
> 동일 조건 재측정 스크립트가 [run/gpu_wait_run.sh](run/gpu_wait_run.sh) 에 있습니다
> (GPU 가 유휴가 될 때까지 기다렸다 5종을 같은 설정으로 돌립니다).

### 이론 상한

라운드당 9.11 GB 를 읽어야 하고 실측 대역폭이 225 GB/s 이므로 **읽기만 40.5 ms** 입니다.
현재 라운드 59.5 ms 는 그 바닥의 **68%** 지점이고, 완전 최적화 상한은 약 144 tok/s 입니다.

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

## 4. 적용한 최적화 5가지

| | 내용 | 효과 |
|---|---|---|
| ① | **드래프터 KV 캐시 상주** — 컨텍스트를 매 라운드 재전송하지 않음 | draft 15.7 → 5.4 ms (16K 컨텍스트에선 158.9 → 13.6 ms) |
| ② | **비정렬 재개** — 청크 배수가 아닌 오프셋에서 재개, 꼬리 재계산 제거 | verify 63.4 → 47.7 ms |
| ③ | **verify 청크 64 → 17** — 패딩 48칸 → 1칸 | verify 47.7 → 41.2 ms |
| ④ | **prefill 청크 64 → 256** — ③과 캐시 공유 | prefill 호출 수 1/4 |
| ⑤ | **hidden state 출력 37개 → 6개** | verify 41.2 → 38.8 ms |

②③④ 는 모두 **벤더가 파이썬 레벨에 걸어둔 방어적 검사**였고, 커널은 처음부터
지원하고 있었습니다. ⑤ 는 벤더 래퍼가 `output_hidden_states=True` 일 때 37개를
전부 호스트로 내보내던 것을 실제로 쓰는 6개로 줄인 것입니다.
상세는 [docs/OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md).

---

## 5. 알아낸 벤더 제약

9가지를 [docs/VENDOR_LIMITS.md](docs/VENDOR_LIMITS.md) 에 정리했습니다. 요약:

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
| Append 폭 확대 | 미시도 — 초기 컨텍스트 적재가 P/16 회 호출 (남은 개선점) |
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
| [bench/](bench/) | 벤치 하네스 5단계 (`bench2.py` → `bench_sf_paper.py` → `bench_sf_ua.py` → `bench_sf_dual.py` → `bench_sf_slim.py`) |
| [compile/](compile/) | 그래프 컴파일 (`slim_1_256.py` 가 최종: 청크 17+256 공유 캐시 + 출력 6개) |
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
