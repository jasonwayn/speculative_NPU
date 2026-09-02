# optimum-rbln / rebel-compiler 0.10.2 제약

실측으로 확인한 것만 적었습니다. 벤더 문서에 없는 내용입니다.

---

## 1. prefill 입력 폭이 64의 배수여야 한다 — **우회 가능**

`RBLNDecoderOnlyModelForCausalLMConfig.__init__` 의 파이썬 검사일 뿐입니다.
청크 32 / 17 로 컴파일해서 정상 동작을 확인했습니다 (`cos 0.999922` / `0.999982`).
생성자에서 검사만 통과시키고 실제 값을 덮어쓰면 됩니다.

## 2. 비정렬 prefix caching 금지 — **우회 가능**

`decoderonly_runtime_utils.py:551` 의 `NotImplementedError`.
그 가드가 보호하는 코드는 전부 `if self.rbln_config.use_attention_mask:` 안에 있으므로
`use_attention_mask=False` 인 모델에서는 무해합니다. 제거 후 `cos 0.999996`.

## 3. `logits_to_keep > 1` 미지원

마지막 위치 하나의 logits 만 받을 수 있습니다. 여러 위치를 한 번에 검증하려면
hidden state 를 받아서 별도 lm_head 그래프로 돌려야 합니다.

## 4. `topk` / `sort` 는 컴파일러를 죽인다 — argmax 는 **이제 된다**

> **2026-08-24 재측정.** 이 항목의 기존 내용("argmax 컴파일 불가, topk 는 컴파일되나
> 실행 불가")은 **둘 다 틀렸습니다.** 0.10.2 에서 다시 쟀습니다
> ([checks/topk_one.py](../checks/topk_one.py), `N=640 K=32 fp32`).

| op | 결과 |
|---|---|
| `torch.cumsum` | compile OK / run OK |
| `torch.max(...).values` | compile OK / run OK |
| **`torch.argmax`** | **compile OK / run OK, 값 일치** |
| `torch.topk(...).values` | **세그폴트 (코어덤프)** |
| `torch.topk(...).indices` | **세그폴트 (코어덤프)** |
| `topk` 로 k 번째 값만 뽑아 임계 마스크 | **세그폴트 (코어덤프)** |
| `torch.sort(...).indices` | **세그폴트 (코어덤프)** |

**`topk` 은 "미지원" 이 아니라 `librbln.so` 안에서 프로세스가 죽습니다.** 파이썬 예외가
아니라 코어 덤프라 `try/except` 로 못 잡습니다. `sort` 도 같고, **인덱스를 쓰지 않고
k 번째 값으로 마스크만 만드는 우회로도 같이 죽습니다** — `topk` 이 그래프에 들어가는
순간 끝입니다.

`argmax` 가 되므로 **argmax + 마스킹 반복**으로 top-k 를 흉내낼 수는 있습니다
(640 개 대상이면 연산은 사소하지만 그래프에 순차 K 단계가 들어갑니다). 시도해보지
않았습니다 — [CMR_TUNING.md](CMR_TUNING.md) 의 `CMR_ONCE` 로 top-k 가 프리필에
1 회만 돌게 되어 NPU 로 옮길 이유가 없어졌습니다.

## 5. `decoder_batch_sizes` 가 작은 배치에서 깨짐

`decoder_batch_sizes=[8,4,2,1]` 로 컴파일하면 배치 1 호출에서

```
block_tables (shape=(8,1)) has a shape different to required shape (1,1)
```

배치별로 모델을 따로 컴파일해야 합니다.

## 6. prefill 에 배치 차원이 없음

prefill 그래프는 배치 1 고정입니다. 배치 서빙을 하려면 prefill 을 순차로 돌려야 합니다.

## 7. 양자화가 **조용히** 실패

int4 / int8 로 컴파일하면 통과하고 실행도 되는데 토큰 id 0(`'!!!!'`)만 무한히 뱉습니다.
int8 과 int4 결과물이 **바이트 단위로 동일**했습니다.
`rebel` 에도 `optimum-rbln` 에도 캘리브레이션 API 가 없습니다.

> 속도 데이터는 유효합니다(그래프 모양은 가중치 값과 무관). verify 60.8 → 39.0 ms.
> 하지만 출력이 쓰레기라 쓸 수 없습니다.

---

## 8. `mark_static_address` 는 3개 인자 조합이 다 있어야 함 — **문서에 없음**

```python
rebel.compile_from_torch(mod, input_info=..., example_inputs=..., compile_context=...)
```

셋 중 하나라도 빠지면 **에러 없이** 매 호출마다 캐시를 재업로드합니다.
결과는 맞고 속도만 안 나옵니다.

---

## 9. `output_hidden_states=True` 는 37개를 전부 내보냄 — **우회 가능**

벤더 래퍼가 `return logits, all_hidden_states` 로 embedding + 전 레이어를 호스트로
복사합니다. 필요한 것만 고르는 옵션이 없습니다.

래퍼를 한 겹 더 감싸 원하는 인덱스만 반환하고, 런타임에 넘기는 `config` 의
`num_hidden_layers` 를 잘린 개수 - 1 로 맞추면 됩니다 (`_prepare_prefill_outputs` 가
`config.num_hidden_layers + 1` 개 버퍼를 잡습니다).

**비용은 바이트가 아니라 텐서 개수입니다.** 37 → 6 으로 줄였을 때 절감이
verify(3.2 MB) 5.4 ms, prefill(97 MB) 5.1 ms 로 거의 같았습니다 —
출력 텐서 하나당 약 0.17 ms 의 고정 비용이 붙습니다.

---

## 10. `sin` / `cos` 가 인자 크기에 비례해 틀림 — **우회 가능**

절대오차 `약 6e-4 × |x|`. 값역이 [-1,1] 인데 |x|≈4096 에서 오차 1.67, 1e5 이상에서 2.0
(= 참값과 무관). 같은 머신 CPU fp32 는 |x|=1e8 까지 3.3e-08 로 평평합니다 — **최대 6천만 배**.
fp32 자리를 쓰면서 fp16 으로 계산한 것보다 나쁩니다.

범위를 벗어나도 `nan` 이 아니라 **조용히 무관한 값**을 냅니다.

`x` 와 `x+1000·2π` 는 참값이 같은데 오차가 3.3e-04 → 1.94 이므로 범위 축소 문제이고,
인자를 fp16 격자에 미리 맞춰 넣어도 동일하므로 입력 절단은 아닙니다.

**우회**: 호스트에서 2π 나머지만 구해 넘기고 계산은 그래프에 남깁니다. 축소 후 오차는
5e-3 로 줄고 인자 크기와 무관해집니다. 상세는 [ROPE_ROOT_CAUSE.md](ROPE_ROOT_CAUSE.md).

> **일반화**: RBLN 그래프에 큰 인자의 초월함수를 넣지 마세요. DFlash 만의 문제가 아니라
> 이 NPU 를 쓰는 무엇이든 해당됩니다.

---

## 11. KV 캐시를 커스텀 어텐션 op 외의 연산으로 읽으면 짝 그래프 컴파일이 깨진다

> **2026-08-24 정정.** 이 항목의 전제였던 "두 그래프를 `CompileContext` 로 공유해야
> 한다" 는 **우리 선택이 아니라 optimum-rbln 의 내부 경로**였습니다. `CompileContext`,
> `use_weight_sharing`, `use_global_ctx` 는 **공식 문서에 없습니다.** 벤더가 문서화한
> 방법은 **bucketing** 입니다 (§13). 따라서 아래는 "벤더 제약" 이 아니라
> **"문서화되지 않은 내부 경로의 제약"** 으로 읽어야 합니다.

`prefill_17` 과 `prefill_256` 은 `CompileContext(use_weight_sharing=True)` 로 가중치를
공유합니다. 청크 크기가 달라도 되지만, **한쪽 그래프에서 `past_key_values[i][0]` 을
`torch.ops.rbln_custom_ops.paged_*` 가 아닌 연산으로 읽으면 다른 쪽 컴파일이 실패**합니다.

| 구성 | 결과 |
|---|---|
| 두 그래프 다 점수 없음 | 둘 다 `COMPILE_OK` |
| 청크 17 에만 점수 추가 | 17 OK, **256 FAIL** |
| 청크 256 을 먼저 컴파일 | 256 OK, **17 FAIL** |
| `use_weight_sharing=False`, TP1 | 둘 다 OK, **로드에서 `SYS_ENOMEM`** (가중치 2벌 16 GB > 15.7 GiB) |
| `use_weight_sharing=False`, TP4 | **`RBLNCompileError: DEVICE_GRAPH_CONVERSION`** |

순서와 무관하게 "두 번째" 가 죽습니다. 캐시를 아주 조금 읽는 것(`K[0,:,:640,0].mean(0)`)은
통과하므로 접근 자체가 아니라 **가중치 공유 맵이 어긋나는 것**이 원인으로 보입니다.

실용적 결론: **CMR 의 어텐션 점수 계산을 타깃 그래프 안으로 옮길 수 없습니다.**
단, 막는 것은 이 항목이 아니라 **TP4 의 `DEVICE_GRAPH_CONVERSION` 실패**입니다.
그쪽은 `use_weight_sharing=False` 로 **단일 컴파일에서** 터지므로 짝 그래프 구조와
무관하고, bucketing 으로 바꿔도 남습니다.
자세한 조사 기록은 [HOST_THREAD_STARVATION.md](HOST_THREAD_STARVATION.md) §4.1.

## 12. 5차원 브로드캐스트 `matmul` 이 컴파일러를 죽인다 — **우회 가능**

GQA 확장을 브로드캐스트로 표현하면 `IndexError: map::at` 으로 죽습니다.
`rep` 를 쿼리 쪽에 접어 평범한 3D `torch.bmm` 으로 쓰면 통과합니다. 같은 수학입니다.

```python
# 죽는다
kb = K.view(1, nkv, 1, S, hd)
a  = torch.matmul(q.view(1, nkv, rep, L, hd), kb.transpose(-1, -2))

# 된다 (컴파일 96 s)
qg = q.view(1, nkv, rep, L, hd).reshape(nkv, rep * L, hd)
a  = torch.bmm(qg, K.view(nkv, S, hd).transpose(1, 2))
```

크기와 무관합니다 (컨텍스트를 2048 로 줄여도 동일하게 실패). K 를 `torch.clone` 해도
같습니다. **K 가 상수 buffer 일 때는 5D 도 통과**하므로, 격리 테스트로는 재현되지
않습니다 — 실제 그래프 입력으로 시험해야 합니다.

## 13. 여러 입력 shape 은 bucketing 이 정답이다 — **우리가 안 쓰고 있었다**

`rebel.compile_from_torch` 는 `input_info` 에 **설정들의 리스트**를 받아 한 모델이
여러 입력 shape 을 지원하게 합니다 ([공식 튜토리얼](https://docs.rbln.ai/latest/software/api/python/tutorial/advanced/bucketing.html)).

```python
input_infos = [ [("x", [1, 17, H], "float32")],
                [("x", [1, 256, H], "float32")] ]
cm = rebel.compile_from_torch(model, input_info=input_infos)
rt = rebel.Runtime(cm, tensor_type="pt")   # 런타임 하나가 두 shape 을 자동 선택
```

실측 ([checks/bucket_mem.py](../checks/bucket_mem.py), 4096x4096 fp16 가중치 33.5 MB):

| | 디바이스 할당 | 컨텍스트 | 실행기 |
|---|---|---|---|
| 버킷 1 개 (L17) | 44.0 MB | 1 | 1 |
| 버킷 2 개 (L17, L256) | **56.6 MB** | 2 | 2 |
| `CompileContext` 공유 #1 | 44.0 MB | 1 | 1 |
| `CompileContext` 공유 #2 | 12.6 MB | 1 | 1 |
| 공유 방식 합계 | **56.6 MB** | | |

**두 방식의 디바이스 메모리가 동일합니다.** 둘 다 가중치를 한 벌만 올립니다
(56.6 = 가중치 33.5 + L17 활성 10.5 + L256 활성 12.6). 저장 파일은 버킷마다
가중치를 복제하지만(34 -> 69 MB) 디바이스에서는 공유됩니다.

차이는 **버킷이 컴파일 1 회 / 런타임 1 개**라는 점이고, 그래서 §11 의 실패
모드("두 번째 컴파일이 죽는다")가 구조적으로 존재하지 않습니다.

> 우리는 optimum-rbln 이 쓰는 내부 경로(`CompileContext`)를 그대로 따라가느라
> 이 기능을 몰랐습니다. **그래프 구조에 손댈 일이 생기면 bucketing 부터 보세요.**

## 14. 정적 캐시를 공유하는 쓰기/읽기 그래프 쌍은 TP>1 에서 못 쓴다

> **2026-09-02 2차 정정.** 처음엔 "TP 런타임 두 개 불가"(오류 — 타깃이 반례),
> 다음엔 "원인 미상"이었다가, 격리 프로브 두 개로 규칙을 확정했습니다. 결론:
> **생성 순서는 우회 가능하지만 실행 순서는 우회 불가**이고, 드래프터의
> append/block 쌍은 후자에 걸려 TP>1 을 쓸 수 없습니다.

드래프터를 텐서 병렬로 쪼개려는 시도가 여기서 막혔습니다. **각 그래프는 혼자서는
TP4 로 잘 되고, 두 번째 런타임을 만들 때만 죽습니다.**

격리 측정 ([checks/probe_append_tp.py](../checks/probe_append_tp.py), 타깃 미적재):

| | TP1 | TP4 | 결과 |
|---|---|---|---|
| A `AppendRR` 단독 | 0.71 ms | 0.81 ms | OK |
| B `BlockRR` 단독 | 4.63 ms | **1.43 ms** | OK (3.2 배) |
| C 둘 동시, in-process | `rt_a` OK / `rt_b` OK | `rt_a` OK / **`rt_b` INIT_INTERNAL** | TP4 만 실패 |
| D 둘 동시, 저장 후 재로드 | — | `rt_a` OK / **`rt_b` INIT_INTERNAL** | 경로 무관 |

**배제한 가설**

- `AppendRR` 자체 — A 가 TP4 로 통과합니다.
- 디바이스 배치 — 두 런타임을 겹치지 않는 카드에 줘도 같은 지점에서 죽습니다.
- TP 런타임을 두 개 만드는 것 자체 — TP1 에서는 C 가 공존합니다.
- 저장/재로드 경로 — D 가 in-process 와 같은 지점에서 죽습니다.

**확정한 규칙 두 개** ([checks/probe_append_tp3.py](../checks/probe_append_tp3.py),
[checks/probe_append_tp4.py](../checks/probe_append_tp4.py), 타깃 미적재로 재현)

1. **생성 순서**: `append` 를 먼저 만들면 두 번째(`block`) 런타임이
   `INIT_INTERNAL`. **`block` 을 먼저 만들면 둘 다 생성된다.** TP2/TP4 동일.
2. **실행 순서**: 생성이 어떻게 됐든, **`append` 가 한 번이라도 실행된 뒤의
   `block` 실행은 `SYS_TASK_ABORTED`(디바이스 실행 오류)로 죽는다.** 폭 16/64,
   block 워밍업 선행 여부, TP2/TP4 모두 동일. probe E 가 "둘 다 생존"으로 보인
   것은 block 들을 전부 먼저 돌리고 append 로 끝나서 — append 뒤 block 을 한
   번도 안 불렀기 때문이다.

DFlash 루프는 매 라운드 append(수용 토큰 기록) → block(다음 초안) 이므로 규칙
2 를 피할 수 없다. **드래프터 TP>1 은 폐기.** block 단독 TP4 의 4.63 → 1.43 ms
는 영원히 못 쓰는 숫자다.

첫 실행 그래프가 배치를 굳힌다는 가설(블록 워밍업 후 append→block)도 죽는 것을
확인했으므로, 남는 설명은 쓰기 전용 그래프(append)의 캐시 접근이 TP 샤딩 아래서
읽기 그래프(block)와 양립하지 않는 컴파일러/런타임 결함이다.

**반례 — 타깃이 이미 이 구성으로 돌고 있습니다**

`fused_tp4_4k_l35` 에는 prefill 청크 **256** 과 verify **17**, 두 shape 의 정적
그래프가 들어 있고, 둘은 같은 `CompileContext` 에서 나와 같은 KV 캐시를 공유하며
**TP4 로 동시에 살아 있습니다.** 프로덕션 구성이 그것입니다. 따라서 "정적 캐시 +
TP + 런타임 두 개" 라는 조합 자체는 벤더 제약이 아닙니다.

타깃 쌍과의 구조적 차이는 하나입니다: 타깃의 prefill/decode 는 **둘 다**
어텐션 커스텀 op 안에서 캐시를 읽고+쓰는 그래프인 반면, 우리 `append` 는
어텐션 없이 **쓰기만** 하는 그래프입니다. TP 샤딩에서 쓰기 전용 접근의 배치가
읽기+쓰기 그래프와 다르게 결정되는 것으로 보이나, 그 결정을 들여다볼
인터페이스는 없습니다. 벤더 문의용 최소 재현이 probe_append_tp4.py 입니다
(모델 로드 + 그래프 2개 + 호출 6번).

**우회로도 막혀 있습니다.** 자연스러운 해법인 "`append` 와 `block` 을 한 그래프로
합치기" 는 [OPTIMIZATIONS.md](OPTIMIZATIONS.md) ① 의 (c) 에 걸립니다 — 한 그래프에
`PAGED` 호출 두 개를 넣으면 `export` 가 캐시 변형을 기록하지 않아 순서가 보장되지
않고, 인위적 의존성을 만들면 `IndexError: map::at` 으로 죽습니다. TP 수준을 섞는
것도 불가합니다 (TP4 면 캐시가 카드별로 쪼개지는데 TP1 `append` 가 통짜 캐시에
쓰려 하면 어긋납니다).

**실익 정리**

```
block   TP4 로 4.63 -> 1.43 ms 가능하지만 쓸 수 없다      -3.2 ms 상실 (규칙 2)
lm_head TP4 로 4.3  -> 2.0  ms                            -2.3 ms 확보 (`LMH_TP=4`)
append  TP4 는 0.71 -> 0.81 ms 로 오히려 손해             원래부터 TP1 이 맞다
```

`bench_dtp.py` 는 block 을 먼저 컴파일·생성하도록 바꿔 두었다 (규칙 1 대응,
TP1 에서는 무해). `DRAFT_TP>1` 은 규칙 2 로 첫 라운드에서 죽으므로 쓰지 말 것.

`lm_head` 는 캐시도 `CompileContext` 도 없는 단독 그래프라 통합에서도 TP4 가
됩니다. 동일 조건 A/B (gsm8k, NSAMP=5, MAXNEW=256, 타깃 TP4, 각 2 회):

| `LMH_TP` | lmh | tok/s | tau | J/tok |
|---|---|---|---|---|
| 1 | 4.3 ms | 182.90 / 174.34 (평균 **178.62**) | 5.115 | 0.823 |
| 4 | 2.0 ms | 202.98 / 206.58 (평균 **204.78**) | 5.138 | 0.702 |

**처리량 +14.6%, 토큰당 에너지 -14.7%.** 두 그룹의 범위가 겹치지 않습니다.
lm_head 자체 절감(2.3 ms)보다 라운드 감소가 큰데, 카드 0 에 같이 얹혀 있던
드래프터·append 의 경합이 함께 풀린 것으로 보입니다. tau 는 5.115 -> 5.138 로
움직입니다 (로짓 축을 쪼개면서 fp16 리덕션 순서가 바뀝니다). `DRAFT_TP` /
`LMH_TP` 손잡이는 [bench/bench_dtp.py](../bench/bench_dtp.py) 에 있습니다.

> `argmax` 를 lm_head 그래프에 넣는 것은 **N=151936 에서** `DEVICE_GRAPH_CONVERSION`
> 으로 실패합니다. §4 에서 `argmax` 가 된다고 재측정한 것은 N=640 이었습니다 —
> **크기 의존**입니다. 로짓을 호스트로 받아 argmax 하는 현행을 유지해야 합니다.

## 가장 위험한 부류 — 에러 없이 틀린 답

제약 자체보다 이게 더 위험했습니다.

| 증상 | 원인 |
|---|---|
| `cos 0.57` | Append PAGED 호출에서 `q_len != k_len` |
| 캐시가 매번 재업로드 | `compile_from_torch` 인자 3종 미충족 |
| 토큰 id 0 무한 출력 | int4/int8 캘리브레이션 부재 |
| 드래프터에 엉뚱한 레이어 주입 | `extract_context_feature` 의 offset=1 을 빠뜨림 |
| 긴 컨텍스트에서 tau 붕괴 | `sin`/`cos` 가 큰 인자에서 무관한 값 반환 |

전부 예외가 안 납니다. **수치 등가 검증을 매 단계 넣어야 합니다.**
[checks/](../checks/) 의 스크립트들이 그 용도입니다.

그리고 검증 자체가 틀릴 수 있습니다. 마지막 항목은 제가 잘못된 인덱스로 그래프를
자른 건데, 검증 스크립트가 기준과 대상에 **똑같이 틀린 인덱스**를 써서 `cos 1.000018`
로 통과했습니다. **기준은 독립적으로 만들어야 합니다** — 원본 코드가 실제로 읽는
경로를 그대로 따라가야지, 같은 상수를 양쪽에 재사용하면 안 됩니다.
