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

## 4. argmax 컴파일 불가 / topk 실행 불가

`argmax` 는 컴파일 단계에서 거부되고, `topk` 는 컴파일은 되는데 실행이 안 됩니다.
샘플링은 호스트에서 해야 합니다.

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
TP1 단일 그래프(청크 17 만)로는 컴파일·로드·실행이 다 되지만 프롬프트 프리필을
17 토큰씩 964 회 돌아야 해서 프로덕션 구성이 아닙니다.
자세한 조사 기록은 [HOST_THREAD_STARVATION.md](HOST_THREAD_STARVATION.md) §7.

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
