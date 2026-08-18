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

## 가장 위험한 부류 — 에러 없이 틀린 답

제약 자체보다 이게 더 위험했습니다.

| 증상 | 원인 |
|---|---|
| `cos 0.57` | Append PAGED 호출에서 `q_len != k_len` |
| 캐시가 매번 재업로드 | `compile_from_torch` 인자 3종 미충족 |
| 토큰 id 0 무한 출력 | int4/int8 캘리브레이션 부재 |

전부 예외가 안 납니다. **수치 등가 검증을 매 단계 넣어야 합니다.**
[checks/](../checks/) 의 스크립트들이 그 용도입니다.
