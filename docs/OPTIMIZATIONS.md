# 최적화 5가지 — 상세

각 단계는 앞 단계의 하네스를 패치해서 만들었습니다. 패치 스크립트가
[generators/](../generators/) 에 있으므로 단계 간 diff 를 그대로 볼 수 있습니다.

```
bench2.py                (원본, stateless)
  └ mk_sf_paper.py  →  bench_sf_paper.py   ① 드래프터 KV 캐시
      └ mk_ua.py     →  bench_sf_ua.py     ② 비정렬 재개
          └ mk_dual_bench.py → bench_sf_dual.py  ③④ 이중 그래프
              └ mk_slim_bench.py → bench_sf_slim.py  ⑤ 출력 슬라이싱
```

---

## ① 드래프터 KV 캐시 상주

**문제.** 원래 드래프터는 매 라운드 전체 컨텍스트를 다시 올렸습니다. 컨텍스트가
길어질수록 draft 시간이 선형으로 늘었습니다 (16K 에서 158.9 ms).

**해결.** K/V 를 디바이스 상주 텐서로 만들고, Append 그래프가 새 토큰의 K/V 만
기록하고 Block 그래프가 캐시 전체를 읽게 했습니다.

```python
PAGED = torch.ops.rbln_custom_ops.paged_causal_attn_prefill
o = PAGED(q=q0, k=k, v=v, kcache=kc.unsqueeze(2), vcache=vc.unsqueeze(2),
          seq=seq_ctx, scale=s.scale, block_table=block_tables,
          block_size=s.max_ctx, is_bidirectional=True)
```

### 문서에 없는 요구사항 두 가지

**(a) `compile_from_torch` 는 세 개가 다 있어야 캐시가 상주합니다.**

```python
rebel.compile_from_torch(mod, input_info=..., example_inputs=..., compile_context=...)
```

`input_info` / `example_inputs` / `compile_context` 중 **하나라도 빠지면 에러 없이**
매 호출마다 캐시를 다시 업로드합니다. 속도만 안 나오고 결과는 맞아서 알아채기 어렵습니다.

**(b) Append 의 q 길이는 k 길이와 같아야 합니다.**

`q` 를 길이 1로 넣으면 (KV 만 쓰는 게 목적이므로 자연스러운 선택입니다)
**에러 없이 `cos 0.57` 이 나옵니다.** 길이 A 짜리 더미 q 를 넣어야 합니다.

```python
q0 = torch.zeros(1, s.nkv, s.rep, A, s.hd, dtype=th.dtype)   # A == k 길이
```

**(c) 한 그래프에 PAGED 호출 두 개를 넣으면 순서가 보장되지 않습니다.**
`export` 가 캐시 변형(mutation)을 기록하지 않기 때문입니다.
`q = q + (o1.sum()*0.0)` 로 인위적 의존성을 만들면 컴파일러가
`IndexError: map::at` 로 죽습니다. **Append / Block 두 그래프로 쪼개고
호스트에서 순서대로 호출**해야 합니다.

**효과.** draft 15.7 → 5.4 ms. 16K 컨텍스트에서 158.9 → 13.6 ms.
컨텍스트 길이별 버킷 그래프(256/512/1024/2048/3072)가 통째로 불필요해졌습니다.

---

## ② 비정렬 재개 (unaligned prefix caching)

**문제.** 벤더 런타임은 prefix cache 재개 지점이 청크 배수가 아니면 거부합니다.

```python
# optimum/rbln/.../decoderonly_runtime_utils.py:551
if prefix_cached_len % self.rbln_config.prefill_chunk_size != 0:
    raise NotImplementedError(...)
```

그래서 100번째 토큰까지 처리한 상태에서 16개를 검증하려면, 64의 배수인 64로
되돌아가 **이미 계산한 36개를 다시 보내야** 했습니다.

**해결.** 그 가드가 지키는 코드는 **전부 `if self.rbln_config.use_attention_mask:`
블록 안**에 있고, 우리 모델은 `use_attention_mask=False` 입니다. 즉 가드가 막는
경로를 우리는 애초에 타지 않습니다. 런타임에서 가드만 제거했습니다.

```python
_ls = inspect.getsource(RBLNRuntimeModel.prefill_forward).split("\n")
# ... 가드 블록 삭제 ...
_body = textwrap.dedent("\n".join(_o)).replace("super()", "super(_Base, self)")
_ns = dict(RU.__dict__); _ns["_Base"] = RBLNRuntimeModel
exec(compile(_body, "<patched>", "exec"), _ns)
RBLNRuntimeModel.prefill_forward = _ns["prefill_forward"]
```

> `super()` 를 `super(_Base, self)` 로 바꾸는 건 필수입니다. 모듈 레벨에서
> re-exec 하면 `__class__` 셀이 없어서 `super(): __class__ cell not found` 가 납니다.

**검증.** 꼬리를 붙였을 때와 블록만 보냈을 때 hidden state `cos = 0.999996`.
파이프라인 전체로도 tau/라운드 수/토큰 수가 **완전히 동일**했습니다
(gsm8k 5.699 / 1131 / 6446). 순수 속도 최적화라는 증거입니다.

**효과.** verify 63.4 → 47.7 ms.

---

## ③ verify 청크 64 → 17

**문제.** verify 는 항상 16토큰(bonus 1 + 제안 15)만 보내는데 그래프 폭이 64라
48칸이 패딩이었습니다.

**해결.** 청크 17 짜리 prefill 그래프를 따로 컴파일했습니다.
`prefill_chunk_size % 64 == 0` 검사는 **파이썬 설정 객체에만 있는 방어적 검사**입니다
(청크 32 로 먼저 확인: 컴파일 성공 + `cos 0.999922`). 생성자만 우회하면 됩니다.

```python
def _init(self, *a, **kw):
    want = kw.get("prefill_chunk_size")
    if want is not None and want % 64 != 0:
        kw = dict(kw); kw["prefill_chunk_size"] = 64   # 검사만 통과시키고
        _orig(self, *a, **kw); self.prefill_chunk_size = want   # 진짜 값으로 덮어씀
    else:
        _orig(self, *a, **kw)
```

**왜 16 이 아니라 17 인가.** 하네스가 `L % CHUNK == 0` 이면 마지막 토큰을 복제해
패딩합니다. 청크 16 이면 매번 그 경로를 타서 17토큰이 되고 두 번째 청크가 열립니다.
17 로 두면 16토큰이 패딩 1칸으로 한 번에 들어갑니다.

**효과.** verify 47.7 → 41.2 ms. 단독 측정으로는 16토큰 검증 `chunk17 39.5 ms`
vs `chunk256 112.9 ms`.

---

## ④ prefill 청크 64 → 256, 캐시 공유

**문제.** ③에서 verify 를 17로 줄이면 초기 prefill 도 17칸씩 쪼개야 합니다.
prefill 은 호출당 약 37 ms 의 가중치 스트리밍 바닥 비용이 있어서 재앙입니다.

**해결.** 청크 17(verify) 과 청크 256(prefill) **두 그래프를 하나의 KV 캐시를
공유하도록** 컴파일했습니다. `rebel.CompileContext` 와 `mark_static_address` 를
쓰는 벤더 하위 API 로 내려가야 합니다.

```python
for ch in CHUNKS:                      # [17, 256]
    info = cls.get_input_info(batch_size=1, query_length=ch, rbln_config=rc, model_config=mcfg)
    cc = RBLNCompileConfig(compiled_model_name="prefill_%d" % ch, input_info=info)
    meta = [n for n, _, _ in cc.input_info if "past_key_values" in n]
    if ctx is None:                                  # 첫 그래프가 캐시를 만들고
        ex = cc.get_dummy_inputs(fill=0, meta_tensor_names=meta)
        ctx, static = cls._get_compile_context(cc, ex)
    else:                                            # 둘째 그래프는 그걸 물려받음
        ex = cc.get_dummy_inputs(fill=0, static_tensors=static)
    cm = cls._compile_model(wrapped, cc, ex, ctx, rc, phase="prefill")
    cm.save("dual_17_256/prefill_%d.rbln" % ch)
```

> `rc.max_seq_len = 4096` 을 `_update_rbln_config` **전에** 다시 넣어야 합니다.
> 안 그러면 모델 설정의 40960 으로 덮어써집니다.

런타임에서는 두 컴파일 결과를 각각 벤더 래퍼로 감싸서 `prefill_forward` 의
패딩/마스킹/청크 루프/`cache_position` 관리를 그대로 씁니다.

```python
DUAL[ch] = RBLNRuntimeModel(runtime=rebel.Runtime(cm, tensor_type="pt", device=DEV),
                            phase="prefill", batch_size=rc.batch_size,
                            rbln_config=cfg, **common)
```

**검증.** 청크 256 으로 prefill → 청크 17 로 검증한 hidden state 가
전부 청크 256 으로 처리한 기준과 `cos 0.999982`.

---

## ⑤ hidden state 출력 37개 → 6개

**문제.** 벤더 래퍼는 `output_hidden_states=True` 일 때

```python
if self.rbln_config.output_hidden_states:
    return logits, all_hidden_states     # 37개 = embedding + 36 layer
```

로 **37개를 전부 호스트로 내보냅니다.** 런타임도 `num_hidden_layers + 1` 개의
출력 버퍼를 잡습니다.

DFlash 가 실제로 쓰는 건 6개뿐입니다.

| 용도 | 인덱스 |
|---|---|
| 드래프터 입력 (fc 12800→2560) | 2, 10, 18, 26, 34 |
| lm_head 입력 (`hs[-1]`) | 36 |
| (CMR 을 쓸 때만) `hs[-2]` | 35 |

> ⚠️ `target_layer_ids = [1, 9, 17, 25, 33]` 이지만 `extract_context_feature` 가
> **`hidden_states[layer_id + 1]`** 로 읽습니다 (offset=1). 실제 인덱스는
> `[2, 10, 18, 26, 34]` 입니다. 처음에 offset 을 빠뜨리고 `[1,9,17,25,33]` 을
> 잘라 컴파일했는데, 검증 스크립트가 양쪽에 똑같이 틀린 인덱스를 써서
> `cos 1.000018` 로 통과했습니다. **비교 대상이 같은 실수를 공유하면 검증이 아닙니다.**

**해결.** 래퍼를 한 겹 더 감싸서 필요한 것만 반환합니다.

```python
class SliceHS(nn.Module):
    def __init__(self, m, keep):
        super().__init__(); self.m = m; self._keep = keep

    @property
    def phase(self): return self.m.phase
    @phase.setter
    def phase(self, v): self.m.phase = v     # _compile_model 이 여기에 대입한다

    def forward(self, *a):
        logits, hs = self.m(*a)
        return logits, tuple(hs[i] for i in self._keep)
```

런타임 쪽은 출력 버퍼 개수를 맞춰야 합니다. `_prepare_prefill_outputs` 가
`config.num_hidden_layers + 1` 개를 잡으므로, 래퍼에 넘기는 config 만 복사해서
`num_hidden_layers = 5` 로 둡니다 (`rbln_config` 는 건드리지 않습니다).

```python
wcfg = copy.deepcopy(mcfg); wcfg.num_hidden_layers = 5
RBLNRuntimeModel(..., config=wcfg)
```

**효과.** 단독 측정 verify 42.7 → 37.3 ms, prefill(300토큰) 223.8 → 213.0 ms.
파이프라인 전체로는 verify 41.2 → 38.8 ms, 라운드 62.3 → 59.5 ms.
**tau·라운드 수·토큰 수가 이전 단계와 완전히 동일**했으므로 순수 속도 이득입니다.

### 비용은 바이트가 아니라 텐서 개수였다

절감량이 호출당 **verify 5.4 ms / prefill 5.1 ms 로 거의 같았습니다.** 두 호출은
전송량이 15배 차이납니다 (3.2 MB vs 97 MB). 즉 대역폭이 아니라

```
31개 텐서 × 약 0.17 ms/개 = 약 5 ms   (호출당 고정)
```

**출력 텐서 하나당 붙는 고정 비용**이 지배적입니다. "37 → 6 이니 트래픽 8배 감소"
라는 계산은 맞지 않고, 실제로 줄어든 건 **호스트 복사 호출 31번**입니다.

---

## 부록 — 시도했지만 아닌 것

**패딩 낭비는 위치 수로 세면 안 됩니다.** verify 가 64칸 중 16칸만 쓰니
"75% 낭비" 처럼 보이지만, 비용의 약 37/42.9 ms 는 가중치 스트리밍이라
위치 수와 무관합니다. 실제 회수 가능분은 약 29% 였습니다.
draft 패딩도 "위치 25%" 였지만 시간으로는 0.5% 였습니다.

**`.float()` 제거는 효과가 없었습니다.** 처음엔 라운드당 36 ms(28%) 절감이라고
주장했는데, ⓐ 다른 카드가 바쁠 때 마이크로벤치를 돌렸고 ⓑ 캐시가 식은 `.float()` 를
캐시가 더워진 fp16 argmax 와 비교한 결함이 있었습니다. 단독 A/B 재측정 결과
65.95 vs 62.98 tok/s 로 실행 간 노이즈(약 10%) 안이었습니다.
