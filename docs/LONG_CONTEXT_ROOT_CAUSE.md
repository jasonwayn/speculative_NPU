> ## ⚠️ 이 문서는 대체되었습니다 — [ROPE_ROOT_CAUSE.md](ROPE_ROOT_CAUSE.md)
>
> 아래의 결정적 근거(K=1040 정상 / K=1041 붕괴)는 **재현되지 않았습니다.**
> 재현 시 K=1040 도 동일하게 cos 0.8528 로 틀리며, 문서의 `0.999972` 는
> NPU-vs-CPU 가 아니라 NPU-vs-NPU 비교값입니다. shape 의존 결함은 없습니다.
>
> 마스크 값 변경으로 마스킹 문제를 배제한 논리도 무효입니다 — 시험한 네 값이
> 모두 `exp()` 후 0 이라 마스크 적용 여부를 구분하지 못합니다.
>
> 실제 원인은 **RBLN `sin`/`cos` 의 인자 크기 의존 오차**이고,
> 수정 후 tau 1.933 → 6.156~6.459 (1/2/4장 전부), 처리량 2.5~2.8배입니다.
>
> 아래 내용 중 살아남은 것: 레이어 오포팅이 아니라는 결론, 드래프트 컨텍스트를
> 제한하라는 권고(수정 전 한정), 타깃 SiLU 근사 오차는 원인이 아니라는 관찰.

# DFlash long-context accuracy root cause on ATOM+

Date: 2026-08-19

## Conclusion

The long-context tau collapse is not caused by a wrong target layer index or by
one incorrectly ported DFlash layer. The primary cause is a shape-dependent
attention accuracy failure in the RBLN 0.10.2 backend on ATOM+.

The strongest controlled result uses the same 1024-token target hidden states,
the same 16-token draft block, float32 inputs, and standard (non-paged) DFlash
attention:

| Compiled target context | Total K length | NPU vs CPU min cosine | Proposal matches |
|---:|---:|---:|---:|
| 1024 | 1040 | 0.999972 | 16/16 |
| 1025 (one masked zero) | 1041 | 0.853079 | 5/16 |

On CPU, adding and masking that one zero context position changes the output by
less than `1e-6` relative error and keeps 16/16 proposal tokens. Changing the
mask value among `-100`, `-10000`, `-1e9`, and float32 minimum does not change
the bad NPU result. This isolates the failure to the compiled attention shape,
not DFlash semantics or padding-mask construction.

The production stateful draft uses `paged_causal_attn_prefill`. It shows the
same failure around 1K context and accumulates error through all five draft
layers. Replacing it with the SDK's explicit non-causal `paged_attn_prefill`
plus a valid-token mask produces the same 5/16 proposal result.

## Evidence

### Target path

- Target logits and target continuation tokens match CPU at 1024 context.
- Target hidden-state divergence was traced to the NPU SiLU approximation, not
  to target layer selection.
- Replacing SiLU with `x / (1 + exp(-clamp(x, -20, 20)))` reduces target layer-1
  final-hidden max relative error from 7.52% to 0.82%.
- Accurate target SiLU does not recover long-context tau or draft proposals.

Therefore the target SiLU issue is real, but it is not the cause of the tau
collapse.

### Draft path

At 1024 context, the paged draft output differs from the official PyTorch
DFlash forward:

- Final draft min cosine: about 0.853
- Proposal matches: 5/16
- Layer errors accumulate across all five draft layers.
- The NPU result is much closer to bidirectional attention than causal
  attention, confirming that `is_bidirectional=True` is active.

Selected first-round length results are input-sensitive but show the failure
region clearly:

| Context | Min cosine | Proposal matches |
|---:|---:|---:|
| 256 | 0.99945 | 16/16 |
| 512 | 0.99667 | 16/16 |
| 768 | 0.99648 | 14/16 |
| 960 | 0.99412 | 13/16 |
| 1008 | 0.98427 | 14/16 |
| 1023 | 0.83858 | 11/16 |
| 1024 | 0.85359 | 5/16 |
| 1025 | 0.93221 | 9/16 |
| 1040 | 0.87355 | 10/16 |

The non-monotonic values are expected when small hidden-state changes cross
near-tied LM-head logits. The controlled 1040-vs-1041 standard-attention shape
test is the decisive result.

## Ruled out

- Wrong target hidden-state layer IDs (`[1, 9, 17, 25, 33]` in DFlash,
  corresponding to hidden-state tuple indices `[2, 10, 18, 26, 34]`)
- A layer-16/position-608 indexing error
- FP16 versus BF16 alone
- Target prefill width 17 versus 256
- Append width 16 versus 256
- Draft cache capacity 4096 versus 16384
- Causal versus bidirectional semantics
- `paged_causal_attn_prefill` versus explicit `paged_attn_prefill`
- Accurate SiLU in either target or draft as a tau fix
- One 4096-token page versus sixteen 256-token pages; the latter was worse
- Padding-mask construction on CPU

## Interpretation

This is an SDK/compiler attention-kernel limitation or defect in the installed
RBLN 0.10.2 stack, not evidence that ATOM+ hardware is inherently unable to run
DFlash. The exact internal cause (tiling, reduction precision, or kernel
selection) cannot be determined without vendor compiler/kernel visibility.

It is also not a single-layer porting mistake. Every draft layer calls the same
attention path, so the per-call error compounds through five layers and changes
the LM-head argmax tokens.

## Recommended next actions

1. Send Rebellions a minimal issue containing the 1040-vs-1041 shape result,
   ATOM+ device, RBLN SDK/compiler 0.10.2, GQA 32/8 heads, head dimension 128,
   query length 16, float32, and the 16/16 versus 5/16 comparison.
2. Test a newer SDK in a separate environment. Do not replace the working 0.10.2
   environment until the compiled target and draft graphs are reproduced.
3. For experiments before a vendor fix, cap or retrieve the draft context below
   the unstable region (start with 512, then validate 768/960 on the full
   workload). The target can still keep its full KV cache.
4. Report long-context NPU tau only together with CPU/GPU proposal agreement.
   A fast round with incorrect proposals is not a valid performance result.

## Scope and safety

All work used separate diagnostic scripts and in-memory compiled graphs. The
production graph directories `fused_17_256`, `fused_tp2`, and `fused_tp4` were
not modified. No changes were committed or pushed.
