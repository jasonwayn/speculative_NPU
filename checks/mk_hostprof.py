"""bench_dtp.py 의 디코드 루프에 호스트 구간 타이머를 넣는다 -> bench_hp.py

라운드 25.7 ms 중 21.0 ms 만 계측되고 4.8 ms (18.7%) 가 어디에도 안 잡힌다.
py-spy 는 컨테이너에서 ptrace 가 막혀 못 쓰므로 직접 잰다.

넣는 구간 (전부 호스트):
  hemb     블록 임베딩
  hang     RoPE 각도 + 드래프터 인자 구성. 지금은 T["draft"] 안에 섞여 있다
  hcast    드래프터 출력 as_tensor + lm_head 입력 .to(DT)
  hargmax  lm_head 로짓의 argmax. 어휘 151936 을 fp32 로 올린다 (라운드당 9.7 MB)
  hacc     타깃 로짓 argmax + 수용 길이 판정
  hnf      hidden state 5 개 concat + THB 기록
  hbook    위치/버퍼 갱신
  hstop    정지 토큰 검사. VRB[0, P:VL] 를 매 라운드 전부 순회한다
  hround   라운드 전체 (잔차 = hround - 나머지 전부)
"""
import ast
import io
import sys

src, dst = sys.argv[1], sys.argv[2]
s = io.open(src, encoding="utf-8").read()
n = 0


def sub(old, new):
    global s, n
    assert s.count(old) == 1, (s.count(old), old[:120])
    s = s.replace(old, new, 1)
    n += 1


sub("""    T = dict(draft=0.0, verify=0.0, lmh=0.0, score=0.0, prefill=0.0, append=0.0)""",
    """    T = dict(draft=0.0, verify=0.0, lmh=0.0, score=0.0, prefill=0.0, append=0.0,
             hemb=0.0, hang=0.0, hcast=0.0, hargmax=0.0, hacc=0.0, hnf=0.0,
             hbook=0.0, hstop=0.0, hround=0.0)""")

# lm_head 호스트 argmax
sub("""        T["lmh"] += time.time() - s
        return torch.argmax(o[:, :n].float(), dim=-1)""",
    """        T["lmh"] += time.time() - s
        s = time.time()
        _am = torch.argmax(o[:, :n].float(), dim=-1)
        T["hargmax"] += time.time() - s
        return _am""")

# 라운드 시작 + 임베딩
sub("""            blk = torch.full((1, B), MASK, dtype=torch.long); blk[0, 0] = bonus
            with torch.no_grad(): noi = embed(blk).detach()""",
    """            _rs = time.time()
            blk = torch.full((1, B), MASK, dtype=torch.long); blk[0, 0] = bonus
            _h0 = time.time()
            with torch.no_grad(): noi = embed(blk).detach()
            T["hemb"] += time.time() - _h0""")

# RoPE 각도 + 인자 구성 (현재 T["draft"] 에 포함돼 있다)
sub("""            else:
                _ang = host_angle(""",
    """            else:
                _h0 = time.time()
                _ang = host_angle(""")
sub("""                    _block_args.append(_block_mask)
                _block_result = rt_b(*_block_args)""",
    """                    _block_args.append(_block_mask)
                T["hang"] += time.time() - _h0
                _block_result = rt_b(*_block_args)""")

# 드래프터 출력 캐스트
sub("""            _trace = _block_result if isinstance(_block_result, (tuple, list)) else None
            h = torch.as_tensor(_block_result[-1] if _trace is not None else _block_result)""",
    """            _h0 = time.time()
            _trace = _block_result if isinstance(_block_result, (tuple, list)) else None
            h = torch.as_tensor(_block_result[-1] if _trace is not None else _block_result)
            T["hcast"] += time.time() - _h0""")

# lm_head 입력 캐스트
sub("""            blk[:, 1:] = lmh(h[:, 1 - B:, :].to(DT))""",
    """            _h0 = time.time(); _lx = h[:, 1 - B:, :].to(DT)
            T["hcast"] += time.time() - _h0
            blk[:, 1:] = lmh(_lx)""")

# 수용 판정
sub("""            post = torch.argmax(post_logits[:, vl - c0:vl - c0 + B].float(), dim=-1)
            a = int((blk[:, 1:] == post[:, :-1]).cumprod(dim=1).sum(dim=1)[0])
            acc.append(a + 1)""",
    """            _h0 = time.time()
            post = torch.argmax(post_logits[:, vl - c0:vl - c0 + B].float(), dim=-1)
            a = int((blk[:, 1:] == post[:, :-1]).cumprod(dim=1).sum(dim=1)[0])
            acc.append(a + 1)
            T["hacc"] += time.time() - _h0""")

# hidden state 병합
sub("""            nf = torch.cat([hs2[i] for i in range(5)], dim=-1)[:, vl - c0:vl - c0 + a + 1, :].to(DT)
            THB[:, TL:TL + nf.shape[1]] = nf
            newpos = torch.arange(ctx_len, ctx_len + a + 1)""",
    """            _h0 = time.time()
            nf = torch.cat([hs2[i] for i in range(5)], dim=-1)[:, vl - c0:vl - c0 + a + 1, :].to(DT)
            THB[:, TL:TL + nf.shape[1]] = nf
            newpos = torch.arange(ctx_len, ctx_len + a + 1)
            T["hnf"] += time.time() - _h0""")

# 장부 갱신
sub("""            TL += nf.shape[1]; ctx_len += a + 1
            VRB[:, VL:VL + a + 1] = blk[:, :a + 1]; VL += a + 1
            bonus = int(post[0, a]); cached = (VL // CHUNK) * CHUNK""",
    """            _h0 = time.time()
            TL += nf.shape[1]; ctx_len += a + 1
            VRB[:, VL:VL + a + 1] = blk[:, :a + 1]; VL += a + 1
            bonus = int(post[0, a]); cached = (VL // CHUNK) * CHUNK
            T["hbook"] += time.time() - _h0""")

# 정지 검사 + 라운드 총계
sub("""            if any(int(t) in STOP for t in VRB[0, P:VL]): break
            if ctx_len >= MAXC - 2 * B: break""",
    """            _h0 = time.time()
            _hit = any(int(t) in STOP for t in VRB[0, P:VL])
            T["hstop"] += time.time() - _h0
            T["hround"] += time.time() - _rs
            if _hit: break
            if ctx_len >= MAXC - 2 * B: break""")

# 리포트
sub("""         lmh_ms=round(1000 * T["lmh"] / n, 1), prefill_s=round(T["prefill"], 2),""",
    """         lmh_ms=round(1000 * T["lmh"] / n, 1), prefill_s=round(T["prefill"], 2),
         **{k + "_ms": round(1000 * T[k] / n, 2) for k in
            ("hemb", "hang", "hcast", "hargmax", "hacc", "hnf", "hbook", "hstop", "hround")},""")

io.open(dst, "w", encoding="utf-8", newline="\n").write(s)
ast.parse(s)
print("patched %d hunks -> %s" % (n, dst))
