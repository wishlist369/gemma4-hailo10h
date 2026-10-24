#!/bin/bash
# chunk_quant_env6.sh (CT142; + LSMODE=norm|conv|none; + NOKV=1 → g4conv --no_kvout; + TAP=레이어명 → g4conv --tap(탭 그래프); + CALIB_FULL=1·PROMPT_FILE=경로 → g4chunk 보정셋 풀윈도우·긴 참조 프롬프트; chunk_quant.sh + QENV 환경변수 전달, 예 QENV="GM8=1"): 청크 1개 양자화(R2, ctx512 기본, PLE split=블록별 입력). 사용: ./chunk_quant.sh <name> <lo-hi> [ctx=512] [calib=12] [EVAL=fp] [head=chunk|hidden]
#   g4conv(head chunk) → g4chunk(HF 정답+보정셋; hidden_in·공유 K/V 포함) → (lo=0이면 g4conv 재실행으로 압축 어휘) → p1_quant(KV=off EMU=1 EVAL=$EVAL) → HAR
cd /root/hailo/g4 || exit 1; mkdir -p chunk; chmod 777 chunk
n=$1; L=$2; CTX=${3:-512}; NC=${4:-12}; EV=${5:-fp}; HEAD=${6:-chunk}; PY=/root/hailo/exp/bin/python
lo=${L%-*}
echo "== $n blocks $L ctx $CTX calib $NC $(date +%H:%M)"
$PY g4conv.py --layers $L --ctx $CTX --head $HEAD --fold scale --lnact --ple split ${CONVARGS} ${TAP:+--tap $TAP} ${NOKV:+--no_kvout} ${LSMODE:+--lsmode $LSMODE} --out chunk/$n > out/cq_${n}_conv.log 2>&1 || { echo "CONV FAIL"; tail -3 out/cq_${n}_conv.log; exit 1; }
$PY g4chunk.py --meta chunk/${n}_meta.json --out chunk/${n}_ref.npz --calib $NC --out_calib chunk/${n}_calib.npz ${CALIB_FULL:+--calib_full} ${PROMPT_FILE:+--prompt "$(cat $PROMPT_FILE)"} > out/cq_${n}_ref.log 2>&1 || { echo "REF FAIL"; tail -3 out/cq_${n}_ref.log; exit 1; }
tail -1 out/cq_${n}_ref.log | cut -c1-160
if [ "$lo" = "0" ]; then   # 임베딩 입력 청크: 보정/정답 토큰으로 압축 어휘 재생성
  $PY g4conv.py --layers $L --ctx $CTX --head $HEAD --fold scale --lnact --ple split ${CONVARGS} --vocab /root/hailo/g4/chunk/${n}_calib.npz,/root/hailo/g4/chunk/${n}_ref.npz --out chunk/$n > out/cq_${n}_conv2.log 2>&1 || { echo "CONV2 FAIL"; tail -3 out/cq_${n}_conv2.log; exit 1; }
fi
tail -1 out/cq_${n}_conv*.log | tail -1 | cut -c1-160
chmod a+r chunk/*
ML0=$(wc -l < out/memlog.txt 2>/dev/null || echo 1)
echo "quant start $(date +%H:%M)"
docker run --rm --cpus 8 --name q_$n -v /root/hailo/g4:/work2 -w /work2 --entrypoint bash hailo_ai_sw_suite_2026-08:1 \
  -lc "source /local/workspace/hailo_virtualenv/bin/activate; KV=off EMU=1 EVAL=$EV $QENV python p1_quant.py /work2/chunk/$n /work2/chunk/${n}_calib.npz /work2/chunk/${n}_ref.npz r2 enabled a8_w4_a16 ln_in_w,ln_ffn_w,final_norm_w - disabled; echo EXIT=\$?" > out/cq_${n}_quant.log 2>&1
sed -e 's/\x1b\[[0-9;]*[A-Za-z]//g' out/cq_${n}_quant.log | tr '\r' '\n' | grep -a -E 'NATIVE|optimize OK|FP_OPTIMIZED|QUANTIZED|BIT_EXACT|rror:|EXIT' | tail -6 | cut -c1-200
echo "peak: $(tail -n +$ML0 out/memlog.txt | grep -o "q_$n:[0-9.]*GiB" | sort -t: -k2 -n | tail -1)  $(date +%H:%M)"
ls -la chunk/${n}_r2.q.har 2>/dev/null | awk '{print $5, $9}'
echo ALLDONE
