#!/bin/bash
# kvq.sh (DFC 호스트): 목록(name lo-hi)별 KV=on 양자화만(tbt 분리·컴파일은 별도 배치). 순차. 디스크 10G 미만이면 대기. 사용: nohup ./kvq.sh kv_list.txt > out/kvq.log 2>&1 &
cd /root/hailo/g4 || exit 1; LIST=${1:-kv_list.txt}
log(){ echo "$(date -u '+%m-%d %H:%M') $*"; }
while read -r n R; do
  [ -z "$n" ] && continue
  if [ -s chunk/${n}_r2.q.har ]; then log "$n q.har exists, skip"; continue; fi
  while [ "$(df --output=avail -BG / | tail -1 | tr -dc 0-9)" -lt 10 ]; do log "disk low, waiting 10m"; sleep 600; done
  log "$n quant start ($R)"
  QENV="KV=on" CALIB_FULL=1 nice -n 10 ./chunk_quant.sh $n $R 512 12 fp chunk > out/cq_${n}_pipeline.log 2>&1
  if grep -q ALLDONE out/cq_${n}_pipeline.log && [ -s chunk/${n}_r2.q.har ]; then
    log "$n quant done $(grep -o -E 'peak: [^ ]+' out/cq_${n}_pipeline.log | tail -1) $(grep -o -E 'FP_OPTIMIZED: cos [0-9.]+' out/cq_${n}_pipeline.log | tail -1) size $(stat -c %s chunk/${n}_r2.q.har)"
  else
    log "$n QUANT FAILED: $(tail -n 2 out/cq_${n}_pipeline.log | tr '\n' ' ' | cut -c1-160)"
  fi
  rm -f chunk/${n}_params.npz
  log "$n disk $(df -h / | awk 'NR==2{print $5}')"
done < $LIST
log "KVQ DONE"
