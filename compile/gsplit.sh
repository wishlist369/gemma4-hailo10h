#!/bin/bash
# gsplit.sh (DFC 호스트): g_list.txt(<name> <src q.har 접두> <cache_off>) 순차 분리 — NET_NAME=g4_<name>, CACHE_OFF → chunk/<name>_tbt.har, meta/ref 복사, out/kvx_<name>.log 'SPLIT OK'
cd /root/hailo/g4 || exit 1; IMG=hailo_ai_sw_suite_2026-08:1
while read n src off; do
  [ -z "$n" ] && continue; L=out/kvx_$n.log
  cp -f chunk/${src}_meta.json chunk/${n}_meta.json; cp -f chunk/${src}_ref.npz chunk/${n}_ref.npz
  docker run --rm --cpus 2 -e CACHE_OFF=$off -e NET_NAME=g4_$n -v /root/hailo/g4:/work2 -w /work2 --entrypoint bash $IMG -lc "source /local/workspace/hailo_virtualenv/bin/activate; python har_split_cid.py /work2/chunk/${src}_r2.q.har /work2/chunk/${n}_tbt.har gemma4__tbt" > out/split_$n.log 2>&1
  ids=$(grep -a -o 'ids \[.*' out/split_$n.log | head -1)
  if [ -s chunk/${n}_tbt.har ] && grep -a -q "net name -> g4_$n" out/split_$n.log; then echo "$(date -u '+%m-%d %H:%M') SPLIT OK $n <- $src off $off $ids" | tee -a $L >> out/gsplit.log
  else echo "$(date -u '+%m-%d %H:%M') SPLIT FAILED $n: $(tail -n 2 out/split_$n.log | tr '\n' ' ' | cut -c1-200)" | tee -a $L >> out/gsplit.log; fi
done < g_list.txt
echo "$(date -u '+%m-%d %H:%M') GSPLIT DONE disk $(df -h / | awk 'NR==2{print $5}')" >> out/gsplit.log
