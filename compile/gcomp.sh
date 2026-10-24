#!/bin/bash
# gcomp.sh (DFC 호스트) <name...>: 각 이름의 SPLIT OK 대기 → u75 컴파일(--cpus 4) → chunk/<name>_tbt.hef, out/gcomp.log 'COMPILE OK|FAILED <name>'
cd /root/hailo/g4 || exit 1; IMG=hailo_ai_sw_suite_2026-08:1
for n in "$@"; do
  until grep -q -E "SPLIT (OK|FAILED) $n" out/kvx_$n.log 2>/dev/null; do sleep 60; done
  grep -q "SPLIT OK $n" out/kvx_$n.log || { echo "$(date -u '+%m-%d %H:%M') SKIP $n (split failed)" >> out/gcomp.log; continue; }
  t0=$(date +%s); echo "$(date -u '+%m-%d %H:%M') compile start $n" >> out/gcomp.log
  docker run --rm --cpus 4 --name c_${n}_tbt -v /root/hailo/g4:/work2 -w /work2 --entrypoint bash $IMG \
    -lc "source /local/workspace/hailo_virtualenv/bin/activate; nice -n 5 python p1_compile.py /work2/chunk/${n}_tbt.har /work2/chunk/${n}_tbt.hef /work2/micro/compile_l0_u75.alls; echo EXIT=\$?" > out/cc2_$n.log 2>&1
  r=$(sed -e 's/\x1b\[[0-9;]*[A-Za-z]//g' out/cc2_$n.log | tr '\r' '\n' | grep -a -E 'partition to|Successful Compilation|rror' | grep -v -i torchvision | tail -2 | cut -c1-110 | tr '\n' ' ')
  if grep -q 'EXIT=0' out/cc2_$n.log && [ -s chunk/${n}_tbt.hef ]; then echo "$(date -u '+%m-%d %H:%M') COMPILE OK $n $(( ($(date +%s)-t0)/60 ))m $r" >> out/gcomp.log
  else echo "$(date -u '+%m-%d %H:%M') COMPILE FAILED $n $r" >> out/gcomp.log; fi
done
