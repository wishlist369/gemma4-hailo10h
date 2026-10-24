# 범용 컴파일: python p1_compile.py <q.har> <out.hef> [alls_file]   (alls 없으면 자동 모드 level max)
import sys, time
from hailo_sdk_client import ClientRunner
har, out = sys.argv[1], sys.argv[2]
r = ClientRunner(har=har, hw_arch="hailo10h")
script = open(sys.argv[3]).read() if len(sys.argv) > 3 else "performance_param(compiler_optimization_level=max)\nallocator_param(timeout=14400)\n"
r.load_model_script(script)
t0 = time.time()
hef = r.compile()
open(out, "wb").write(hef)
print(f"compile OK {time.time()-t0:.0f}s hef {len(hef)/1e6:.1f}MB -> {out}", flush=True)
