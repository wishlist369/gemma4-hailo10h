#!/usr/bin/env bash
# install.sh — set up a ready-to-run Gemma 4 E2B on Hailo-10H directory (no DFC needed).
#   1) downloads the 12 precompiled HEFs + metadata from the GitHub Release and verifies SHA256
#   2) downloads google/gemma-4-E2B-it from Hugging Face and exports the host tables (numpy only, ~5.5 GB)
#   3) copies the runtime; then:  cd <dir> && python g4chat.py
# Prerequisite: Hailo-10H with HailoRT 5.4.0 driver/firmware and the matching `hailort` Python wheel
#   (from the Hailo Developer Zone) installed in the Python you run this with.
# Usage: ./install.sh [target_dir=./g4hailo]     env: PY=python3  RELEASE=v0.1  HF_MODEL=google/gemma-4-E2B-it
set -euo pipefail
DIR=${1:-./g4hailo}; PY=${PY:-python3}; REL=${RELEASE:-v0.1}; HF_MODEL=${HF_MODEL:-google/gemma-4-E2B-it}
REPO=wishlist369/gemma4-hailo10h; URL=https://github.com/$REPO/releases/download/$REL
SRC=$(cd "$(dirname "$0")" && pwd)
CHUNKS="g00 g03 g06 g09 g12 g15 g18 g21 g24 g27 g30 g33"

say(){ printf '\n== %s\n' "$*"; }
say "checking Python and HailoRT"
"$PY" -c "import hailo_platform, sys; print('hailort OK, python', sys.version.split()[0])" || {
  echo "!! hailo_platform not importable from $PY. Install the HailoRT 5.4.0 Python wheel (Hailo Developer Zone) first."; exit 1; }
"$PY" -m pip install -q "numpy>=1.24" "tokenizers>=0.20" "huggingface_hub>=0.25"
mkdir -p "$DIR"; DIR=$(cd "$DIR" && pwd)

say "downloading HEFs ($REL, ~1.1 GB)"
cd "$DIR"
curl -fL --retry 3 -o SHA256SUMS "$URL/SHA256SUMS"
for f in $(for c in $CHUNKS; do echo ${c}_tbt.hef; done) g4_meta_json.tar; do
  if [ -s "$f" ] && grep " $f\$" SHA256SUMS | sha256sum -c --status - 2>/dev/null; then echo "  $f (cached)"; continue; fi
  curl -fL --retry 3 -o "$f" "$URL/$f"
done
sha256sum -c SHA256SUMS
tar xf g4_meta_json.tar

say "downloading $HF_MODEL and exporting host tables"
"$PY" - "$HF_MODEL" "$DIR/model" <<'EOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2], allow_patterns=["*.safetensors", "*.json", "*.jinja"])
EOF
"$PY" "$SRC/convert/g4export_np.py" "$DIR/model" "$DIR/host"

say "installing runtime"
cp "$SRC/runtime/g4rt_host.py" "$SRC/runtime/g4tbt.py" "$SRC/runtime/g4chat.py" "$DIR/"
cat <<EOF

Done. Run:
  cd $DIR && $PY g4chat.py                 # interactive (startup ~26 s, then ~2 tok/s)
  cd $DIR && $PY g4chat.py --prompt "..."  # one shot
You can delete $DIR/model afterwards (the runtime only needs host/ and the HEFs).
EOF
