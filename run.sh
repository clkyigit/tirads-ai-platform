#!/usr/bin/env bash
# Linux / macOS başlatıcı.
#   ./run.sh            -> normal (config'teki kaynak ve model)
#   ./run.sh --demo     -> sentetik görüntü + demo segmentasyon
#   ./run.sh --ml       -> PyTorch bağımlılıklarını da kur, sonra başlat
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
if [ ! -d .venv ]; then
  echo "Sanal ortam oluşturuluyor (.venv)..."
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

stamp=.venv/.req-stamp
want=$(cksum requirements.txt | cut -d" " -f1)
if [ ! -f "$stamp" ] || [ "$(cat "$stamp")" != "$want" ]; then
  python -m pip install --upgrade pip >/dev/null
  python -m pip install -r requirements.txt
  echo "$want" > "$stamp"
fi

args=()
for a in "$@"; do
  if [ "$a" = "--ml" ]; then
    python -m pip install -r requirements-ml.txt
  else
    args+=("$a")
  fi
done

if [ "$(uname)" = "Linux" ] && ls /dev/video* >/dev/null 2>&1; then
  if ! [ -r "$(ls /dev/video* | head -n1)" ]; then
    echo "UYARI: /dev/video* okunamıyor. 'sudo usermod -aG video \$USER' ve oturumu yeniden açın."
  fi
fi

exec python -m backend ${args[@]+"${args[@]}"}
