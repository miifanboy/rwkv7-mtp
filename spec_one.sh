#!/usr/bin/env bash
# Run one llama-server speculative-decoding config and print the draft stats.
# usage: spec_one.sh <port> <target.gguf> <draft.gguf> <n_max>
set -u
PORT=$1; TGT=$2; DFT=$3; NMAX=$4
BIN=C:/Software/rwkv-related/rwkv7-autoparser-fix/llama.cpp/build/bin/Release/llama-server.exe
LOG="spec_one_${PORT}.log"
RESP="spec_resp_${PORT}.json"

CUDA_VISIBLE_DEVICES="" "$BIN" \
    -m "$TGT" -md "$DFT" --spec-type draft-mtp \
    --spec-draft-n-max "$NMAX" \
    -c 256 -b 32 -ngl 0 -np 1 --no-webui --port "$PORT" \
    > "$LOG" 2>&1 &
SRV=$!

for i in $(seq 1 90); do
    sleep 1
    if curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then
        break
    fi
    if ! kill -0 $SRV 2>/dev/null; then
        echo "server died (see $LOG)"
        tail -5 "$LOG" | sed 's/\x1b\[[0-9;]*m//g'
        exit 1
    fi
done

curl -s "http://127.0.0.1:$PORT/completion" -H "Content-Type: application/json" \
    -d '{"prompt":"hello world this is a test","n_predict":16,"temperature":0.0}' \
    -o "$RESP"

kill $SRV 2>/dev/null
wait $SRV 2>/dev/null

python - "$RESP" <<'PY'
import json, sys
path = sys.argv[1]
try:
    d = json.load(open(path, encoding="utf-8", errors="replace"))
except Exception as e:
    print(f"  !! could not read {path}: {e}")
    sys.exit(0)
t = d.get("timings", {})
print(f"  timings: draft_n={t.get('draft_n')} accepted={t.get('draft_n_accepted')} "
      f"predicted_n={t.get('predicted_n')} tps={t.get('predicted_per_second')}")
PY

sed 's/\x1b\[[0-9;]*m//g' "$LOG" | grep -aE "statistics        draft-mtp|draft acceptance|nextn_layer_offset out of range|GGML_ASSERT|terminate called" | sed 's/^/  /'
