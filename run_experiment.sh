#!/usr/bin/env bash
set -euo pipefail
PY=${PY:-python}
N=${N:-300}
mkdir -p results outputs
"$PY" -m qlora_demo.train --output-dir outputs "$@" | tee outputs/train.log
cp outputs/train_summary.json results/train_summary.json
for model in base-chat merged; do
  if [[ ! -f "results/eval_$model.jsonl" ]]; then
    "$PY" -m evalh.cli run --model "hf:outputs/$model" --n "$N" --out "results/eval_$model.jsonl"
  fi
done
"$PY" -m evalh.cli report results/eval_base-chat.jsonl results/eval_merged.jsonl \
  --markdown results/before_after.md --json results/before_after.json
