#!/usr/bin/env bash
# Train the stage-2 listener across multiple seeds, evaluate each, and
# aggregate into mean+-std tables/figures (claude_code_prompt_stage1-2.md §8).
#
# Usage: SEEDS="0 1 2" EXP_DIR=runs/listener_exp1 ./scripts/run_seeds.sh
set -e
cd "$(dirname "$0")/.."

SEEDS=(${SEEDS:-0 1 2})
CONFIG=${CONFIG:-coop_comm/configs/listener.yaml}
EVAL_CONFIG=${EVAL_CONFIG:-coop_comm/configs/eval_listener.yaml}
EXP_DIR=${EXP_DIR:-runs/listener_exp1}

for seed in "${SEEDS[@]}"; do
  echo "=== training seed $seed ==="
  python -m coop_comm.train.train_listener --config "$CONFIG" --seed "$seed" --out_dir "$EXP_DIR/$seed"
done

for seed in "${SEEDS[@]}"; do
  echo "=== evaluating seed $seed ==="
  python -m coop_comm.eval.eval_listener --config "$EVAL_CONFIG" \
    --checkpoint "$EXP_DIR/$seed/listener.pt" --out_dir "$EXP_DIR/$seed/eval"
done

seeds_csv=$(IFS=,; echo "${SEEDS[*]}")
python -m coop_comm.eval.aggregate_seeds --exp_dir "$EXP_DIR" --seeds "$seeds_csv" --out_dir "$EXP_DIR/aggregate"
