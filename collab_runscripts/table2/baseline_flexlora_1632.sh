#!/usr/bin/env zsh

ground=20
num_steps=10
seed_list=(42 45 47)
exp0_importance=0.5
r=0
update_router_every=30

OUT_DIR="..."
DATA_PATH="..."

for seed in ${seed_list[@]}; do
    python3 collab_run.py \
    -gr $ground \
    -num_steps $num_steps \
    -wandb \
    -nc 4 \
    -el '[16, 32, 32, 32]' \
    -en '[1, 1, 1, 1]' \
    -k 1 \
    -out_dir $OUT_DIR \
    -data_path $DATA_PATH \
    -wandb_run_name "16323232_flexlora_agnews_num_grounds_${ground}_num_steps_${num_steps}_seed_${seed}_exp0_${exp0_importance}_r_${r}_update_every_${update_router_every}" \
    -seed $seed \
    -eval_every 20 \
    -log_every 5 \
    -update_router_every $update_router_every \
    -exp0_importance $exp0_importance \
    --aggregation_strategy "flexlora" \
    --is_pruning \
    -gating_update_iters $r \
    -wandb_proj "CoMiGS-Ablation"
done
