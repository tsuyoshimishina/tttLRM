#!/bin/bash
#
# Usage: bash script/inference_colmap.sh <data_path_json> [exp_name]
#   data_path_json : path to the JSON listing opencv_cameras.json paths
#   exp_name       : experiment name (default: colmap_eval)
#
# Example:
#   bash script/inference_colmap.sh ./data_example/colmap_processed/colmap_data_path.json my_scene

set -euo pipefail
export NCCL_DEBUG=WARN
PORT=29501

DATAPATH=${1:?"Usage: $0 <data_path_json> [exp_name]"}
EXP_NAME=${2:-colmap_eval}

NUM_GPUS=${NUM_GPUS:-1}
CKPT_PATH=${CKPT_PATH:-./checkpoints/dl3dv_full.pt}
NUM_VIEWS=32
FOLD_SIZE=8
CONFIG=./configs/dl3dv_full.yaml

# Compute num_views from DATAPATH:
#   kmeans view selection returns (NUM_VIEWS + num_target_frames) views,
#   where num_target_frames = count of frames at indices divisible by FOLD_SIZE.
#   training.num_views must not exceed this, otherwise __getitem__ retries infinitely.
COMPUTED=$(python3 -c "
import json, sys
fold = ${FOLD_SIZE}
nv = ${NUM_VIEWS}
paths = json.load(open('${DATAPATH}'))
min_total = float('inf')
for p in paths:
    n = len(json.load(open(p))['frames'])
    n_target = len([i for i in range(n) if i % fold == 0])
    n_rest = n - n_target
    if nv > n_rest:
        print(f'WARNING: NUM_VIEWS={nv} exceeds available frames ({n_rest}) in {p}, clamping', file=sys.stderr)
        nv = min(nv, n_rest)
    min_total = min(min_total, nv + n_target)
print(f'{nv} {min_total}')
")
NUM_VIEWS=$(echo $COMPUTED | cut -d' ' -f1)
NUM_TOTAL_VIEWS=$(echo $COMPUTED | cut -d' ' -f2)

echo "=== COLMAP inference: DATAPATH=${DATAPATH}, EXP_NAME=${EXP_NAME}, NUM_VIEWS=${NUM_VIEWS}, num_total_views=${NUM_TOTAL_VIEWS} ==="

torchrun --nproc_per_node ${NUM_GPUS} --nnodes 1 \
        --rdzv_id 18638 --rdzv_backend c10d \
        --rdzv_endpoint localhost:${PORT}  \
        inference.py ${CONFIG} \
        -s evaluation true -s evaluation_out_dir ./evaluation/${EXP_NAME}/views_${NUM_VIEWS} \
        -s eval_dataset_path $DATAPATH \
        -s training.target_has_input False \
        -s training.wandb_exp_name ${EXP_NAME} \
        -s training.batch_size_per_gpu 1 -s training.dataset_path $DATAPATH \
        -s training.num_views ${NUM_TOTAL_VIEWS} -s training.num_input_views ${NUM_VIEWS} -s training.num_target_views ${NUM_VIEWS} -s training.num_virtual_views ${NUM_VIEWS} \
        -s training.checkpoint_dir ./checkpoints/${EXP_NAME} -s model.use_anything False -s model.act_ckpt False -s kmeans_input True -s training.reset_training_state False -s metrics_only False -s training.data_repeat 1 -s training.perceptual_loss_weight 0.0 -s num_frames ${FOLD_SIZE} -s training.view_selector.type kmeans \
        -s sp_size 1 -s model.gaussians.usage_threshold 0.001 -s training.torch_compile False --load ${CKPT_PATH}
