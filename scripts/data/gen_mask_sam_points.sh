# Prepare a 'sam' folder for running scenes with delete and lama-processing

# Params
DATASET=$1
SCENE=$2
DATADIR=$3

set -e

# SAM predict
python datasets/post_sam.py \
--in_dir "$DATADIR" \
--out_dir "$DATADIR" \
--dataset_name "$DATASET" \
--scene_name "$SCENE" \
--json_path configs/prepare_data/sam_points.json \
--ckpt_path ckpts/sam/sam_vit_h_4b8939.pth
