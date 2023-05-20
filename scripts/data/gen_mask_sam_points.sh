# Prepare a 'sam' folder for running scenes with delete and lama-processing

# Params
DATASET=$1
SCENE=$2
INDIR=$3
OUTDIR=$4

set -e

# SAM predict (use --is_test to test spinnerf)
python datasets/run_sam_points.py \
  --in_dir "$INDIR" \
  --out_dir "$OUTDIR" \
  --dataset_name "$DATASET" \
  --scene_name "$SCENE" \
  --json_path configs/prepare_data/sam_points.json \
  --ckpt_path ckpts/sam/sam_vit_h_4b8939.pth

# --text_prompt to enable text prompt (use --is_test to test spinnerf)
python datasets/run_sam_points.py \
  --in_dir "$INDIR" \
  --out_dir "$OUTDIR" \
  --dataset_name "$DATASET" \
  --scene_name "$SCENE" \
  --json_path configs/prepare_data/sam_points.json \
  --ckpt_path ckpts/sam/sam_vit_h_4b8939.pth \
  --text_prompt
