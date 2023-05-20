# Prepare a 'text' folder for text prompts

# Config params
DATASET=$1
SCENE=$2
INDIR=$3
OUTDIR=$4

set -e

# SAM predict with text prompt  (use --is_test to test spinnerf, actually no effect)
# Use hybrid to give the initial mask ONLY
python datasets/run_sam_text.py \
  --in_dir "$INDIR" \
  --out_dir "$OUTDIR" \
  --dataset "$DATASET" \
  --scene "$SCENE" \
  --json_path configs/prepare_data/sam_text.json \
  --hybrid
