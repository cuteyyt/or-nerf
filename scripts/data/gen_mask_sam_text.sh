# Prepare a 'text' folder for text prompts

# Config params
DATASET=$1
SCENE=$2
DATADIR=$3

set -e

# SAM predict with text prompt
python datasets/pre_text_prompt.py \
  --config prior/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint ckpts/grounded_sam/groundingdino_swint_ogc.pth \
  --sam_checkpoint ckpts/sam/sam_vit_h_4b8939.pth \
  --dataset "$DATASET" \
  --scene "$SCENE" \
  --text_prompt_json configs/prepare_data/sam_text.json \
  --device "cuda" \
  --hybrid
