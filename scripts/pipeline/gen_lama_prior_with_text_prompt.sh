# Config params
DATASET=$1
SCENE=$2
DATADIR=$3

set -e

# Prepare a 'sparse' folder for sam processing and running NeRF without delete
python datasets/pre_sam.py \
  --in_dir "${DATADIR}" \
  --out_dir "${DATADIR}" \
  --dataset_name "${DATASET}" \
  --scene_name "${SCENE}" \
  --json_path configs/prepare_data/sam.json

# SAM predict with text prompt
python datasets/pre_text_prompt \
  --config prior/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint ckpts/grounded_sam/groundingdino_swint_ogc.pth \
  --sam_checkpoint ckpts/sam/sam_vit_h_4b8939.pth \
  --dataset "${DATASET}" \
  --scene "${SCENE}" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt_json configs/prepare_data/text_prompt.json \
  --device "cuda"


# Lama dataset preparation
python datasets/pre_lama.py \
  --in_dir "${DATADIR}" \
  --out_dir "${DATADIR}" \
  --dataset_name "${DATASET}" \
  --scene_name "${SCENE}" \
  --json_path configs/prepare_data/lama.json

# Lama inpainting
cd prior/lama
TORCH_HOME=$(pwd)
PYTHONPATH=$(pwd)
export TORCH_HOME && export PYTHONPATH

python3 bin/predict.py \
  model.path="$(pwd)"/../../ckpts/lama/big-lama \
  indir="$(pwd)"/../../"${DATADIR}"/"${DATASET}"_sam/"${SCENE}"/lama \
  outdir="$(pwd)"/../../"${DATADIR}"/"${DATASET}"_sam/"${SCENE}"/lama_out_refine \
  refine=True

# Post lama (merge pictures to videos)
cd ../../
python datasets/post_lama.py \
  --in_dir "${DATADIR}" \
  --out_dir "${DATADIR}" \
  --dataset_name "${DATASET}" \
  --scene_name "${SCENE}" \
  --json_path configs/prepare_data/lama.json
