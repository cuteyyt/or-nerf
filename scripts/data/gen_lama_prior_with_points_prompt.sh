# Config params
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
  --json_path configs/prepare_data/sam.json \
  --ckpt_path ckpts/sam/sam_vit_h_4b8939.pth

# Lama dataset preparation
python datasets/pre_lama.py \
  --in_dir "$DATADIR" \
  --out_dir "$DATADIR" \
  --dataset_name "$DATASET" \
  --scene_name "$SCENE" \
  --json_path configs/prepare_data/lama.json

# Lama inpainting
cd prior/lama
TORCH_HOME=$(pwd)
PYTHONPATH=$(pwd)
export TORCH_HOME && export PYTHONPATH

python3 bin/predict.py \
  model.path="$(pwd)"/../../ckpts/lama/big-lama \
  indir="$(pwd)"/../../"$DATADIR"/"$DATASET"_sam/"$SCENE"/lama \
  outdir="$(pwd)"/../../"$DATADIR"/"$DATASET"_sam/"$SCENE"/lama_out_refine \
  refine=True

cd ../../

# Post lama (merge pictures to videos)
python datasets/post_lama.py \
  --in_dir "$DATADIR" \
  --out_dir "$DATADIR" \
  --dataset_name "$DATASET" \
  --scene_name "$SCENE" \
  --json_path configs/prepare_data/lama.json
