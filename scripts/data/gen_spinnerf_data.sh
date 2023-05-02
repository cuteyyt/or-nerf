DATASET=$1
SCENE=$2
DATADIR=$3

set -e

python datasets/pre_spinnerf.py \
  --in_dir "$DATADIR" \
  --out_dir "$DATADIR" \
  --dataset_name "$DATASET" \
  --scene_name "$SCENE" \
  --json_path configs/prepare_data/lama.json
