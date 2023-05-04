# Prepare a 'lama' folder to generate lama prior according to img and corresponding mask

# Params
DATASET=$1
SCENE=$2
DATADIR=$3

set -e

# Lama dataset preparation
python datasets/pre_lama.py \
  --in_dir "$DATADIR" \
  --out_dir "$DATADIR" \
  --dataset_name "$DATASET" \
  --scene_name "$SCENE" \
  --json_path configs/prepare_data/lama.json

# Lama inpainting
cd prior/lama
TORCH_HOME=$(pwd) && PYTHONPATH=$(pwd)
export TORCH_HOME && export PYTHONPATH

python3 bin/predict.py \
  model.path="$(pwd)"/../../ckpts/lama/big-lama \
  indir="$(pwd)"/../../"$DATADIR"/"$DATASET"_sam/"$SCENE"/lama \
  outdir="$(pwd)"/../../"$DATADIR"/"$DATASET"_sam/"$SCENE"/lama_out_refine \
  refine=True

cd ../../

# Post lama
python datasets/post_lama.py \
  --in_dir "$DATADIR" \
  --out_dir "$DATADIR" \
  --dataset_name "$DATASET" \
  --scene_name "$SCENE" \
  --json_path configs/prepare_data/lama.json
