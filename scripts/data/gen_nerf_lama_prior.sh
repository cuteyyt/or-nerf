# Prepare a 'depth' folder for running nerf with depth prior
# note this script is not include in the 'run in one' edition as it has a little modification
# or you can add LOGDIR to run

# Params
DATASET=$1
SCENE=$2
DATADIR=$3
LOGDIR=$4
set -e

# Copy necessary files from log and sam
python datasets/post_nerf.py \
  --log_dir "$LOGDIR" \
  --in_dir "$DATADIR" \
  --dataset "$DATASET" \
  --scene "$SCENE"

# Prepare lama dataset
python datasets/pre_lama_nerf_depth.py \
  --in_dir "$DATADIR" \
  --out_dir "$DATADIR" \
  --dataset_name "$DATASET" \
  --scene_name "$SCENE" \
  --json_path configs/prepare_data/lama.json

# Lama inpainting for depth
cd prior/lama
TORCH_HOME=$(pwd) && PYTHONPATH=$(pwd)
export TORCH_HOME && export PYTHONPATH

python3 bin/predict.py \
  model.path="$(pwd)"/../../ckpts/lama/big-lama \
  indir="$(pwd)"/../../"$DATADIR"/"$DATASET"_depth/"$SCENE"/lama \
  outdir="$(pwd)"/../../"$DATADIR"/"$DATASET"_depth/"$SCENE"/lama_out_refine \
  refine=True

cd ../../

# Restore filenames from lama
python datasets/post_lama_nerf_depth.py \
  --in_dir "$DATADIR" \
  --out_dir "$DATADIR" \
  --dataset_name "$DATASET" \
  --scene_name "$SCENE" \
  --json_path configs/prepare_data/lama.json
