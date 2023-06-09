# Prepare a 'lama' folder to generate lama prior according to img and corresponding mask

# Params
DATASET=$1
SCENE=$2
INDIR=$3
OUTDIR=$4

set -e

SFX=sam
# Lama dataset preparation
python datasets/pre_lama.py \
  --in_dir "$INDIR" \
  --out_dir "$OUTDIR" \
  --dataset_name "$DATASET" \
  --scene_name "$SCENE" \
  --json_path configs/prepare_data/lama.json \
  --sfx "$SFX"

# Lama inpainting
cd prior/lama
TORCH_HOME=$(pwd) && PYTHONPATH=$(pwd)
export TORCH_HOME && export PYTHONPATH

python3 bin/predict.py \
  model.path="$(pwd)"/../../ckpts/lama/big-lama \
  indir="$(pwd)"/../../"$OUTDIR"/"$DATASET"_"$SFX"/"$SCENE"/lama \
  outdir="$(pwd)"/../../"$OUTDIR"/"$DATASET"_"$SFX"/"$SCENE"/lama_out_refine \
  refine=True

cd ../../

# Post lama
python datasets/post_lama.py \
  --in_dir "$OUTDIR" \
  --out_dir "$OUTDIR" \
  --dataset_name "$DATASET" \
  --scene_name "$SCENE" \
  --json_path configs/prepare_data/lama.json \
  --sfx "$SFX"
