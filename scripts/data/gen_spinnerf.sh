# Prepare lama priors for running spinnerf
# note this script is not include in the 'run in one' edition as it has a little modification
# or you can add FACTOR to run

# Params
DATASET=$1
SCENE=$2
INDIR=$3
OUTDIR=$4
FACTOR=$5

set -e

# Gen spinnerf dataset folder
python datasets/pre_spinnerf.py \
  --in_dir "$INDIR" \
  --out_dir "$OUTDIR" \
  --dataset_name "$DATASET" \
  --scene_name "$SCENE" \
  --json_path configs/prepare_data/lama.json

# Generate original depth
cd comparison/SPIn-NeRF

rm -rf lama/LaMa_test_images
rm -rf lama/output
rm -rf lama/outputs

python DS_NeRF/run_nerf.py \
  --config ../../configs/comparison/spinnerf/delete/"$SCENE".txt \
  --prepare \
  --i_weight 1000000000 \
  --i_video 1000000000 \
  --i_feat 4000 \
  --N_iters 4001 \
  --N_gt 0 \
  --datadir ../../"$OUTDIR"/"$DATASET"_spinnerf/"$SCENE" \
  --basedir ../../logs/spinnerf/delete

# Generate inpainted depth
cd lama
TORCH_HOME=$(pwd) && PYTHONPATH=$(pwd)
export TORCH_HOME && export PYTHONPATH

python bin/predict.py refine=True model.path="$(pwd)"/../../../ckpts/lama/big-lama \
  indir="$(pwd)"/LaMa_test_images \
  outdir="$(pwd)"/output

# Copy depth and inpainted depth to data folder
mkdir ../../../"$OUTDIR"/"$DATASET"_spinnerf/"$SCENE"/images_"$FACTOR"/depth_ori
mkdir ../../../"$OUTDIR"/"$DATASET"_spinnerf/"$SCENE"/images_"$FACTOR"/depth

cp LaMa_test_images/*.png ../../../"$OUTDIR"/"$DATASET"_spinnerf/"$SCENE"/images_"$FACTOR"/depth_ori
cp output/label/*.png ../../../"$OUTDIR"/"$DATASET"_spinnerf/"$SCENE"/images_"$FACTOR"/depth

rm -rf LaMa_test_images
rm -rf output
rm -rf outputs

# Generate inpainted rgb
mkdir LaMa_test_images
mkdir LaMa_test_images/label

cp ../../../"$OUTDIR"/"$DATASET"_spinnerf/"$SCENE"/images_"$FACTOR"/*.png LaMa_test_images
cp ../../../"$OUTDIR"/"$DATASET"_spinnerf/"$SCENE"/images_"$FACTOR"/label/*.png LaMa_test_images/label

python bin/predict.py refine=True model.path="$(pwd)"/../../../ckpts/lama/big-lama \
  indir="$(pwd)"/LaMa_test_images \
  outdir="$(pwd)"/output

# Copy rgb and inpainted rgb to data folder
mkdir ../../../"$OUTDIR"/"$DATASET"_spinnerf/"$SCENE"/images_"$FACTOR"/lama_images
cp output/label/*.png ../../../"$OUTDIR"/"$DATASET"_spinnerf/"$SCENE"/images_"$FACTOR"/lama_images

rm -rf LaMa_test_images
rm -rf output
rm -rf outputs

cd ..
