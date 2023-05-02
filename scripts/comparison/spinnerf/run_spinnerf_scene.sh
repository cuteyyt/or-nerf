DATASET=$1
SCENE=$2
DATADIR=$3
FACTOR=$4

set -e

# Generate ori depth
cd comparison/SPIN-NeRF

rm -rf lama/LaMa_test_images
rm -rf lama/output
rm -rf lama/outputs

python DS_NeRF/run_nerf.py \
  --config ../../configs/comparison/spinnerf/delete/qq3.txt \
  --prepare \
  --i_weight 1000000000 \
  --i_video 1000000000 \
  --i_feat 4000 \
  --N_iters 4001 \
  --N_gt 0

# Generate inpainted depth
cd lama
TORCH_HOME=$(pwd) && PYTHONPATH=$(pwd)
export TORCH_HOME && export PYTHONPATH

python bin/predict.py refine=True model.path="$(pwd)"/../../../ckpts/lama/big-lama \
  indir="$(pwd)"/LaMa_test_images \
  outdir="$(pwd)"/output

# Copy depth and inpainted depth to data folder
mkdir ../../../"$DATADIR"/"$DATASET"_spinnerf/"$SCENE"/images_"$FACTOR"/depth_ori
mkdir ../../../"$DATADIR"/"$DATASET"_spinnerf/"$SCENE"/images_"$FACTOR"/depth

cp LaMa_test_images/*.png ../../../"$DATADIR"/"$DATASET"_spinnerf/"$SCENE"/images_"$FACTOR"/depth_ori
cp output/label/*.png ../../../"$DATADIR"/"$DATASET"_spinnerf/"$SCENE"/images_"$FACTOR"/depth

rm -rf LaMa_test_images
rm -rf output
rm -rf outputs

# Generate inpainted rgb
mkdir LaMa_test_images
mkdir LaMa_test_images/label

cp ../../../"$DATADIR"/"$DATASET"_spinnerf/"$SCENE"/images_"$FACTOR"/*.png LaMa_test_images
cp ../../../"$DATADIR"/"$DATASET"_spinnerf/"$SCENE"/images_"$FACTOR"/label/*.png LaMa_test_images/label

python bin/predict.py refine=True model.path="$(pwd)"/../../../ckpts/lama/big-lama \
  indir="$(pwd)"/LaMa_test_images \
  outdir="$(pwd)"/output

# Copy rgb and inpainted rgb to data folder
mkdir ../../../"$DATADIR"/"$DATASET"_spinnerf/"$SCENE"/images_"$FACTOR"/lama_images
cp output/label/*.png ../../../"$DATADIR"/"$DATASET"_spinnerf/"$SCENE"/images_"$FACTOR"/lama_images

rm -rf LaMa_test_images
rm -rf output
rm -rf outputs

cd ..

# Training with spinnerf backbone
python DS_NeRF/run_nerf.py \
  --config ../../configs/comparison/spinnerf/delete/qq3.txt \
  --i_feat 200 \
  --lpips \
  --i_weight 1000000000000 \
  --i_video 10000 \
  --N_iters 10001 \
  --N_gt 0 \
  --llffhold 8
