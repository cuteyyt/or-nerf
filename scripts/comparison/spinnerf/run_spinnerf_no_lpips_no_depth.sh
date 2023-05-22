# Run spinnerf without depth prior and perceptual loss

# Params
SCENE=$1
set -e

python DS_NeRF/run_nerf.py \
  --config ../../configs/comparison/spinnerf/ablation/"$SCENE".txt \
  --i_feat 200 \
  --i_video 10000 \
  --N_iters 10001 \
  --N_gt 0 \
  --llffhold 8 \
  --basedir ../../logs/spinnerf/ablation/no_depth \
  --datadir ../../data/test/spinnerf_dataset_spinnerf/"$SCENE" \
  --no_geometry

python DS_NeRF/run_nerf.py \
  --config ../../configs/comparison/spinnerf/ablation/"$SCENE".txt \
  --i_feat 200 \
  --i_video 10000 \
  --N_iters 10001 \
  --N_gt 0 \
  --llffhold 8 \
  --basedir ../../logs/spinnerf/ablation/no_depth \
  --datadir ../../data/spinnerf_dataset_spinnerf/"$SCENE" \
  --no_geometry \
  --render_all

# To run spinnerf test
#python DS_NeRF/run_nerf_test.py \
#  --config ../../configs/comparison/spinnerf/ablation/"$SCENE".txt \
#  --i_feat 200 \
#  --i_video 10000 \
#  --N_iters 10001 \
#  --N_gt 0 \
#  --llffhold 8 \
#  --basedir ../../logs/test/spinnerf/ablation/no_depth \
#  --datadir ../../data/test/spinnerf_dataset_spinnerf/"$SCENE" \
#  --no_geometry
#
#python DS_NeRF/run_nerf_test.py \
#  --config ../../configs/comparison/spinnerf/ablation/"$SCENE".txt \
#  --i_feat 200 \
#  --i_video 10000 \
#  --N_iters 10001 \
#  --N_gt 0 \
#  --llffhold 8 \
#  --basedir ../../logs/test/spinnerf/ablation/no_depth \
#  --datadir ../../data/test/spinnerf_dataset_spinnerf/"$SCENE" \
#  --no_geometry \
#  --render_all
#
#python DS_NeRF/run_nerf_test.py \
#  --config ../../configs/comparison/spinnerf/ablation/"$SCENE".txt \
#  --i_feat 200 \
#  --i_video 10000 \
#  --N_iters 10001 \
#  --N_gt 0 \
#  --llffhold 8 \
#  --basedir ../../logs/test/spinnerf/ablation/no_depth \
#  --datadir ../../data/test/spinnerf_dataset_spinnerf/"$SCENE" \
#  --no_geometry \
#  --render_all --render_gt
