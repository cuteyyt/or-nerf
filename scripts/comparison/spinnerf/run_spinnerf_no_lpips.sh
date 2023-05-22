# Run spinnerf without perceptual loss

# Params
SCENE=$1
set -e

python DS_NeRF/run_nerf.py \
  --config ../../configs/comparison/spinnerf/delete/"$SCENE".txt \
  --i_feat 200 \
  --i_video 10000 \
  --N_iters 10001 \
  --N_gt 0 \
  --llffhold 8 \
  --basedir ../../logs/spinnerf/ablation/no_lpips \
  --datadir ../../data/spinnerf_dataset_spinnerf/"$SCENE"

python DS_NeRF/run_nerf.py \
  --config ../../configs/comparison/spinnerf/delete/"$SCENE".txt \
  --i_feat 200 \
  --i_video 10000 \
  --N_iters 10001 \
  --N_gt 0 \
  --llffhold 8 \
  --basedir ../../logs/spinnerf/ablation/no_lpips \
  --datadir ../../data/spinnerf_dataset_spinnerf/"$SCENE" \
  --render_all

# To run spinnerf test
#python DS_NeRF/run_nerf_test.py \
#  --config ../../configs/comparison/spinnerf/delete/"$SCENE".txt \
#  --i_feat 200 \
#  --i_video 10000 \
#  --N_iters 10001 \
#  --N_gt 0 \
#  --llffhold 8 \
#  --basedir ../../logs/test/spinnerf/ablation/no_lpips \
#  --datadir ../../data/test/spinnerf_dataset_spinnerf/"$SCENE"
#
#python DS_NeRF/run_nerf_test.py \
#  --config ../../configs/comparison/spinnerf/delete/"$SCENE".txt \
#  --i_feat 200 \
#  --i_video 10000 \
#  --N_iters 10001 \
#  --N_gt 0 \
#  --llffhold 8 \
#  --basedir ../../logs/test/spinnerf/ablation/no_lpips \
#  --datadir ../../data/test/spinnerf_dataset_spinnerf/"$SCENE" \
#  --render_all
#
#python DS_NeRF/run_nerf_test.py \
#  --config ../../configs/comparison/spinnerf/delete/"$SCENE".txt \
#  --i_feat 200 \
#  --i_video 10000 \
#  --N_iters 10001 \
#  --N_gt 0 \
#  --llffhold 8 \
#  --basedir ../../logs/test/spinnerf/ablation/no_lpips \
#  --datadir ../../data/test/spinnerf_dataset_spinnerf/"$SCENE" \
#  --render_all --render_gt
