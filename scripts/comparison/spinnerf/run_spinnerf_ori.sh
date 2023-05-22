# Run spinnerf with original version

# Params
SCENE=$1
set -e

# Run spinnerf with original version
python DS_NeRF/run_nerf.py \
  --config ../../configs/comparison/spinnerf/delete/"$SCENE".txt \
  --i_feat 200 \
  --lpips \
  --i_video 10000 \
  --N_iters 10001 \
  --N_gt 0 \
  --llffhold 8

python DS_NeRF/run_nerf.py \
  --config ../../configs/comparison/spinnerf/delete/"$SCENE".txt \
  --i_feat 200 \
  --lpips \
  --i_video 10000 \
  --N_iters 10001 \
  --N_gt 0 \
  --llffhold 8 \
  --render_all

# To run spinnerf test
#python DS_NeRF/run_nerf_test.py \
#  --config ../../configs/comparison/spinnerf/delete/"$SCENE".txt \
#  --i_feat 200 \
#  --lpips \
#  --i_video 10000 \
#  --N_iters 10001 \
#  --N_gt 0 \
#  --llffhold 8 \
#  --datadir ../../data/test/spinnerf_dataset_spinnerf/"$SCENE" \
#  --basedir ../../logs/test/spinnerf/delete
#
#python DS_NeRF/run_nerf_test.py \
#  --config ../../configs/comparison/spinnerf/delete/"$SCENE".txt \
#  --i_feat 200 \
#  --lpips \
#  --i_video 10000 \
#  --N_iters 10001 \
#  --N_gt 0 \
#  --llffhold 8 \
#  --datadir ../../data/test/spinnerf_dataset_spinnerf/"$SCENE" \
#  --basedir ../../logs/test/spinnerf/delete \
#  --render_all
#
#python DS_NeRF/run_nerf_test.py \
#  --config ../../configs/comparison/spinnerf/delete/"$SCENE".txt \
#  --i_feat 200 \
#  --lpips \
#  --i_video 10000 \
#  --N_iters 10001 \
#  --N_gt 0 \
#  --llffhold 8 \
#  --datadir ../../data/test/spinnerf_dataset_spinnerf/"$SCENE" \
#  --basedir ../../logs/test_2/spinnerf/delete \
#  --render_all --render_gt
