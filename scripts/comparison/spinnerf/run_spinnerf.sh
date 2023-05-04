# Run spinnerf with original version

# Params
SCENE=$1
set -e

python DS_NeRF/run_nerf.py \
  --config ../../configs/comparison/spinnerf/delete/"$SCENE".txt \
  --i_feat 200 \
  --lpips \
  --i_video 10000 \
  --N_iters 10001 \
  --N_gt 0 \
  --llffhold 8
