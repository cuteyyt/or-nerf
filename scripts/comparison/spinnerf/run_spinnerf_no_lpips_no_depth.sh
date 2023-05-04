# Run spinnerf without depth prior and perceptual loss

# Params
SCENE=$1
set -e

python DS_NeRF/run_nerf.py \
  --config ../../configs/comparison/spinnerf/no_depth/"$SCENE".txt \
  --i_feat 200 \
  --i_video 10000 \
  --N_iters 10001 \
  --N_gt 0 \
  --llffhold 8 \
  --basedir ../../logs/spinnerf/ablation/no_depth \
  --no_geometry
