# Run nerf training with depth prior, all mode
SCENE=$1
set -e

# Run nerf with original scenes
python nerf-depth/run_nerf_dynamic.py --config configs/comparison/nerf_depth/"$SCENE".txt
