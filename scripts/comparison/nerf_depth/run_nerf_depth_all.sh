# Run nerf training with depth prior
SCENE=$1
set -e

# Run nerf with original scenes
python nerf-depth/run_nerf.py --config configs/comparison/nerf_depth/"$SCENE".txt --N_rand 2048
