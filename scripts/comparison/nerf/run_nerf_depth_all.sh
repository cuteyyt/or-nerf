# Run nerf training with depth prior, all mode
SCENE=$1
set -e

# Run nerf da
python comparison/nerf-pytorch/run_nerf_depth.py --config configs/comparison/nerf/ablation/"$SCENE".txt --mode depth_all
python comparison/nerf-pytorch/run_nerf_depth.py --config configs/comparison/nerf/ablation/"$SCENE".txt --mode depth_all --render_all

# Run spinnerf test
#python nerf-depth/run_nerf_depth.py --config configs/comparison/nerf/ablation/"$SCENE".txt --mode depth_all \
#  --datadir data/test/spinnerf_dataset_depth/"$SCENE" \
#  --basedir ./logs/test/nerf
#
#python nerf-depth/run_nerf_depth.py --config configs/comparison/nerf/ablation/"$SCENE".txt --mode depth_all \
#  --datadir data/test/spinnerf_dataset_depth/"$SCENE" \
#  --basedir ./logs/test/nerf --render_gt
#
#python nerf-depth/run_nerf_depth.py --config configs/comparison/nerf/ablation/"$SCENE".txt --mode depth_all \
#  --datadir data/test/spinnerf_dataset_depth/"$SCENE" \
#  --basedir ./logs/test/nerf --render_gt --render_all
