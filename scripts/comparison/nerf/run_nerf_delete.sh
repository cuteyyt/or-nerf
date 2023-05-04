# Run nerf original version with delete scenes
SCENE=$1
set -e

# Run nerf with original scenes
python comparison/nerf-pytorch/run_nerf.py --config configs/comparison/nerf/delete/"$SCENE".txt
python comparison/nerf-pytorch/run_nerf.py --config configs/comparison/nerf/delete/"$SCENE".txt --render_all
