# Run nerf original version with delete scenes
SCENE=$1
set -e

# Run nerf with delete scenes
python comparison/nerf-pytorch/run_nerf.py --config configs/comparison/nerf/delete/"$SCENE".txt
python comparison/nerf-pytorch/run_nerf.py --config configs/comparison/nerf/delete/"$SCENE".txt --render_all

# Run spinnerf test
#python comparison/nerf-pytorch/run_nerf_test.py --config configs/comparison/nerf/delete/"$SCENE".txt \
#  --datadir data/test/spinnerf_dataset_sam/"$SCENE" \
#  --basedir ./logs/test/nerf/delete
#
#python comparison/nerf-pytorch/run_nerf_test.py --config configs/comparison/nerf/delete/"$SCENE".txt \
#  --datadir data/test/spinnerf_dataset_sam/"$SCENE" \
#  --basedir ./logs/test/nerf/delete \
#  --render_all
#
#python comparison/nerf-pytorch/run_nerf_test.py --config configs/comparison/nerf/delete/"$SCENE".txt \
#  --datadir data/test/spinnerf_dataset_gt/"$SCENE" \
#  --basedir ./logs/test/nerf/delete \
#  --render_all
