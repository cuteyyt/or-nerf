# Run nerf original version with original scenes
SCENE=$1
set -e

# Run nerf with original scenes
python comparison/nerf-pytorch/run_nerf.py --config configs/comparison/nerf/ori/"$SCENE".txt

# Run nerf original version with original scenes (after training) to render all views' rgb and depth
# This must carry out after the training is complete to run further nerf + depth experiments
python comparison/nerf-pytorch/run_nerf.py --config configs/comparison/nerf/ori/"$SCENE".txt --render_all

# To run spinnerf test
CUDA_VISIBLE_DEVICES=1 python comparison/nerf-pytorch/run_nerf.py --config configs/comparison/nerf/ori/"$SCENE".txt \
  --datadir data/test/spinnerf_dataset_sparse/"$SCENE" \
  --basedir ./logs/nerf/ori/test
