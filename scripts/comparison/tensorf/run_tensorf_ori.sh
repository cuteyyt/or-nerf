# Run tensorf original version with original scenes
SCENE=$1
set -e

# Run tensorf with original scenes
python comparison/Tensorf/train.py --config configs/comparison/tensorf/ori/"$SCENE".txt

# To run spinnerf test
#python comparison/Tensorf/train.py --config configs/comparison/tensorf/ori/"$SCENE".txt \
#  --datadir data/test/spinnerf_dataset_sparse/"$SCENE" \
#  --basedir ./logs/test/nerf/ori
