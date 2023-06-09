# Run tensorf original version with delete scenes
SCENE=$1
set -e

# Run tensorf with delete scenes
python comparison/Tensorf/train.py --config configs/comparison/tensorf/delete/"$SCENE".txt

# Run spinnerf test
#python comparison/Tensorf/train.py --config configs/comparison/tensorf/delete/"$SCENE".txt \
#  --datadir data/test/spinnerf_dataset_sam/"$SCENE" \
#  --basedir ./logs/test/tensorf/delete
#
#python comparison/Tensorf/train.py --config configs/comparison/tensorf/delete/"$SCENE".txt \
#  --datadir data/test/spinnerf_dataset_gt/"$SCENE" \
#  --basedir ./logs/test/tensorf/delete
#  --ckpts ./logs/test/tensorf/delete/"$SCENE"/"$SCENE".th
#  --render_only=1 --render_gt=1
