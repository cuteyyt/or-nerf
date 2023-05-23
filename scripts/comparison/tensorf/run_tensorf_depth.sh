# Run tensorf original version with delete scenes
SCENE=$1
set -e

# Run tensorf with delete scenes
python comparison/Tensorf/train.py --config configs/comparison/tensorf/ablation/"$SCENE".txt --depth --Depth_loss_weight=1.0

# Run spinnerf test
# python comparison/Tensorf/train.py --config configs/comparison/tensorf/ablation/"$SCENE".txt \
#  --datadir data/test/spinnerf_dataset_sam/"$SCENE" \
#  --basedir ./logs/test/tensorf/da

# python comparison/Tensorf/train.py --config configs/comparison/tensorf/ablation/"$SCENE".txt \
#  --datadir data/test/spinnerf_dataset_gt/"$SCENE" \
#  --basedir ./logs/test/tensorf/da
#  --ckpts ./logs/test/tensorf/da/"$SCENE"/"$SCENE".th
#  --render_only=1 --render_gt=1

