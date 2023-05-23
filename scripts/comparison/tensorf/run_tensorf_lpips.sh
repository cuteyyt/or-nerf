# Run tensorf original version with delete scenes
SCENE=$1
set -e

# Run tensorf with delete scenes
python comparison/Tensorf/train.py --config configs/comparison/tensorf/ablation/"$SCENE".txt --depth --Depth_loss_weight=1.0 --lpips --Lpips_loss_weight=1e-2

# Run spinnerf test
# python comparison/Tensorf/train.py --config configs/comparison/tensorf/ablation/"$SCENE".txt \
#  --datadir data/test/spinnerf_dataset_sam/"$SCENE" \
#  --basedir ./logs/test/tensorf/lpips

# python comparison/Tensorf/train.py --config configs/comparison/tensorf/ablation/"$SCENE".txt \
#  --datadir data/test/spinnerf_dataset_gt/"$SCENE" \
#  --basedir ./logs/test/tensorf/lpips
#  --ckpts ./logs/test/tensorf/lpips/"$SCENE"/"$SCENE".th
#  --render_only=1 --render_gt=1

