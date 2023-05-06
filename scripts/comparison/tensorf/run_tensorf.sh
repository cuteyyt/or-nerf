# Run tensorf original version with delete scenes
SCENE=$1
set -e

# Run tensorf with original scenes
python comparison/TensoRF/train.py --config configs/comparison/tensorf/${SCENE}.txt
