# Config params
DATASET=$1
SCENE=$2
DATADIR=$3

# Enable quit when occurring error
set -e

# Prepare a 'sparse' folder for sam processing and running nerf without delete
python datasets/pre_sam.py \
  --in_dir "$DATADIR" \
  --out_dir "$DATADIR" \
  --dataset_name "$DATASET" \
  --scene_name "$SCENE" \
  --json_path configs/prepare_data/sam.json

# Use colmap to reconstruct cam params
cd prior/LLFF
python imgs2poses.py ../../"$DATADIR"/"$DATASET"_sparse/"$SCENE"
cd ../../

# If LLFF ori code fails to reconstruct cam params,
# you need to reconstruct it manually using the following codes
# Specify path
#DATASET_PATH="$DATADIR"/"$DATASET"_sparse/"$SCENE"
#
#colmap feature_extractor \
#  --database_path "$DATASET_PATH"/database.db \
#  --image_path "$DATASET_PATH"/images \
#  --ImageReader.single_camera '1'
#
#colmap exhaustive_matcher \
#  --database_path "$DATASET_PATH"/database.db
#
#mkdir "$DATASET_PATH"/sparse
#
#colmap mapper \
#  --database_path "$DATASET_PATH"/database.db \
#  --image_path "$DATASET_PATH"/images \
#  --output_path "$DATASET_PATH"/sparse
