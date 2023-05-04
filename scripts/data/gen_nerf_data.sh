# Prepare a 'sparse' folder for running scenes without delete and sam-processing

# Params
DATASET=$1
SCENE=$2
DATADIR=$3

# Enable quit when occurring error
set -e

# Create the 'sparse' folder
python datasets/pre_nerf.py \
  --in_dir "$DATADIR" \
  --out_dir "$DATADIR" \
  --dataset_name "$DATASET" \
  --scene_name "$SCENE" \
  --json_path configs/prepare_data/sam.json

# Use LLFF to reconstruct cam params
cd prior/LLFF
python imgs2poses.py ../../"$DATADIR"/"$DATASET"_sparse/"$SCENE"
cd ../../

# In case LLFF fails to reconstruct cam params,
# you may reconstruct cam params manually using the following commands

## First remove broken reconstruction files
#rm -rf "$DATADIR"/"$DATASET"_sparse/"$SCENE"/sparse
#rm -rf "$DATADIR"/"$DATASET"_sparse/"$SCENE"/database.db
#rm -rf "$DATADIR"/"$DATASET"_sparse/"$SCENE"/colmap_output.txt
#
## Then reconstruct
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
#
## Finally run imgs2poses again to generate poses_bounds.npy
#cd prior/LLFF
#python imgs2poses.py ../../"$DATADIR"/"$DATASET"_sparse/"$SCENE"
#cd ../../
