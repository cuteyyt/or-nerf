# Prepare a 'sparse' folder for running scenes without delete and sam-processing

# Params
DATASET=$1
SCENE=$2
INDIR=$3
OUTDIR=$4

# Enable quit when occurring error
set -e

# Create the 'sparse' folder (use --is_test to test spinnerf)
python datasets/pre_sparse.py \
  --in_dir "$INDIR" \
  --out_dir "$OUTDIR" \
  --dataset_name "$DATASET" \
  --scene_name "$SCENE" \
  --json_path configs/prepare_data/sam_points.json

# Use LLFF to reconstruct cam params
cd prior/LLFF
python imgs2poses.py ../../"$OUTDIR"/"$DATASET"_sparse/"$SCENE"
cd ../../

# In case LLFF fails to reconstruct cam params,
# you may reconstruct cam params manually using the following commands

## First remove broken reconstruction files
#rm -rf "$OUTDIR"/"$DATASET"_sparse/"$SCENE"/sparse
#rm -rf "$OUTDIR"/"$DATASET"_sparse/"$SCENE"/database.db
#rm -rf "$OUTDIR"/"$DATASET"_sparse/"$SCENE"/colmap_output.txt
#
## Then reconstruct
#DATASET_PATH="$OUTDIR"/"$DATASET"_sparse/"$SCENE"
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
#python imgs2poses.py ../../"$OUTDIR"/"$DATASET"_sparse/"$SCENE"
#cd ../../
