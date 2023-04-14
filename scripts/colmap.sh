# Sparse Reconstruction

# Dense Reconstruction

# Specify path
DATASET_PATH=data/nerf_llff_data_dense/room

# Colmap dense reconstruction in one-line
colmap automatic_reconstructor --image_path $DATASET_PATH/images --workspace_path $DATASET_PATH

# Or you can
# Colmap dense reconstruction step by step

# The same as LLFF https://github.com/Fyusion/LLFF
colmap feature_extractor \
  --database_path $DATASET_PATH/database.db \
  --image_path $DATASET_PATH/images \
  --ImageReader.single_camera '1'

colmap exhaustive_matcher \
  --database_path $DATASET_PATH/database.db

mkdir $DATASET_PATH/sparse

colmap mapper \
  --database_path $DATASET_PATH/database.db \
  --image_path $DATASET_PATH/images \
  --output_path $DATASET_PATH/sparse

mkdir $DATASET_PATH/dense/0

colmap image_undistorter \
  --image_path $DATASET_PATH/images \
  --input_path $DATASET_PATH/sparse/0 \
  --output_path $DATASET_PATH/dense/0 \
  --output_type COLMAP

colmap patch_match_stereo \
  --workspace_path $DATASET_PATH/dense/0 \
  --workspace_format COLMAP \
  --PatchMatchStereo.geom_consistency true

colmap stereo_fusion \
  --workspace_path $DATASET_PATH/dense/0 \
  --workspace_format COLMAP \
  --input_type geometric \
  --output_path $DATASET_PATH/dense/0/fused.ply

colmap poisson_mesher \
  --input_path $DATASET_PATH/dense/0/fused.ply \
  --output_path $DATASET_PATH/dense/0/meshed-poisson.ply

colmap delaunay_mesher \
  --input_path $DATASET_PATH/dense/0 \
  --output_path $DATASET_PATH/dense/0/meshed-delaunay.ply

# Convert .bin cam files to .txt
colmap model_converter \
  --input_path $DATASET_PATH/sparse/0 \
  --output_path $DATASET_PATH/sparse/0 \
  --output_type TXT

colmap model_converter \
  --input_path $DATASET_PATH/dense/0/sparse \
  --output_path $DATASET_PATH/dense/0/sparse \
  --output_type TXT
