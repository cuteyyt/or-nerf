# Prepare test gt from spinnerf

# Params
DATASET=$1
SCENE=$2
INDIR=$3
OUTDIR=$4

set -e

# Copy necessary files from sparse folder
cp -r "$INDIR"/"$DATASET"_sparse/"$SCENE"/sparse/0/* "$OUTDIR"/"$DATASET"_gt/"$SCENE"/sparse/0
cp -r "$INDIR"/"$DATASET"_sparse/"$SCENE"/poses_bounds.npy "$OUTDIR"/"$DATASET"_gt/"$SCENE"/poses_bounds.npy
cp -r "$INDIR"/"$DATASET"_sparse/"$SCENE"/images/* "$OUTDIR"/"$DATASET"_gt/"$SCENE"/images
cp -r "$INDIR"/"$DATASET"_sparse/"$SCENE"/images_gt/* "$OUTDIR"/"$DATASET"_gt/"$SCENE"/images_4
