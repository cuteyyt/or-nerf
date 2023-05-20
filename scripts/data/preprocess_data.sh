# Specify a script name to run all scenes in one

# Params
SCRIPT=$1
INDIR=$2
OUTDIR=$3
set -e

# ibrnet_data
sh "$SCRIPT" ibrnet_data qq3 "$INDIR" "$OUTDIR"
sh "$SCRIPT" ibrnet_data qq6 "$INDIR" "$OUTDIR"
sh "$SCRIPT" ibrnet_data qq10 "$INDIR" "$OUTDIR"
sh "$SCRIPT" ibrnet_data qq11 "$INDIR" "$OUTDIR"
sh "$SCRIPT" ibrnet_data qq13 "$INDIR" "$OUTDIR"
sh "$SCRIPT" ibrnet_data qq16 "$INDIR" "$OUTDIR"
sh "$SCRIPT" ibrnet_data qq17 "$INDIR" "$OUTDIR"
sh "$SCRIPT" ibrnet_data qq21 "$INDIR" "$OUTDIR"

# llff_real_iconic
sh "$SCRIPT" llff_real_iconic data5_piano "$INDIR" "$OUTDIR"

# nerf_llff_data
sh "$SCRIPT" nerf_llff_data room "$INDIR" "$OUTDIR"
sh "$SCRIPT" nerf_llff_data horns "$INDIR" "$OUTDIR"
sh "$SCRIPT" nerf_llff_data fortress "$INDIR" "$OUTDIR"

# spinnerf_dataset
sh "$SCRIPT" spinnerf_dataset 2 "$INDIR" "$OUTDIR"
sh "$SCRIPT" spinnerf_dataset 3 "$INDIR" "$OUTDIR"
sh "$SCRIPT" spinnerf_dataset 4 "$INDIR" "$OUTDIR"
sh "$SCRIPT" spinnerf_dataset 7 "$INDIR" "$OUTDIR"
sh "$SCRIPT" spinnerf_dataset 10 "$INDIR" "$OUTDIR"
sh "$SCRIPT" spinnerf_dataset 12 "$INDIR" "$OUTDIR"
sh "$SCRIPT" spinnerf_dataset book "$INDIR" "$OUTDIR"
sh "$SCRIPT" spinnerf_dataset trash "$INDIR" "$OUTDIR"
