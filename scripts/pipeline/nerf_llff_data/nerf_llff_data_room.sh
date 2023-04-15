# Copy img files from ori folder to sparse folder (colmap sparse reconstruction)
# Use your target resolution
cd data
mkdir nerf_llff_data_sparse
cd nerf_llff_data_sparse
mkdir room
cd room
cd ../../
cp -r nerf_llff_data/room/images_4/. nerf_llff_data_sparse/room/images_4

# COLMAP sparse/dense reconstruction
# Refer to scripts/colmap.sh

# Here we use sparse reconstruction
# nerf_llff_data has done this already
# Run LLFF if necessary
cp -r nerf_llff_data/room/sparse/. nerf_llff_data_sparse/room/sparse
cd ../

# SAM predict
python prior/segment-anything/post_sam.py \
  --in_dir data \
  --out_dir data \
  --dataset_name nerf_llff_data \
  --scene_name room \
  --ckpt_path ckpts/sam/sam_vit_h_4b8939.pth \
  --points_json_path prior/segment-anything/points_prompt.json

# Lama
# dataset preparation
python prior/lama/pre_lama.py \
  --in_dir data/ \
  --out_dir data/ \
  --dataset_name nerf_llff_data \
  --scene_name room \
  --mask_refine_json prior/lama/mask_refine.json

# inpainting
cd prior/lama
TORCH_HOME=$(pwd)
PYTHONPATH=$(pwd)
export TORCH_HOME && export PYTHONPATH

# without refine
python3 bin/predict.py \
  model.path="$(pwd)"/../../ckpts/lama/big-lama \
  indir="$(pwd)"/../../data/nerf_llff_data_sam/room/lama \
  outdir="$(pwd)"/../../data/nerf_llff_data_sam/room/lama_out

python3 bin/predict.py \
  model.path="$(pwd)"/../../ckpts/lama/big-lama \
  indir="$(pwd)"/../../data/nerf_llff_data_sam/room/lama \
  outdir="$(pwd)"/../../data/nerf_llff_data_sam/room/lama_out_refine \
  refine=True

cd ../../

# pre nerf input
# Pay attention if we need to transform pose
python prior/nerf-pytorch/pre_nerf.py \
  --in_dir data/ \
  --out_dir data/ \
  --dataset_name nerf_llff_data \
  --scene_name room \
  --pose_transform_json prior/nerf-pytorch/pose_transform.json
cd ../

#cd prior/nerf-pytorch
## Train images before delete using nerf
#python run_nerf.py --config configs/nerf_llff_data/room.txt
#
## Train images after delete using nerf
#python run_nerf.py --config configs/nerf_llff_data/room_delete.txt
