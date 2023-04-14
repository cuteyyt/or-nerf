# Copy img files from ori folder to sparse folder (colmap sparse reconstruction)
# Use your target resolution
cd cor-nerf/data
mkdir nerf_llff_data_sparse
cd nerf_llff_data_sparse
mkdir room
cd room
mkdir images
cd ../../
cp -r nerf_llff_data/room/images_4/. nerf_llff_data_sparse/room/images

# COLMAP sparse/dense reconstruction
# Refer to scripts/colmap.sh

# Here we use sparse reconstruction
# nerf_llff_data has done this already
# Run LLFF if necessary
cp -r nerf_llff_data/room/sparse/. nerf_llff_data_sparse/room/sparse
cd ../

# SAM predict
cd cor-nerf
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
cd prior/lama || exit
TORCH_HOME=$(pwd)
PYTHONPATH=$(pwd)
export TORCH_HOME && export PYTHONPATH

# without refine
python3 bin/predict.py \
  model.path="$(pwd)"/../../ckpts/lama/big-lama \
  indir="$(pwd)"/../../data/statue-sam/lama \
  outdir="$(pwd)"/../../data/statue-sam/lama_out

python3 bin/predict.py \
  model.path="$(pwd)"/../../ckpts/lama/big-lama \
  indir="$(pwd)"/../../data/statue-sam/lama \
  outdir="$(pwd)"/../../data/statue-sam/lama_refine_out \
  refine=True

# pre nerf input to avoid pose mismatch
cp -r nerf_llff_data/room/pose_bounds.npy nerf_llff_data_sparse/room/pose_bounds.npy
cd prior/nerf || exit
python preprocess.py

cd prior/LLFF || exit
python imgs2poses.py ../../data/statue-sam/

# Train images before delete using nerf
CUDA_VISIBLE_DEVICES=0 python run_nerf.py --config configs/statue.txt

# Train images after delete using nerf
CUDA_VISIBLE_DEVICES=1 python run_nerf.py --config configs/statue_delete.txt

# TODO: seems train directly with NeRF has some problems
# use another dataset

# SAM predict and lama dataset preparation
cd prior/segment-angthing || exit
python preprocess.py --in_dir ../../data/spinnerf-dataset/10 --out_dir ../../data/spinnerf-dataset/10-sam

# Lama inpainting
cd prior/lama || exit
TORCH_HOME=$(pwd)
PYTHONPATH=$(pwd)
export TORCH_HOME && export PYTHONPATH

# without refine
python3 bin/predict.py \
  model.path="$(pwd)"/../../ckpts/lama/big-lama \
  indir="$(pwd)"/../../data/spinnerf-dataset/10-sam/lama \
  outdir="$(pwd)"/../../data/spinnerf-dataset/10-sam/lama_out

python3 bin/predict.py \
  model.path="$(pwd)"/../../ckpts/lama/big-lama \
  indir="$(pwd)"/../../data/spinnerf-dataset/10-sam/lama \
  outdir="$(pwd)"/../../data/spinnerf-dataset/10-sam/lama_refine_out \
  refine=True

# pre nerf input to avoid pose mismatch
cd prior/nerf-pytorch || exit
python preprocess.py --in_dir ../../data/spinnerf-dataset/10 --out_dir ../../data/spinnerf-dataset/10-sam

cd prior/LLFF || exit
python imgs2poses.py ../../data/spinnerf-dataset/10-sam

cd prior/HashNeRF-pytorch || exit
python run_nerf.py --config configs/spinnerf_10_sam.txt --finest_res 512 --log2_hashmap_size 19 --lrate 0.01 --lrate_decay 10

cd prior/torch-ngp || exit
python scripts/llff2nerf.py ../../data/spinnerf-dataset/10-sam --images images --downscale=1
python main_nerf.py ../../data/spinnerf-dataset/10-sam --workspace ../../logs/spinnerf_10_delete -O --iters 200000

python scripts/llff2nerf.py ../../data/spinnerf-dataset/10 --images images --downscale=4
python main_nerf.py ../../data/spinnerf-dataset/10 --workspace ../../logs/spinnerf_10 -O --iters 200000

# Train images before delete using nerf
CUDA_VISIBLE_DEVICES=0 python run_nerf.py --config configs/spinnerf_10.txt
CUDA_VISIBLE_DEVICES=0 python run_nerf.py --config configs/spinnerf_10_delete.txt

# Train images after delete using nerf
CUDA_VISIBLE_DEVICES=1 python run_nerf.py --config configs/statue_delete.txt
