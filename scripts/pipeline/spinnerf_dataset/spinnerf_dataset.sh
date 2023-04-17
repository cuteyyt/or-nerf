#SCENE = [2, 3, 4, 7, ${SCENE}, 12, book, trash]
export SCENE=2

# Copy img files from ori folder to sparse folder (colmap sparse reconstruction)
# Use your target resolution
cd data
mkdir spinnerf_dataset_sparse
cd spinnerf_dataset_sparse
mkdir ${SCENE}
cd ${SCENE}
cd ../../
#cp -r spinnerf_dataset/${SCENE}/images_4/. spinnerf_dataset_sparse/${SCENE}/images_4
# May do this manually by choosing the 60 images

# COLMAP sparse/dense reconstruction
# Refer to scripts/colmap.sh

# Here we use sparse reconstruction
# spinnerf_dataset has done this already
# Run LLFF if necessary
cp -r spinnerf_dataset/${SCENE}/sparse/. spinnerf_dataset_sparse/${SCENE}/sparse
cd ../

# SAM predict
python prior/segment-anything/post_sam.py \
  --in_dir data \
  --out_dir data \
  --dataset_name spinnerf_dataset \
  --scene_name ${SCENE} \
  --ckpt_path ckpts/sam/sam_vit_h_4b8939.pth \
  --points_json_path prior/segment-anything/points_prompt.json

# Lama
# dataset preparation
python prior/lama/pre_lama.py \
  --in_dir data/ \
  --out_dir data/ \
  --dataset_name spinnerf_dataset \
  --scene_name ${SCENE} \
  --mask_refine_json prior/lama/mask_refine.json

# inpainting
cd prior/lama
TORCH_HOME=$(pwd)
PYTHONPATH=$(pwd)
export TORCH_HOME && export PYTHONPATH

# without refine
python3 bin/predict.py \
  model.path="$(pwd)"/../../ckpts/lama/big-lama \
  indir="$(pwd)"/../../data/spinnerf_dataset_sam/${SCENE}/lama \
  outdir="$(pwd)"/../../data/spinnerf_dataset_sam/${SCENE}/lama_out

python3 bin/predict.py \
  model.path="$(pwd)"/../../ckpts/lama/big-lama \
  indir="$(pwd)"/../../data/spinnerf_dataset_sam/${SCENE}/lama \
  outdir="$(pwd)"/../../data/spinnerf_dataset_sam/${SCENE}/lama_out_refine \
  refine=True

cd ../../

# pre nerf input
# Pay attention if we need to transform pose
python prior/nerf-pytorch/pre_nerf.py \
  --in_dir data/ \
  --out_dir data/ \
  --dataset_name spinnerf_dataset \
  --scene_name ${SCENE} \
  --pose_transform_json prior/nerf-pytorch/pose_transform.json
cd ../

#cd prior/nerf-pytorch
## Train images before delete using nerf
#python run_nerf.py --config configs/spinnerf_dataset/${SCENE}.txt
#
## Train images after delete using nerf
#python run_nerf.py --config configs/spinnerf_dataset/${SCENE}_delete.txt
