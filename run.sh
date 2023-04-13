# SAM predict and lama dataset preparation
cd prior/segment-angthing || exit
python preprocess.py

# Lama inpainting
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
