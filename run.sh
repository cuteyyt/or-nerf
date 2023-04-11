# SAM predict and lama dataset preparation
cd prior/segment-angthing || exit
python preprocess.py

# Lama inpainting
cd prior/lama || exit
TORCH_HOME=$(pwd)
PYTHONPATH=$(pwd)
export TORCH_HOME && export PYTHONPATH

python3 bin/predict.py \
  model.path="$(pwd)"/../../ckpts/lama/big-lama \
  indir="$(pwd)"/../../data/statue-sam/lama \
  outdir="$(pwd)"/../../data/statue-sam/lama_refine_out \
  refine=True

# without refine
python3 bin/predict.py \
  model.path="$(pwd)"/../../ckpts/lama/big-lama \
  indir="$(pwd)"/../../data/statue-sam/lama \
  outdir="$(pwd)"/../../data/statue-sam/lama_out

# pre nerf input to avoid pose mismatch
cd prior/nerf || exit
python preprocess.py

# Train images before delete using nerf
CUDA_VISIBLE_DEVICES=0 python run_nerf.py --config configs/statue.txt

# Train images after delete using nerf
# TODO: code for copy cam params
CUDA_VISIBLE_DEVICES=1 python run_nerf.py --config configs/statue_delete.txt
