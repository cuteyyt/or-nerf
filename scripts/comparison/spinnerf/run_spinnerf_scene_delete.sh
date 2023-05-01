# Generate ori depth
cd comparison/SPIN-NeRF
rm -rf lama/LaMa_test_images
rm -rf lama/output/label
rm -rf lama/outputs
python DS_NeRF/run_nerf.py --config DS_NeRF/configs/config.txt \
  --render_factor 1 \
  --prepare \
  --i_weight 1000000000 \
  --i_video 1000000000 \
  --i_feat 4000 \
  --N_iters 4001 \
  --expname statue_1 \
  --datadir ../../data/statue \
  --factor 1 \
  --N_gt 0 \
  --basedir ../../logs/spinnerf/delete

# Generate inpainted depth
cd lama
TORCH_HOME=$(pwd) && PYTHONPATH=$(pwd)
export TORCH_HOME && export PYTHONPATH

python bin/predict.py refine=True model.path="$(pwd)"/../../../ckpts/lama/big-lama \
  indir="$(pwd)"/LaMa_test_images \
  outdir="$(pwd)"/output

# Inpainted Training
python DS_NeRF/run_nerf.py --config DS_NeRF/configs/config.txt \
  --i_feat 200 \
  --lpips \
  --i_weight 1000000000000 \
  --i_video 1000 \
  --N_iters 10001 \
  --expname statue_1 \
  --datadir ../../data/statue \
  --N_gt 0 \
  --factor 1 \
  --basedir ../../logs/spinnerf/delete
