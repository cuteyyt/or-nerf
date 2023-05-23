# OR-NeRF: Object Removing from 3D Scenes Guided by Multiview Segmentation with Neural Radiance Fields

[**Project Page**]() | [**Paper**](https://arxiv.org/abs/2305.10503) | [**Supplementary Materials**]()

Pytorch Implementation of OR-NeRF. OR-NeRF removes objects from 3D scenes with points or text prompts on only one image. To realise, OR-NeRF first uses [**SAM**](https://github.com/facebookresearch/segment-anything) to predict multiview masks and [**LaMa**](https://github.com/advimman/lama) to inpaint the unwanted region. Then, scene with objects deleted can be reconstructed from inpainting priors with Neural Radiance Fields.

<center class="half">
<video width="320" height="240" controls preload="auto" poster="assets/imgs/ori.png">
    <source src="assets/imgs/ori.mp4" type="video/mp4">
</video>
<video width="320" height="240" controls preload="auto" poster="assets/imgs/delete.png">
    <source src="assets/imgs/delete.mp4" type="video/mp4">
</video>
</center>

---

## Start up

### Environment
Conda is recommended.

```shell
conda create -n ornerf python=3.9
pip install -r requirements.txt
```

Code has been tested on CUDA version 11 or higher with RTX3090 or A5000. Torch versions that are compatible with your CUDA version should work.

### Data

We run our methods on 20 scenes from the following datasets:

1. [**IBRNet data:**](https://github.com/googleinterns/IBRNet) We use 8 scenes from this dataset: qq3, qq6, qq10, qq11 and qq13 from *ibrnet_collected\ibrnet_collected_1*, and qq16, qq17 and qq21 from *ibrnet_collected/ibrnet_collected_2*.
2. [**SPIn-NeRF data:**](https://github.com/SamsungLabs/SPIn-NeRF) We use all scenes from *spinnerf-dataset* for multiview segmentation and 8 scenes (excluding 1 and 9) for scene object removal. Note their statue scene is actually qq11 in IBRNet data.
3. [**NeRF LLFF data:**](https://github.com/bmild/nerf) We use 3 scenes from this dataset: room, horns, fortress from *nerf_llff_data*.
4. [**LLFF Real data:**](https://github.com/Fyusion/LLFF) We use only one scene from this dataset: data5_piano from *real_iconic*.

### File Structure

Our file structure is: (You can change it as you wish)

```shell
OR-NeRF
├── assets
├── ckpts # checkpoints for pre-trained models
├── comparison # Code for scene object removal
├── configs # Configurations
├── data # Data folder, our code will create several '_suffix' for different usage
│   ├── spinnerf_dataset # Original download data
│   ├── spinnerf_dataset_depth # Containing inpainted rgb and depth for loss add-ons 
│   ├── spinnerf_dataset_sam # Containing inpainted rgb only for training NeRF directly
│   ├── spinnerf_dataset_sam_text # Text prompt sam, while no '_text' is gen from points
│   ├── spinnerf_dataset_sparse # For training NeRF withou removal directly
│   ├── spinnerf_dataset_spinnerf # For training SPIn-NeRF pipeline
│   └── test # For quantative test, file structure under this is the same as 'data'
├── datasets # Code for process data 
├── logs # Logs folder
│   ├── nerf # Logs for Ours-NeRF
│   │   ├── dir # Reconstruct removal scenes from inpainted priors directly
│   │   ├── da # Train with all depth supervision 
│   │   ├── dp # Train with partial depth supervision 
│   │   └── lpips # Train with perceptual loss and all depth supervision
│   ├── spinnerf # Logs for SPIn-NeRF, subfolders are similar to NeRF
│   └── tensorf # logs for Ours-TensoRF, subfolders are similar to NeRF
├── prior # Code for running pre-trained models like SAM    
└── scripts # Scripts for running experiments
```

## Reproduce
**Note**: You can refer to **scripts** folder for running all experiment settings mentioned in the paper.

### Scene Object Removal
**Note**: This part is for run NeRF and TensoRF architecture.

```shell
# Step 1. Prepare a 'sparse' folder for training NeRF without deletion
# sh scripts/data/gen_sparse.sh [dataset_name] [scene_name] [in_dir] [out_dir]
# Refer to datasets/pre_sparse.py, prior/LLFF for more info 
sh scripts/data/gen_sparse.sh spinnerf_dataset 2 data data

# If using text prompt, first convert this prompt to points prompt ---> 'sam_text'
# sh scripts/data/gen_mask_sam_text.sh [dataset_name] [scene_name] [in_dir] [out_dir]
# Refer to datasets/run_sam_text.py, prior/Grounded-Segment-Anything for more info
sh scripts/data/gen_mask_sam_text.sh spinnerf_dataset 2 data data

# Step 2. Prepare a 'sam' folder containing multiview masks from points prompt
# sh scripts/data/gen_mask_sam_points.sh [dataset_name] [scene_name] [in_dir] [out_dir]
# Refer to datasets/run_sam_points.py, prior/segment-anything for more info 
sh scripts/data/gen_mask_sam_points.sh spinnerf_dataset 2 data data

# If using text prompt, uncomment --text_prompt in scripts/data/gen_mask_sam_points.sh and comment the other command for points prompt

# Step 3. Add inpainted RGB priors to 'sam' or 'sam_text' folder
# sh scripts/data/gen_lama_prior.sh [dataset_name] [scene_name] [in_dir] [out_dir]
# Refer to datasets/pre_lama.py, datasets/post_lama.py and prior/LaMa for more info 
sh scripts/data/gen_lama_prior.sh spinnerf_dataset 2 data data

# If using text prompt, change 'SFX=sam' to 'SFX=sam_text' in scripts/data/gen_lama_prior.sh

# Step 4. Train a Neural Radiance Fields without deletion, this should gen log dir 'ori'
# For NeRF
# sh scripts/comparison/nerf/run_nerf_ori.sh [scene_name]
# Refer to comparison/nerf-pytorch/run_nerf.py for more info
sh scripts/comparison/nerf/run_nerf_ori.sh 2
# For TensoRF
# sh scripts/comparison/tensorf/run_tensorf_ori.sh [scene_name]
# Refer to comparison/Tensorf/train.py for more info
sh scripts/comparison/tensorf/run_tensorf_ori.sh 2

# Step 5. Prepare a 'depth' folder containing inpainted rgb and depth priors
# sh scripts/data/gen_depth.sh [dataset_name] [scene_name] [in_dir] [out_dir]
# Refer to datasets/pre_depth.py for more info
# Please change LOGDIR in scripts/data/gen_depth.sh to match your own
sh scripts/data/gen_depth.sh spinnerf_dataset 2 data data

# Step 6. Reconstruct deleted scenes directly, this should gen log dir 'delete'
# For NeRF
# sh scripts/comparison/nerf/run_nerf_delete.sh [scene_name]
# Refer to comparison/nerf-pytorch/run_nerf.py for more info
sh scripts/comparison/nerf/run_nerf_delete.sh 2
# For TensoRF
# sh scripts/comparison/tensorf/run_tensorf_delete.sh [scene_name]
# Refer to comparison/Tensorf/train.py for more info
sh scripts/comparison/tensorf/run_tensorf_delete.sh 2

# Step 7. Reconstruct deleted scenes with depth supervision, 
# This should gen log dir 'depth_all' or 'depth_partial' 
# For NeRF, mode controls 'depth all' or 'dpeth partial'
# sh scripts/comparison/nerf/run_nerf_depth.sh [scene_name] [mode]
# Refer to comparison/nerf-pytorch/run_nerf_depth.py for more info
sh scripts/comparison/nerf/run_nerf_depth_all.sh 2 # Or
sh scripts/comparison/nerf/run_nerf_depth_partial.sh 2
# For TensoRF, 'depth partial' is not applicable
# sh scripts/comparison/tensorf/run_tensorf_depth.sh [scene_name]
# Refer to comparison/Tensorf/train.py for more info
sh scripts/comparison/tensorf/run_tensorf_depth.sh 2

# Step 8. Reconstruct deleted scenes with depth supervision and perceptual loss
# This should gen log dir 'lpips' 
# For NeRF
# sh scripts/comparison/nerf/run_nerf_lpips.sh [scene_name]
# Refer to comparison/nerf-pytorch/run_nerf_lpips.py for more info
sh scripts/comparison/nerf/run_nerf_lpips.sh 2
# For TensoRF
# sh scripts/comparison/tensorf/run_tensorf_lpips.sh [scene_name]
# Refer to comparison/Tensorf/train.py for more info
sh scripts/comparison/tensorf/run_tensorf_lpips.sh 2

```

### SPIn-NeRF Comparison

We adjust [**SPIn-NeRF**](https://github.com/SamsungLabs/SPIn-NeRF) with minimum change to fit our pipeline.

```shell
# Step 1. Prepare a 'sparse' folder for training NeRF without deletion
# sh scripts/data/gen_sparse.sh [dataset_name] [scene_name] [in_dir] [out_dir]
# Refer to datasets/pre_sparse.py, prior/LLFF for more info 
sh scripts/data/gen_sparse.sh spinnerf_dataset 2 data data

# Step 2. Prepare a 'spinnerf' folder for running SPIn-NeRF code
# sh scripts/data/gen_spinnerf.sh [dataset_name] [scene_name] [in_dir] [out_dir] [factor]
# Refer to datasets/pre_spinnerf.py, comparison/SPIn-NeRF for more info
sh scripts/data/gen_pre_spinnerf.py spinnerf_dataset 2 data data 4

# Step 3. Train spinnerf without modification (all depth + perceptual loss)
# This should gen log dir 'delete'
# sh scripts/comparison/spinnerf/run_spinnerf_ori.sh [scene_name]
# Refer to comparison/SPIn-NeRF/DS_NeRF/run_nerf.py for more info
sh scripts/comparison/spinnerf/run_spinnerf_ori.sh 2

# Step 4. Train spinnerf without perceptual loss (all depth supervision only)
# This should gen log dir 'ablation/no_lpips'
# sh scripts/comparison/spinnerf/run_spinnerf_no_lpips.sh [scene_name]
# Refer to comparison/SPIn-NeRF/DS_NeRF/run_nerf.py for more info
sh scripts/comparison/spinnerf/run_spinnerf_no_lpips.sh 2

# Step 5. Train spinnerf without perceptual and depth loss (train DS_NeRF directly)
# This should gen log dir 'ablation/no_depth'
# sh scripts/comparison/spinnerf/run_spinnerf_no_lpips_no_depth.sh [scene_name]
# Refer to comparison/SPIn-NeRF/DS_NeRF/run_nerf.py for more info
sh scripts/comparison/spinnerf/run_spinnerf_no_lpips_no_depth.sh 2
``` 

### Test

To run the test process, we follow a similar procedure as the reconstruction process described above. However, to ensure clarity, we create a separate folder named *data/test* specifically for running tests. Please refer to the related scripts and Python files for controlling the "test" process. If you have any questions, please feel free to raise an issue or reach out to us directly.

**Note**: You may check *scripts/comparison/run_net.sh* and *scripts/data/preprocess_data.sh* to run multiple scenes with one command line.

## Citation
If you find OR-NeRF useful in your work, please consider citing it:

```shell
@misc{
  yin2023ornerf,
  title={OR-NeRF: Object Removing from 3D Scenes Guided by Multiview Segmentation with Neural Radiance Fields}, 
  author={Youtan Yin and Zhoujie Fu and Fan Yang and Guosheng Lin},
  year={2023},
  eprint={2305.10503},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

This repository is developed on [**NeRF-Pytorch**](https://github.com/yenchenlin/nerf-pytorch), [**SPIn-NeRF**](https://github.com/SamsungLabs/SPIn-NeRF), [**TensoRF**](https://github.com/apchenstu/TensoRF), [**SAM**](https://github.com/facebookresearch/segment-anything), [**Grounded-SAM**](https://github.com/IDEA-Research/Grounded-Segment-Anything) and [**LaMa**](https://github.com/advimman/lama). Thanks for their great work and you may also consider cite them.