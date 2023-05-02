# SPIn-NeRF: Multiview Segmentation and Perceptual Inpainting with Neural Radiance Fields

[**Project**](https://spinnerf3d.github.io/) | [**Paper**](https://arxiv.org/abs/2211.12254) | [**YouTube**](https://youtu.be/WEgJf1WC5SQ) | [**Dataset**](https://drive.google.com/drive/folders/1N7D4-6IutYD40v9lfXGSVbWrd47UdJEC)

Pytorch implementation of SPIn-NeRF. SPIn-NeRF leverages 2D priors from image inpainters, and enables view-consistent inpainting of NeRFs.

---

## Quick Start

### Dependencies

After installing [Pytorch](https://pytorch.org/get-started/locally/) according to your CUDA version, install the rest of the dependencies:
```
pip install -r requirements.txt
```
Also, install LaMa dependencies with the following:
```
pip install -r lama/requirements.txt
```

You will also need [COLMAP](https://github.com/colmap/colmap) installed to compute poses if you want to run on your data.


### Dataset preparation

Download the zip files of the dataset from [here](https://drive.google.com/drive/folders/1N7D4-6IutYD40v9lfXGSVbWrd47UdJEC?usp=share_link). Extract them under `/data`. 
Here, we provide information for running the statue scene. For other scenes, a similar approach with potentially different `factor` can be done. 

Extract `statue.zip` under `/data`. This can be done with `unzip statue.zip -d data`. You might need to install unzip with `sudo apt-get install unzip`. 

If you want to use your own data, make sure that you put your data in a folder under `data` with the following format (Note that labels under `statue/images_2/label` are `1` where we need inpainting, and `0` otherwise):
```
statue
├── images
│   ├── IMG_2707.jpg
│   ├── IMG_2708.jpg
│   ├── ...
│   └── IMG_2736.jpg
└── images_2
    ├── IMG_2707.png
    ├── IMG_2708.png
    ├── ...
    ├── IMG_2736.png
    └── label
        ├── IMG_2707.png
        ├── IMG_2708.png
        ├── ...
        └── IMG_2736.png

```
where in this example, we want to use `--factor 2` for the images to use 2x downsized images for the fitting, thus we have put 2x downsized images under `images_2`. If your original images are larger, put the original images under `images`, and the Nx downsized images under `images_N`, where N is chosen based on your GPU availabitlity. Also, make sure to obtain camera parameters using COLMAP. This can be done with the following command:
```
python imgs2poses.py <your_datadir>
```
For example, for the sample `statue` dataset, the camera parameters can be obtained as `python imgs2poses.py data/statue`. Note that for this specific dataset, we have already provided the camera parameters and you can skip running COLMAP. 

### Running an initial NeRF for getting the depths

First, use the following command to render disparities from the training views. This can be done with the following: 

```
rm -r LaMa_test_images/*
rm -r output/label/*
python DS_NeRF/run_nerf.py --config DS_NeRF/configs/config.txt --render_factor 1 --prepare --i_weight 1000000000 --i_video 1000000000 --i_feat 4000 --N_iters 4001 --expname statue --datadir ./data/statue --factor 2 --N_gt 0
```
After this, rendered disparities (inverse depths) are ready at `lama/LaMa_test_images`, with their corresponding labels at `lama/LaMa_test_images/label`. 

### Running LaMa to generate geometry and appearance guidance

First, let's run LaMa to generate depth priors:
```
cd lama
```
Now, make sure to follow the [LaMa](https://github.com/saic-mdal/lama) instructions for downloading the big-lama model.  
```
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
python bin/predict.py refine=True model.path=$(pwd)/big-lama indir=$(pwd)/LaMa_test_images outdir=$(pwd)/output
```
Now, the inpainted disparities are ready at `lama/output/label`. Copy the images and put the under `data/statue/images_2/depth`. It can be done with the following:
```
dataset=statue
factor=2

rm -r ../data/$dataset/images_$factor/depth
mkdir ../data/$dataset/images_$factor/depth
cp ./output/label/*.png ../data/$dataset/images_$factor/depth
```

Now, let's generate the inpainted RGB images:

```
dataset=statue
factor=2

rm -r LaMa_test_images/*
rm -r output/label/*
cp ../data/$dataset/images_$factor/*.png LaMa_test_images
mkdir LaMa_test_images/label
cp ../data/$dataset/images_$factor/label/*.png LaMa_test_images/label
python bin/predict.py refine=True model.path=$(pwd)/big-lama indir=$(pwd)/LaMa_test_images outdir=$(pwd)/output
rm -r ../data/$dataset/images_$factor/lama_images
mkdir ../data/$dataset/images_$factor/lama_images
cp ../data/$dataset/images_$factor/*.png ../data/$dataset/images_$factor/lama_images
cp ./output/label/*.png ../data/$dataset/images_$factor/lama_images
```
The inpainted RGB images are now ready under `lama/output/label`, and have been copied to `data/statue/images_2/lama_images`. 
```
statue
├── colmap_depth.npy
├── images
│   ├── IMG_2707.jpg
│   ├── ...
│   └── IMG_2736.jpg
├── images_2
│   ├── depth
│   │   ├── img000.png
│   │   ├── ...
│   │   └── img028.png
│   ├── IMG_2707.png
│   ├── IMG_2708.png
│   ├── ...
│   ├── IMG_2736.png
│   ├── label
│   │   ├── IMG_2707.png
│   │   ├── ... 
│   │   ├── IMG_2736.png
│   └── lama_images
│       ├── IMG_2707.png
│       ├── ...
│       └── IMG_2736.png
└── sparse
```

Let's move back to the main directory by `cd ..`. 

### Running multiview inpainter
Now, using the following command, the optimization of the final inpainted NeRF will be started. A video of the inpainted NeRF will be saved every `i_video` iterations. The fitting will be done for `N_iters` iterations. A sample rendering from a random view point is saved to `/test_renders` every `i_feat` iterations, which can be used for early sanity checks and hyper-parameter tunings. 
```
python DS_NeRF/run_nerf.py --config DS_NeRF/configs/config.txt --i_feat 200 --lpips --i_weight 1000000000000 --i_video 1000 --N_iters 10001 --expname statue --datadir ./data/statue --N_gt 0 --factor $factor
```

Note that our experiments were done on Nvidia A6000 GPUs. In case of running on GPUs with lower memory, you might get out-of-memory errors. To prevent that, please try increasing the arguments `--lpips_render_factor` and `--patch_len_factor`, or reducing `--lpips_batch_size`. 

#### Notes on mask dilation
Please note that as mentioned in the paper, the masks are dilated by default with a 5x5 kernel for 5 iterations to ensure that all of the object is masked, and that the effects of the shadow of the unwanted objects on the scene is reduced. If you wish to alter the dilation, first, you need to change the dilations applied by the LaMa model to generate the inpaintings under `lama/saicinpainting/evaluation/refinement.py` at the following line:
```
tmp = cv2.dilate(tmp.cpu().numpy().astype('uint8'), np.ones((5, 5), np.uint8), iterations=5)
```
Then, you also need to change the LLFF loader to load the masks with proper dilations applied to them under `DS_NeRF/load_llff.py`. In this file, the following line is responsible for the dilations:
```
msk = cv2.dilate(msk, np.ones((5, 5), np.uint8), iterations=5)
```


# BibTeX
If you find SPIn-NeRF useful in your work, please consider citing it:
```
@inproceedings{spinnerf,
      title={{SPIn-NeRF}: Multiview Segmentation and Perceptual Inpainting with Neural Radiance Fields}, 
      author={Ashkan Mirzaei and Tristan Aumentado-Armstrong and Konstantinos G. Derpanis and Jonathan Kelly and Marcus A. Brubaker and Igor Gilitschenski and Alex Levinshtein},
      year={2023},
      booktitle={CVPR},
}
```
