# OR-NeRF: Object Removing from 3D Scenes Guided by Multiview Segmentation with Neural Radiance Fields

[**Project**]() | [**Paper**](https://arxiv.org/abs/2305.10503)

Pytorch Implementation of OR-NeRF. OR-NeRF removes objects from 3D scenes with points or text prompts on only one image.

---

**TODO**

1. - [ ] Project Page
2. - [x] All training and test code for the paper.

## Start up

### Environment
Conda is recommended.
```shell
conda create -n ornerf python=3.9
pip install -r requirements.txt
```
Code is tested on RTX3090 or A5000 with CUDA version>11. Torchs compatible with your CUDA version should be working.

### Data
We run our methods on 20 scenes from the following datasets:
1.
2.

## Reproduce
**Note**: You can refer to **scripts** folder for running all experiment settings mentioned in the paper.

### Mask Generation
```shell
# Step 1. Use Points or Text Prompts for 
```

### Scene Object Removal

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