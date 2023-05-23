# nerf + lama prior directly
python datasets/test.py --pred_dir logs/test/nerf/delete/2/render_all_200000 --target_dir data/test/spinnerf_dataset_gt/2/images_4 --mode render
python datasets/test.py --pred_dir logs/test/nerf/delete/3/render_all_200000 --target_dir data/test/spinnerf_dataset_gt/3/images_4 --mode render
python datasets/test.py --pred_dir logs/test/nerf/delete/4/render_all_200000 --target_dir data/test/spinnerf_dataset_gt/4/images_4 --mode render
python datasets/test.py --pred_dir logs/test/nerf/delete/7/render_all_200000 --target_dir data/test/spinnerf_dataset_gt/7/images_4 --mode render
python datasets/test.py --pred_dir logs/test/nerf/delete/10/render_all_200000 --target_dir data/test/spinnerf_dataset_gt/10/images_4 --mode render
python datasets/test.py --pred_dir logs/test/nerf/delete/12/render_all_200000 --target_dir data/test/spinnerf_dataset_gt/12/images_4 --mode render
python datasets/test.py --pred_dir logs/test/nerf/delete/book/render_all_200000 --target_dir data/test/spinnerf_dataset_gt/book/images_4 --mode render
python datasets/test.py --pred_dir logs/test/nerf/delete/trash/render_all_200000 --target_dir data/test/spinnerf_dataset_gt/trash/images_4 --mode render

# nerf + da
python datasets/test.py --pred_dir logs/test/nerf/depth_all/2/render_all_200000 --target_dir data/test/spinnerf_dataset_gt/2/images_4 --mode render
python datasets/test.py --pred_dir logs/test/nerf/depth_all/3/render_all_200000 --target_dir data/test/spinnerf_dataset_gt/3/images_4 --mode render
python datasets/test.py --pred_dir logs/test/nerf/depth_all/4/render_all_200000 --target_dir data/test/spinnerf_dataset_gt/4/images_4 --mode render
python datasets/test.py --pred_dir logs/test/nerf/depth_all/7/render_all_200000 --target_dir data/test/spinnerf_dataset_gt/7/images_4 --mode render
python datasets/test.py --pred_dir logs/test/nerf/depth_all/10/render_all_200000 --target_dir data/test/spinnerf_dataset_gt/10/images_4 --mode render
python datasets/test.py --pred_dir logs/test/nerf/depth_all/12/render_all_200000 --target_dir data/test/spinnerf_dataset_gt/12/images_4 --mode render
python datasets/test.py --pred_dir logs/test/nerf/depth_all/book/render_all_200000 --target_dir data/test/spinnerf_dataset_gt/book/images_4 --mode render
python datasets/test.py --pred_dir logs/test/nerf/depth_all/trash/render_all_200000 --target_dir data/test/spinnerf_dataset_gt/trash/images_4 --mode render

# nerf + dp
python datasets/test.py --pred_dir logs/test/nerf/depth_partial/2/render_all_200000 --target_dir data/test/spinnerf_dataset_gt/2/images_4 --mode render
python datasets/test.py --pred_dir logs/test/nerf/depth_partial/3/render_all_200000 --target_dir data/test/spinnerf_dataset_gt/3/images_4 --mode render
python datasets/test.py --pred_dir logs/test/nerf/depth_partial/4/render_all_200000 --target_dir data/test/spinnerf_dataset_gt/4/images_4 --mode render
python datasets/test.py --pred_dir logs/test/nerf/depth_partial/7/render_all_200000 --target_dir data/test/spinnerf_dataset_gt/7/images_4 --mode render
python datasets/test.py --pred_dir logs/test/nerf/depth_partial/10/render_all_200000 --target_dir data/test/spinnerf_dataset_gt/10/images_4 --mode render
python datasets/test.py --pred_dir logs/test/nerf/depth_partial/12/render_all_200000 --target_dir data/test/spinnerf_dataset_gt/12/images_4 --mode render
python datasets/test.py --pred_dir logs/test/nerf/depth_partial/book/render_all_200000 --target_dir data/test/spinnerf_dataset_gt/book/images_4 --mode render
python datasets/test.py --pred_dir logs/test/nerf/depth_partial/trash/render_all_200000 --target_dir data/test/spinnerf_dataset_gt/trash/images_4 --mode render

# nerf + lpips
python datasets/test.py --pred_dir logs/test/nerf/lpips/2/render_all_200000 --target_dir data/test/spinnerf_dataset_gt/2/images_4 --mode render
python datasets/test.py --pred_dir logs/test/nerf/lpips/3/render_all_200000 --target_dir data/test/spinnerf_dataset_gt/3/images_4 --mode render
python datasets/test.py --pred_dir logs/test/nerf/lpips/4/render_all_200000 --target_dir data/test/spinnerf_dataset_gt/4/images_4 --mode render
python datasets/test.py --pred_dir logs/test/nerf/lpips/7/render_all_200000 --target_dir data/test/spinnerf_dataset_gt/7/images_4 --mode render
python datasets/test.py --pred_dir logs/test/nerf/lpips/10/render_all_200000 --target_dir data/test/spinnerf_dataset_gt/10/images_4 --mode render
python datasets/test.py --pred_dir logs/test/nerf/lpips/12/render_all_200000 --target_dir data/test/spinnerf_dataset_gt/12/images_4 --mode render
python datasets/test.py --pred_dir logs/test/nerf/lpips/book/render_all_200000 --target_dir data/test/spinnerf_dataset_gt/book/images_4 --mode render
python datasets/test.py --pred_dir logs/test/nerf/lpips/trash/render_all_200000 --target_dir data/test/spinnerf_dataset_gt/trash/images_4 --mode render

# spinnerf ori
python datasets/test.py --pred_dir logs/test/spinnerf/delete/2/render_gt --target_dir data/test/spinnerf_dataset_gt/2/images_4 --mode render
python datasets/test.py --pred_dir logs/test/spinnerf/delete/3/render_gt --target_dir data/test/spinnerf_dataset_gt/3/images_4 --mode render
python datasets/test.py --pred_dir logs/test/spinnerf/delete/4/render_gt --target_dir data/test/spinnerf_dataset_gt/4/images_4 --mode render
python datasets/test.py --pred_dir logs/test/spinnerf/delete/7/render_gt --target_dir data/test/spinnerf_dataset_gt/7/images_4 --mode render
python datasets/test.py --pred_dir logs/test/spinnerf/delete/10/render_gt --target_dir data/test/spinnerf_dataset_gt/10/images_4 --mode render
python datasets/test.py --pred_dir logs/test/spinnerf/delete/12/render_gt --target_dir data/test/spinnerf_dataset_gt/12/images_4 --mode render
python datasets/test.py --pred_dir logs/test/spinnerf/delete/book/render_gt --target_dir data/test/spinnerf_dataset_gt/book/images_4 --mode render
python datasets/test.py --pred_dir logs/test/spinnerf/delete/trash/render_gt --target_dir data/test/spinnerf_dataset_gt/trash/images_4 --mode render

# spinnerf no lpips
python datasets/test.py --pred_dir logs/test/spinnerf/ablation/no_lpips/2/render_gt --target_dir data/test/spinnerf_dataset_gt/2/images_4 --mode render
python datasets/test.py --pred_dir logs/test/spinnerf/ablation/no_lpips/3/render_gt --target_dir data/test/spinnerf_dataset_gt/3/images_4 --mode render
python datasets/test.py --pred_dir logs/test/spinnerf/ablation/no_lpips/4/render_gt --target_dir data/test/spinnerf_dataset_gt/4/images_4 --mode render
python datasets/test.py --pred_dir logs/test/spinnerf/ablation/no_lpips/7/render_gt --target_dir data/test/spinnerf_dataset_gt/7/images_4 --mode render
python datasets/test.py --pred_dir logs/test/spinnerf/ablation/no_lpips/10/render_gt --target_dir data/test/spinnerf_dataset_gt/10/images_4 --mode render
python datasets/test.py --pred_dir logs/test/spinnerf/ablation/no_lpips/12/render_gt --target_dir data/test/spinnerf_dataset_gt/12/images_4 --mode render
python datasets/test.py --pred_dir logs/test/spinnerf/ablation/no_lpips/book/render_gt --target_dir data/test/spinnerf_dataset_gt/book/images_4 --mode render
python datasets/test.py --pred_dir logs/test/spinnerf/ablation/no_lpips/trash/render_gt --target_dir data/test/spinnerf_dataset_gt/trash/images_4 --mode render

# spinnerf no depth
python datasets/test.py --pred_dir logs/test/spinnerf/ablation/no_depth/2/render_gt --target_dir data/test/spinnerf_dataset_gt/2/images_4 --mode render
python datasets/test.py --pred_dir logs/test/spinnerf/ablation/no_depth/3/render_gt --target_dir data/test/spinnerf_dataset_gt/3/images_4 --mode render
python datasets/test.py --pred_dir logs/test/spinnerf/ablation/no_depth/4/render_gt --target_dir data/test/spinnerf_dataset_gt/4/images_4 --mode render
python datasets/test.py --pred_dir logs/test/spinnerf/ablation/no_depth/7/render_gt --target_dir data/test/spinnerf_dataset_gt/7/images_4 --mode render
python datasets/test.py --pred_dir logs/test/spinnerf/ablation/no_depth/10/render_gt --target_dir data/test/spinnerf_dataset_gt/10/images_4 --mode render
python datasets/test.py --pred_dir logs/test/spinnerf/ablation/no_depth/12/render_gt --target_dir data/test/spinnerf_dataset_gt/12/images_4 --mode render
python datasets/test.py --pred_dir logs/test/spinnerf/ablation/no_depth/book/render_gt --target_dir data/test/spinnerf_dataset_gt/book/images_4 --mode render
python datasets/test.py --pred_dir logs/test/spinnerf/ablation/no_depth/trash/render_gt --target_dir data/test/spinnerf_dataset_gt/trash/images_4 --mode render

# tensorf + directly
python datasets/test.py --pred_dir logs/test/tensorf/delete/2/imgs_gt --target_dir data/test/spinnerf_dataset_gt/2/images_4 --mode render
python datasets/test.py --pred_dir logs/test/tensorf/delete/3/imgs_gt --target_dir data/test/spinnerf_dataset_gt/3/images_4 --mode render
python datasets/test.py --pred_dir logs/test/tensorf/delete/4/imgs_gt --target_dir data/test/spinnerf_dataset_gt/4/images_4 --mode render
python datasets/test.py --pred_dir logs/test/tensorf/delete/7/imgs_gt --target_dir data/test/spinnerf_dataset_gt/7/images_4 --mode render
python datasets/test.py --pred_dir logs/test/tensorf/delete/10/imgs_gt --target_dir data/test/spinnerf_dataset_gt/10/images_4 --mode render
python datasets/test.py --pred_dir logs/test/tensorf/delete/12/imgs_gt --target_dir data/test/spinnerf_dataset_gt/12/images_4 --mode render
python datasets/test.py --pred_dir logs/test/tensorf/delete/book/imgs_gt --target_dir data/test/spinnerf_dataset_gt/book/images_4 --mode render
python datasets/test.py --pred_dir logs/test/tensorf/delete/trash/imgs_gt --target_dir data/test/spinnerf_dataset_gt/trash/images_4 --mode render

# tensorf + depth
python datasets/test.py --pred_dir logs/test/tensorf/depth_all/2/imgs_gt --target_dir data/test/spinnerf_dataset_gt/2/images_4 --mode render
python datasets/test.py --pred_dir logs/test/tensorf/depth_all/3/imgs_gt --target_dir data/test/spinnerf_dataset_gt/3/images_4 --mode render
python datasets/test.py --pred_dir logs/test/tensorf/depth_all/4/imgs_gt --target_dir data/test/spinnerf_dataset_gt/4/images_4 --mode render
python datasets/test.py --pred_dir logs/test/tensorf/depth_all/7/imgs_gt --target_dir data/test/spinnerf_dataset_gt/7/images_4 --mode render
python datasets/test.py --pred_dir logs/test/tensorf/depth_all/10/imgs_gt --target_dir data/test/spinnerf_dataset_gt/10/images_4 --mode render
python datasets/test.py --pred_dir logs/test/tensorf/depth_all/12/imgs_gt --target_dir data/test/spinnerf_dataset_gt/12/images_4 --mode render
python datasets/test.py --pred_dir logs/test/tensorf/depth_all/book/imgs_gt --target_dir data/test/spinnerf_dataset_gt/book/images_4 --mode render
python datasets/test.py --pred_dir logs/test/tensorf/depth_all/trash/imgs_gt --target_dir data/test/spinnerf_dataset_gt/trash/images_4 --mode render

# tensorf + lpips
python datasets/test.py --pred_dir logs/test/tensorf/lpips/2/imgs_gt --target_dir data/test/spinnerf_dataset_gt/2/images_4 --mode render
python datasets/test.py --pred_dir logs/test/tensorf/lpips/3/imgs_gt --target_dir data/test/spinnerf_dataset_gt/3/images_4 --mode render
python datasets/test.py --pred_dir logs/test/tensorf/lpips/4/imgs_gt --target_dir data/test/spinnerf_dataset_gt/4/images_4 --mode render
python datasets/test.py --pred_dir logs/test/tensorf/lpips/7/imgs_gt --target_dir data/test/spinnerf_dataset_gt/7/images_4 --mode render
python datasets/test.py --pred_dir logs/test/tensorf/lpips/10/imgs_gt --target_dir data/test/spinnerf_dataset_gt/10/images_4 --mode render
python datasets/test.py --pred_dir logs/test/tensorf/lpips/12/imgs_gt --target_dir data/test/spinnerf_dataset_gt/12/images_4 --mode render
python datasets/test.py --pred_dir logs/test/tensorf/lpips/book/imgs_gt --target_dir data/test/spinnerf_dataset_gt/book/images_4 --mode render
python datasets/test.py --pred_dir logs/test/tensorf/lpips/trash/imgs_gt --target_dir data/test/spinnerf_dataset_gt/trash/images_4 --mode render
