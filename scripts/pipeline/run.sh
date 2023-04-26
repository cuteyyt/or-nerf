# nerf_llff_data
sh scripts/pipeline/gen_lama_prior.sh nerf_llff_data room data
sh scripts/pipeline/gen_lama_prior.sh nerf_llff_data horns data
sh scripts/pipeline/gen_lama_prior.sh nerf_llff_data fortress data

# llff_real_iconic
sh scripts/pipeline/gen_lama_prior.sh llff_real_iconic data5_piano data

# spinnerf_dataset
# TODO: cam params have redundancy
sh scripts/pipeline/gen_lama_prior.sh spinnerf_dataset 2 data
sh scripts/pipeline/gen_lama_prior.sh spinnerf_dataset 3 data
sh scripts/pipeline/gen_lama_prior.sh spinnerf_dataset 4 data
sh scripts/pipeline/gen_lama_prior.sh spinnerf_dataset 7 data
sh scripts/pipeline/gen_lama_prior.sh spinnerf_dataset 10 data
sh scripts/pipeline/gen_lama_prior.sh spinnerf_dataset 12 data
sh scripts/pipeline/gen_lama_prior.sh spinnerf_dataset book data
sh scripts/pipeline/gen_lama_prior.sh spinnerf_dataset trash data
