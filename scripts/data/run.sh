SCRIPT=$1
set -e

# ibrnet_data
sh "$SCRIPT" ibrnet_data qq3 data
sh "$SCRIPT" ibrnet_data qq6 data
sh "$SCRIPT" ibrnet_data qq10 data
sh "$SCRIPT" ibrnet_data qq11 data
sh "$SCRIPT" ibrnet_data qq13 data
sh "$SCRIPT" ibrnet_data qq16 data
sh "$SCRIPT" ibrnet_data qq17 data
sh "$SCRIPT" ibrnet_data qq21 data

# llff_real_iconic
sh "$SCRIPT" llff_real_iconic data5_piano data

# nerf_llff_data
sh "$SCRIPT" nerf_llff_data room data
sh "$SCRIPT" nerf_llff_data horns data
sh "$SCRIPT" nerf_llff_data fortress data

# spinnerf_dataset
sh "$SCRIPT" spinnerf_dataset 2 data
sh "$SCRIPT" spinnerf_dataset 3 data
sh "$SCRIPT" spinnerf_dataset 4 data
sh "$SCRIPT" spinnerf_dataset 7 data
sh "$SCRIPT" spinnerf_dataset 10 data
sh "$SCRIPT" spinnerf_dataset 12 data
sh "$SCRIPT" spinnerf_dataset book data
sh "$SCRIPT" spinnerf_dataset trash data
