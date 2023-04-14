# SAM segment
cd prior/segment-angthing || exit
python scripts/amg.py --checkpoint ./ckpts/sam_vit_h_4b8939.pth --model-type vit_h --input <image_or_folder> --output <path/to/output>