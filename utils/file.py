import os
import shutil
from pathlib import Path

from tqdm import tqdm


def copy_files(in_dir, out_dir, file_type=None):
    print(f'Copy files from {in_dir} to {out_dir}')
    in_paths = [os.path.join(in_dir, path) for path in os.listdir(in_dir) if path.lower().endswith(file_type)]
    # in_paths = sorted(in_paths)

    for in_path in tqdm(in_paths):
        filename = Path(in_path).name
        out_path = os.path.join(out_dir, filename)

        shutil.copy(in_path, out_path)
