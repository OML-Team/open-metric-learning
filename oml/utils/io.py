import hashlib
from pathlib import Path

import requests

# TODO: valid only on Linux and Mac
CKPT_SAVE_ROOT = Path("/tmp/torch/checkpoints/")


def calc_file_hash(fname: str) -> str:
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_checkpoint(url: str, hash_md5: str) -> str:
    CKPT_SAVE_ROOT.mkdir(exist_ok=True, parents=True)
    save_path = str(CKPT_SAVE_ROOT / Path(url).name)

    if Path(save_path).exists():
        if calc_file_hash(save_path).startswith(hash_md5):
            print("Checkpoint is already here.")
            return save_path
        else:
            print("Checkpoint is already here, but hashed don't match. " "We will remove the old checkpoint.")
            Path(save_path).unlink()

    print("Downloading checkpoint...")
    ckpt = requests.get(url)

    if ckpt.status_code == 200:
        with open(save_path, "wb+") as f:
            f.write(ckpt.content)
        print(f"File was saved to {str(save_path)}")
    else:
        raise Exception(f"Cannot download file from {save_path}")

    if not calc_file_hash(save_path).startswith(hash_md5):
        raise Exception("Downloaded checkpoint is probably broken. " "Hash values don't match.")

    return str(save_path)
