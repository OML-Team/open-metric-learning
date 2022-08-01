import hashlib
from pathlib import Path
from typing import Optional

import gdown
import requests
import validators

# TODO: valid only on Linux and Mac
CKPT_SAVE_ROOT = Path("/tmp/torch/checkpoints/")


def calc_file_hash(fname: str) -> str:
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_checkpoint(url_or_fid: str, hash_md5: str, fname: Optional[str] = None) -> str:
    """
    Args:
        url_or_fid: URL to the checkpoint or file id on Google Drive
        hash_md5: Value of md5sum
        fname: Name of the checkpoint after the downloading process

    Returns:
        Path to the checkpoint

    """
    CKPT_SAVE_ROOT.mkdir(exist_ok=True, parents=True)

    fname = fname if fname else Path(url_or_fid).name
    save_path = str(CKPT_SAVE_ROOT / fname)

    if Path(save_path).exists():
        if calc_file_hash(save_path).startswith(hash_md5):
            print("Checkpoint is already here.")
            return save_path
        else:
            print("Checkpoint is already here, but hashed don't match. " "We will remove the old checkpoint.")
            Path(save_path).unlink()

    print("Downloading checkpoint...")

    if validators.url(url_or_fid):
        ckpt = requests.get(url_or_fid)

        if ckpt.status_code == 200:
            with open(save_path, "wb+") as f:
                f.write(ckpt.content)
            print(f"Checkpoint was saved to {str(save_path)}")
        else:
            raise Exception(f"Cannot download checkpoint from {save_path}")

    else:  # we assume we work we file id (Google Drive)
        gdown.download(id=url_or_fid, output=save_path, quiet=False)

    if not calc_file_hash(save_path).startswith(hash_md5):
        raise Exception("Downloaded checkpoint is probably broken. " "Hash values don't match.")

    return str(save_path)
