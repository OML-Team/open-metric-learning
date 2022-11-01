import hashlib
from pathlib import Path
from typing import List, Optional, Union

import gdown
import requests
import validators

from oml.const import CKPT_SAVE_ROOT, REQUESTS_TIMEOUT


def calc_file_hash(fname: Union[Path, str]) -> str:
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def calc_folder_hash(folder: Union[Path, str]) -> str:
    """
    The function calculates the hash of the folder iterating over files sorted by their names.
    The function also pays attention to filenames, not only content.
    """

    folder = Path(folder)
    assert folder.exists() and folder.is_dir()
    files = sorted(tuple(str(fname.relative_to(folder).as_posix()) for fname in folder.rglob("*") if fname.is_file()))

    folder_hash = hashlib.md5()
    structure_hash = "".join(files).encode()
    folder_hash.update(structure_hash)

    for file in files:
        file_hash = calc_file_hash(folder / file).encode()
        folder_hash.update(file_hash)
    return folder_hash.hexdigest()


def check_exists_and_validate_md5(path: Union[str, Path], md5: str) -> bool:
    path = Path(path)

    if not path.exists():
        return False

    if path.is_dir():
        return calc_folder_hash(path) == md5
    else:
        return calc_file_hash(path) == md5


def download_file_from_url(url: str, fname: Optional[str] = None, timeout: float = REQUESTS_TIMEOUT) -> Optional[bytes]:
    assert validators.url(url), "Invalid URL"
    response = requests.get(url, timeout=timeout)

    if response.status_code == 200:
        if fname is not None:
            Path(fname).parent.mkdir(parents=True, exist_ok=True)
            with open(fname, "wb+") as f:
                f.write(response.content)
            return None
        else:
            return response.content
    else:
        raise RuntimeError(f"Can not download file from '{url}'")


def download_checkpoint_one_of(
    url_or_fid_list: Union[List[str], str], hash_md5: str, fname: Optional[str] = None
) -> str:
    """
    The function iteratively tries to download a checkpoint from the list of resources and stops at the first
    one available for download.
    """
    if not isinstance(url_or_fid_list, (tuple, list)):
        url_or_fid_list = [url_or_fid_list]

    attempt = 0
    for url_or_fid in url_or_fid_list:
        attempt += 1
        print(url_or_fid)
        try:
            return download_checkpoint(url_or_fid, hash_md5, fname)
        except Exception:
            if attempt == len(url_or_fid_list):
                raise

    return None  # type: ignore


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
        download_file_from_url(url=url_or_fid, fname=save_path)
    else:  # we assume we work with file id (Google Drive)
        gdown.download(id=url_or_fid, output=save_path, quiet=False)

    if not calc_file_hash(save_path).startswith(hash_md5):
        raise Exception("Downloaded checkpoint is probably broken. " "Hash values don't match.")

    return str(save_path)


__all__ = [
    "calc_file_hash",
    "calc_folder_hash",
    "check_exists_and_validate_md5",
    "download_file_from_url",
    "download_checkpoint",
    "download_checkpoint_one_of",
]
