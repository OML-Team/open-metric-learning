import hashlib
import json
from pathlib import Path
from typing import Optional, Union

import requests
import validators
from tqdm.auto import tqdm

from oml.const import CKPT_SAVE_ROOT, REQUESTS_TIMEOUT, STORAGE_URL


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
    files = sorted(tuple(str(fname) for fname in folder.rglob("*") if fname.is_file()))

    folder_hash = hashlib.md5()
    structure_hash = "".join(files).encode()
    folder_hash.update(structure_hash)

    for file in files:
        file_hash = calc_file_hash(file).encode()
        folder_hash.update(file_hash)
    return folder_hash.hexdigest()


def download_file_from_url(url: str, fname: Optional[str] = None, timeout: float = REQUESTS_TIMEOUT) -> Optional[bytes]:  # type: ignore
    assert validators.url(url), "Invalid URL"
    response = requests.get(url, timeout=timeout)

    if response.status_code == 200:
        if fname is not None:
            Path(fname).parent.mkdir(parents=True, exist_ok=True)
            with open(fname, "wb+") as f:
                f.write(response.content)
        else:
            return response.content
    else:
        raise RuntimeError(f"Can not download file from '{url}'")


def _fix_path_for_remote_storage(path: str) -> str:
    if path == "/":
        path = ""
    elif path.startswith("/"):
        path = path[1:]

    if path.endswith("/"):
        path = path[:-1]

    return path


def download_file_from_remote_storage(
    remote_path: str, fname: Optional[str] = None, timeout: float = REQUESTS_TIMEOUT
) -> Optional[bytes]:
    remote_path = _fix_path_for_remote_storage(remote_path)
    return download_file_from_url(f"{STORAGE_URL}/download/{remote_path}", fname, timeout=timeout)


def download_checkpoint(url_or_remote_path: str, hash_md5: str, fname: Optional[str] = None) -> str:
    """
    Args:
        url_or_remote_path: URL to the checkpoint or path to the file on our storage
        hash_md5: Value of md5sum
        fname: Name of the checkpoint after the downloading process

    Returns:
        Path to the checkpoint

    """
    CKPT_SAVE_ROOT.mkdir(exist_ok=True, parents=True)

    fname = fname if fname else Path(url_or_remote_path).name
    save_path = str(CKPT_SAVE_ROOT / fname)

    if Path(save_path).exists():
        if calc_file_hash(save_path).startswith(hash_md5):
            print("Checkpoint is already here.")
            return save_path
        else:
            print("Checkpoint is already here, but hashed don't match. " "We will remove the old checkpoint.")
            Path(save_path).unlink()

    print("Downloading checkpoint...")

    if validators.url(url_or_remote_path):
        download_file_from_url(url=url_or_remote_path, fname=save_path)

    else:  # we assume we work we file id (Google Drive)
        download_file_from_remote_storage(remote_path=url_or_remote_path, fname=save_path)

    if not calc_file_hash(save_path).startswith(hash_md5):
        raise Exception("Downloaded checkpoint is probably broken. " "Hash values don't match.")

    return str(save_path)


def download_folder_from_remote_storage(
    remote_folder: str, local_folder: str, timeout: float = REQUESTS_TIMEOUT
) -> None:
    remote_folder = _fix_path_for_remote_storage(remote_folder)

    remote_ls_response = requests.get(f"{STORAGE_URL}/ls/{remote_folder}", timeout=timeout)
    content = json.loads(remote_ls_response.content)

    if remote_ls_response.status_code == 200:
        remote_files = content["remote_files"]
    else:
        raise Exception(content["detail"])

    local_folder = Path(local_folder)
    local_folder.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(remote_files)
    for remote_file in pbar:
        fname = Path(remote_file).relative_to(remote_folder)
        pbar.set_description_str(f"Downloading '{fname}'")

        local_fname = local_folder / fname
        local_fname.parent.mkdir(exist_ok=True, parents=True)
        download_file_from_remote_storage(remote_file, str(local_fname), timeout=timeout)


__all__ = [
    "calc_file_hash",
    "calc_folder_hash",
    "download_file_from_url",
    "download_file_from_remote_storage",
    "download_checkpoint",
    "download_folder_from_remote_storage",
]
