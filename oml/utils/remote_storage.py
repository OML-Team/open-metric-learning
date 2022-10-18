import json
from pathlib import Path
from typing import Optional

import requests
from tqdm.auto import tqdm

from oml.const import REQUESTS_TIMEOUT, STORAGE_URL
from oml.utils.io import download_file_from_url


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


__all__ = ["download_folder_from_remote_storage", "download_file_from_remote_storage"]
