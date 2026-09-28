# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 18:13:27 2021


Module for miscellaneous utility functions

@author: Bruno Caneco
"""

import shutil
import os
from pathlib import Path

import requests
from tqdm import tqdm

REPO_DIR = Path(__file__).resolve().parent.parent

# Marker file identifying a directory as SSCD output, which may be emptied on
# the next run. Log files from runs made before the marker existed are also
# accepted as proof, so existing output directories keep working.
OUTPUT_MARKER = ".sscd_output"
LEGACY_OUTPUT_FILES = ("log_sscd_detection.log", "log_sscd_evaluation.log")


# ------------------------------------------------------------------------------
def _strtobool(val):
    """Convert a string representation of truth to True or False."""
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError(f"Invalid truth value {val!r}")


# ------------------------------------------------------------------------------
def boolean_string(s):
    """dealing with args_parser issues when taking boolean variables as inputs"""
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


# ------------------------------------------------------------------------------
def clean_output_dir(dir_path):
    if os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path)
        except OSError as e:
            print("Error: %s : %s" % (dir_path, e.strerror))


# ------------------------------------------------------------------------------
def prepare_output_dir(output_dir, input_paths=()):
    """
    Create an empty output directory for a run, deleting the results of a
    previous SSCD run in it if present.

    Refuses (ValueError) rather than deleting anything that is not SSCD output:
    - output_dir equal to, or a parent of, any of `input_paths`, the repository,
      its data/ folder, the current working directory or the home directory
    - a non-empty directory without the SSCD output marker (or a log file from
      an earlier SSCD run)
    """
    out = Path(output_dir).resolve()

    guarded = [Path(p) for p in input_paths] + [REPO_DIR, REPO_DIR / "data", Path.cwd(), Path.home()]
    for path in guarded:
        path = path.resolve()
        if out == path or out in path.parents:
            raise ValueError(
                f"Output directory '{out}' is, or contains, '{path}'. The output directory is "
                "emptied at the start of every run - choose a separate, dedicated directory."
            )
    if out.parent == out:
        raise ValueError(f"Output directory '{out}' is a filesystem root.")

    if out.exists():
        if not out.is_dir():
            raise ValueError(f"Output path '{out}' exists and is not a directory.")
        if any(out.iterdir()):
            is_sscd_output = (out / OUTPUT_MARKER).exists() or any(
                (out / name).exists() for name in LEGACY_OUTPUT_FILES
            )
            if not is_sscd_output:
                raise ValueError(
                    f"Output directory '{out}' is not empty and does not contain previous SSCD "
                    "output. Refusing to delete its contents - choose a new or empty directory."
                )
            shutil.rmtree(out)

    out.mkdir(parents=True, exist_ok=True)
    (out / OUTPUT_MARKER).write_text(
        "This directory holds SSCD output and is emptied at the start of each run.\n"
    )


# ------------------------------------------------------------------------------
def unpack_for_string(s, sep="\n\t"):
    """
    Little utility function to unpack list elements for use in logging messages
    """
    return sep.join(str(x) for x in s)


# ------------------------------------------------------------------------------
def query_yes_no(question, default="no"):
    """
    hacked from https://gist.github.com/garrettdreyfus/8153571
    """
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError(f"Unknown setting '{default}' for default.")

    while True:
        try:
            resp = input(question + prompt).strip().lower()
            if default is not None and resp == "":
                return default == "yes"
            else:
                return _strtobool(resp)
        except ValueError:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


# ------------------------------------------------------------------------------
def download_url(url, save_filepath, chunk_size=8192):
    """
    Download a file from a URL to a local path, with a progress bar.

    Parameters
    ----------
    url : str
        URL of the file to download.
    save_filepath : str
        Local path where the downloaded file will be saved.
    chunk_size : int, optional
        Size in bytes of each streamed chunk. Default is 8192.

    Raises
    ------
    ValueError
        If the server returns an HTML response instead of binary data,
        which typically indicates an error page or redirect rather than
        the expected file.
    requests.HTTPError
        If the server returns a non-2xx HTTP status code.
    """
    # timeout (seconds) applies to connecting and to each read, not the whole download
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()

    content_type = r.headers.get("content-type", "")
    if "text/html" in content_type:
        raise ValueError(
            f"Expected a binary file but got an HTML response from {url!r}. "
            "The URL may be incorrect, require authentication, or be a "
            "folder/preview link rather than a direct download link."
        )

    file_size_bytes = int(r.headers.get("content-length", 0))

    pbar = tqdm(
        total=file_size_bytes,
        position=0,
        leave=True,
        ascii=True,
        desc="Downloading",
        unit="iB",
        unit_scale=True,
    )

    with open(save_filepath, "wb") as file:
        for chunk in r.iter_content(chunk_size=chunk_size):
            pbar.update(len(chunk))
            file.write(chunk)

    pbar.close()
