"""
Download Llama-2 weights for NERRE experiments.
"""
import os
import requests
import tarfile
import hashlib

from tqdm import tqdm

src_url = "https://figshare.com/ndownloader/files/43044994"
dest_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lora_weights")
dest_file = dest_dir + ".tar.gz"
block_size = 1048576  # one mebibyte
md5_expected = "ec5dd3e51a8c176905775849410445dc"


def download(url: str, fname: str, chunk_size=1024):
    """
    Thank you to this user on github: yanqd0
    Via this gist: https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51
    """
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
            desc=fname,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


if __name__ == "__main__":

    if os.path.exists(dest_file):
        print(f"Destintation file {dest_file} exists, skipping download...")
    else:
        print(f"Downloading NERRE LoRA weights to {dest_dir}")
        download(src_url, dest_file)
        with open(dest_file, "rb") as f:
            md5_returned = hashlib.md5(f.read()).hexdigest()
            print(f"MD5Sum was {md5_returned}")

            if md5_returned != md5_expected:
                print("Warning: MD5 checksum of downloaded file not equal to expected MD5Sum.")

    print(f"Weights downloaded, extracting to {dest_dir}...")

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    tar = tarfile.open(dest_file, "r:gz")
    tar.extractall(dest_dir)

    print(f"Weights extracted to {dest_dir}...")
