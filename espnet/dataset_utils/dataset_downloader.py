import os
import tarfile
import urllib.request
import zipfile
from typing import List

from tqdm import tqdm
import pathlib


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_and_extract_data(dataset_urls: List[str], dataset_name: str, download_folder: str):
    data_dir = download_folder

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    dataset_dir = os.path.join(data_dir, dataset_name)

    if not os.path.exists(os.path.join(dataset_dir, 'decompressed')):
        for idx, dataset_url in enumerate(dataset_urls):
            dataset_path = os.path.join(dataset_dir, dataset_url.split('/')[-1])
            print('Dataset path:', dataset_path)

            if not os.path.exists(dataset_path):
                print(f"Downloading {dataset_url}")
                if not os.path.exists(dataset_dir):
                    os.mkdir(dataset_dir)
                download_url(dataset_url, dataset_path)

            else:
                print("Archive already exists.")

            if len(dataset_urls) == 1:
                directory_name = os.path.join(dataset_dir, 'decompressed')
            else:
                directory_name = os.path.join(dataset_dir, 'decompressed', f'decompressed_{idx + 1}')
            print("Decompressing data")
            if dataset_path.endswith('zip'):
                with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
                    zip_ref.extractall(directory_name)
            else:
                with tarfile.open(dataset_path) as tar_ref:
                    tar_ref.extractall(directory_name)
    else:
        print("Archive has been already decompressed")

    final_path = os.path.join(dataset_dir, 'decompressed')
    return pathlib.Path(final_path).absolute()
