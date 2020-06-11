import os
import tarfile
import urllib.request
import zipfile
from typing import List
import boto3 as boto3
from tqdm import tqdm
import pathlib
import logging

logger = logging.root


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
    if not os.path.exists(download_folder):
        os.mkdir(download_folder)
    dataset_dir = os.path.join(download_folder, dataset_name)

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


def download_dir(client, resource, bucket, prefix, local_dir):
    paginator = client.get_paginator('list_objects')
    for result in paginator.paginate(Bucket=bucket, Delimiter='/', Prefix=prefix):
        if result.get('CommonPrefixes') is not None:
            for subdir in result.get('CommonPrefixes'):
                download_dir(client, resource, bucket, subdir.get('Prefix'), local_dir)
        for file in tqdm(result.get('Contents', [])[:5]):
            logger.info(f"Downloading data from {prefix}")
            dest_pathname = os.path.join(local_dir, file.get('Key'))
            if not os.path.exists(os.path.dirname(dest_pathname)):
                os.makedirs(os.path.dirname(dest_pathname))
            if file.get('Size') != 0:
                resource.meta.client.download_file(bucket, file.get('Key'), dest_pathname)


def download_from_s3(key, bucket, dataset_name, download_folder):
    access_key = os.getenv("AWS_GONG_ACCESS_KEY")
    secret_token = os.getenv("AWS_GONG_SECRET")
    client = boto3.client('s3',
                          aws_access_key_id=access_key,
                          aws_secret_access_key=secret_token)
    resource = boto3.resource('s3', aws_access_key_id=access_key,
                              aws_secret_access_key=secret_token)
    dataset_dir = os.path.join(download_folder, dataset_name)
    if os.path.exists(dataset_dir):
        logger.info("Dataset has been already downloaded")
    else:
        logger.info("Getting data from s3")
        download_dir(client, resource, bucket, key, dataset_dir)

    return pathlib.Path(dataset_dir).absolute()
