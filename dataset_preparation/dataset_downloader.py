import os
import tar
import urllib
import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join('..', os.path.dirname(os.path.dirname('.'))))
DATA_PREFIX = 'TEDLIUM'
ENDPOINT = 'http://www.openslr.org/resources/7/'
DATASET_NAME = f"{DATA_PREFIX}_release1.tar.gz"
DATA_DIR = os.path.join(PROJECT_ROOT,'data')
current_datasets_dir = os.path.join(DATA_DIR,DATA_PREFIX)
dataset_path = os.path.join(current_datasets_dir,DATASET_NAME)
print('Dataset path:', dataset_path)

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


if not os.path.exists(dataset_path):
    print(f"Downloading {DATASET_NAME}: {ENDPOINT+DATASET_NAME}")
    
    if not os.path.exists(current_datasets_dir):
        os.mkdir(current_datasets_dir)
    
    download_url(ENDPOINT+DATASET_NAME, dataset_path)
    
else:
    print("Tarfile already exists.")
    
if not os.path.exists(dataset_path[:-7]):
    print("Decompressing data")
    tar = tarfile.open(dataset_path)
    tar.extractall(current_datasets_dir)
else:
    print("Tarfile has been already decompressed")