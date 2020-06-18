import logging
import os

from dataset_utils.dataset_downloader import download_from_s3
from dataset_utils.transformers.spanish_gong import GongSpanish2KaldiTransformer
from dataset_utils.transformers.spanish_gong_unsupervised import GongUnsupervisedSpanish2KaldiTransformer
from settings import PROJECT_ROOT

logger = logging.root
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

if __name__ == '__main__':
    dataset_location = download_from_s3(key='to-y-data', bucket='gong-shared-with-y-data',
                                        dataset_name='spanish_gong',
                                        download_folder=os.path.join(PROJECT_ROOT, 'espnet', 'egs', 'spanish_gong',
                                                                     'asr1', 'raw_data'))
    print("Dataset location:", dataset_location)

    transformers = [GongSpanish2KaldiTransformer(), GongUnsupervisedSpanish2KaldiTransformer()]
    for transformer in transformers:
        transformer.transform(
            raw_data_path=dataset_location,
            espnet_kaldi_eg_directory=os.path.join(PROJECT_ROOT, 'espnet', 'egs', 'spanish_gong', 'asr1'))
