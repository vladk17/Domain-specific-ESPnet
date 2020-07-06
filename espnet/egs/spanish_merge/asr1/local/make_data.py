import logging
import os
from pathlib import Path
from typing import List

from dataset_utils.dataset_downloader import download_and_extract_data, download_from_s3, \
    download_and_extract_data_from_kaggle_datasets
from dataset_utils.transformers.spanish_gong import GongSpanish2KaldiTransformer
from dataset_utils.transformers.spanish_gong_unsupervised import GongUnsupervisedSpanish2KaldiTransformer
from dataset_utils.transformers.spanish_kaggle_120h import Kaggle120hSpanish2KaldiTransformer
from dataset_utils.transformers.spanish_mailabs import MailabsKaldiTransformer
from dataset_utils.transformers.spanish_common_voice import CommonVoiceKaldiTransformer
from dataset_utils.transformers.spanish_tedx import TEDxSpanish2KaldiTransformer
from dataset_utils.transformers.spanish_crowdsource_openasr import CrowdsourcedOpenASR
from collections import namedtuple

DataSet = namedtuple('DataSet', ['name', 'urls', 'transformer_class'])

logger = logging.root
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

datasets = [
    ('Mailabs', ['http://www.caito.de/data/Training/stt_tts/es_ES.tgz'], MailabsKaldiTransformer()),
    ('CommonVoiceSpanish', [
        'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/es.tar.gz'],
     CommonVoiceKaldiTransformer()),
    ('TEDxSpanish', ['http://www.openslr.org/resources/67/tedx_spanish_corpus.tgz'],
     TEDxSpanish2KaldiTransformer()),
    ('Crowdsource', ['http://www.openslr.org/resources/71/es_cl_female.zip',
                     'http://www.openslr.org/resources/71/es_cl_male.zip',
                     'http://www.openslr.org/resources/72/es_co_female.zip',
                     'http://www.openslr.org/resources/72/es_co_male.zip',
                     'http://www.openslr.org/resources/73/es_pe_female.zip',
                     'http://www.openslr.org/resources/73/es_pe_male.zip',
                     'http://www.openslr.org/resources/74/es_pr_female.zip',
                     'http://www.openslr.org/resources/75/es_ve_female.zip',
                     'http://www.openslr.org/resources/75/es_ve_male.zip'

                     ], CrowdsourcedOpenASR()),
    # ('kaggle_120h', None, Kaggle120hSpanish2KaldiTransformer())
]
datasets = [DataSet(_[0], _[1], _[2]) for _ in datasets]
eg_dir = Path('/espnet/egs/spanish_merge/asr1')
raw_data_folder = Path(eg_dir, 'raw_data')


def prepare_public_data_factory(datasets: List[DataSet]):
    for dataset in datasets:
        logger.info(f"\n\nDownloading and extracting data for '{dataset.name}' dataset\n\n")

        if dataset.name == 'kaggle_120h':
            dataset_location = download_and_extract_data_from_kaggle_datasets(
                kuggle_dataset_name=r'carlfm01/120h-spanish-speech',
                kuggle_archive_name=r'120h-spanish-speech.zip',
                dataset_name=dataset.name,
                download_folder=raw_data_folder)
        else:
            dataset_location = download_and_extract_data(
                dataset_urls=dataset.urls,
                dataset_name=dataset.name,
                download_folder=raw_data_folder)

        logger.info(f"Dataset location: {dataset_location}")
        logger.info(f"Using class {dataset.transformer_class}")

        transformer = dataset.transformer_class
        transformer.transform(
            raw_data_path=dataset_location,
            espnet_kaldi_eg_directory=eg_dir)


def prepare_gong_data():
    logger.info(f"\n\nDownloading and extracting data for 'Gongio' datasets\n\n")
    dataset_location = download_from_s3(key='to-y-data',
                                        bucket='gong-shared-with-y-data',
                                        dataset_name='spanish_gong',
                                        download_folder=raw_data_folder)
    logger.info(f"Dataset location: {dataset_location}")

    transformers = [GongSpanish2KaldiTransformer(), GongUnsupervisedSpanish2KaldiTransformer()]
    for transformer in transformers:
        logger.info(f"Using class {transformer}")
        transformer.transform(
            raw_data_path=dataset_location,
            espnet_kaldi_eg_directory=eg_dir)


if __name__ == '__main__':
    # prepare_public_data_factory(datasets)
    prepare_gong_data()
