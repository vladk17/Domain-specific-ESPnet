from pathlib import Path
from typing import List

from dataset_utils.dataset_downloader import download_and_extract_data
from dataset_utils.transformers.spanish_mailabs import MailabsKaldiTransformer
from dataset_utils.transformers.spanish_common_voice import CommonVoiceKaldiTransformer
from dataset_utils.transformers.spanish_tedx import TEDxSpanish2KaldiTransformer
from dataset_utils.transformers.spanish_crowdsource_openasr import CrowdsourcedOpenASR
from collections import namedtuple

DataSet = namedtuple('DataSet', ['name', 'urls', 'transformer_class'])

datasets = [('Mailabs', ['http://www.caito.de/data/Training/stt_tts/es_ES.tgz'], MailabsKaldiTransformer),
            ('CommonVoiceSpanish', [
                'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/es.tar.gz'],
             CommonVoiceKaldiTransformer),
            ('TEDxSpanish', ['http://www.openslr.org/resources/67/tedx_spanish_corpus.tgz'],
             TEDxSpanish2KaldiTransformer),
            ('Crowdsource', ['http://www.openslr.org/resources/71/es_cl_female.zip',
                             'http://www.openslr.org/resources/71/es_cl_male.zip'
                             ], CrowdsourcedOpenASR)]
datasets = [DataSet(_[0], _[1], _[2]) for _ in datasets]
eg_dir = Path('/espnet/egs/spanish_merge/asr1')
raw_data_folder = Path(eg_dir, 'raw_data')


def run_factory(datasets: List[DataSet]):
    for dataset in datasets:

        print(f"\nDownloading and extracting data for '{dataset.name}' dataset\n")

        dataset_location = download_and_extract_data(
            dataset_urls=dataset.urls,
            dataset_name=dataset.name,
            download_folder=raw_data_folder)

        print("Dataset location:", dataset_location)
        print(f"Using class {dataset.transformer_class}")

        transformer = dataset.transformer_class
        transformer.transform(
            raw_data_path=dataset_location,
            espnet_kaldi_eg_directory=eg_dir)


if __name__ == '__main__':
    run_factory(datasets)
