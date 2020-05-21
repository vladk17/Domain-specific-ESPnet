from pathlib import Path
from dataset_utils.dataset_downloader import download_and_extract_data
from dataset_utils.transformers.spanish_tedx import TEDxSpanish2KaldiTransformer
import logging

logger = logging.root
logger.addHandler(logging.StreamHandler)
logger.setLevel(logging.INFO)

dataset_urls = ['http://www.openslr.org/resources/67/tedx_spanish_corpus.tgz']
dataset_name = 'spanish_tedx'
eg_dir = Path('/espnet/egs/spanish_tedx/asr1')
raw_data_folder = Path(eg_dir, 'raw_data')

if __name__ == '__main__':
    dataset_location = download_and_extract_data(
        dataset_urls=dataset_urls,
        dataset_name=dataset_name,
        download_folder=raw_data_folder)

    logger.info("Dataset location:", dataset_location)

    transformer = TEDxSpanish2KaldiTransformer()
    transformer.transform(
        raw_data_path=dataset_location,
        espnet_kaldi_eg_directory=eg_dir)
