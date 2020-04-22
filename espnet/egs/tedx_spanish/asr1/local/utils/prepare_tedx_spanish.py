from .dataset_downloader import download_and_extract_data
from .transformators import TEDxSpanish2KaldiTransformer
import os

espnet_kaldi_eg_directory = '..'
dataset_url = 'http://www.openslr.org/resources/67/tedx_spanish_corpus.tgz'
dataset_name = 'TEDxSpanish'
download_folder = '../raw_data'
subset_size = int(os.environ.get('ESPNET_SUBSET_SIZE'))

if __name__ == '__main__':

    
    dataset_location = download_and_extract_data(
        dataset_url=dataset_url,
        dataset_name=dataset_name,
        download_folder=download_folder)
    print("Dataset location:", dataset_location)

    print("Init data transformer")
    transformer = TEDxSpanish2KaldiTransformer()
    if subset_size:
        transformer.transform(
            raw_data_path=dataset_location,
            espnet_kaldi_eg_directory=espnet_kaldi_eg_directory,
            subset_size=subset_size)
    else:
        transformer.transform(
            raw_data_path=dataset_location,
            espnet_kaldi_eg_directory=espnet_kaldi_eg_directory)