from .dataset_downloader import download_and_extract_data
from .common_voice_spanish import CommonVoiceKaldiTransformer
import os

espnet_kaldi_eg_directory = '..'
dataset_url = 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/es.tar.gz'
dataset_name = 'CommonVoiceSpanish'
download_folder = '../raw_data'
subset_size = int(os.environ.get('ESPNET_SUBSET_SIZE'))

if __name__ == '__main__':

    dataset_location = download_and_extract_data(
        dataset_url=dataset_url,
        dataset_name=dataset_name,
        download_folder=download_folder)
    print("Dataset location:", dataset_location)

    print("Init data transformer")
    transformer = CommonVoiceKaldiTransformer()
    if subset_size:
        transformer.transform(
            raw_data_path=dataset_location,
            espnet_kaldi_eg_directory=espnet_kaldi_eg_directory,
            subset_size=subset_size)
    else:
        transformer.transform(
            raw_data_path=dataset_location,
            espnet_kaldi_eg_directory=espnet_kaldi_eg_directory)
