from pathlib import Path
from dataset_utils.dataset_downloader import download_and_extract_data
from dataset_utils.transformers.spanish_common_voice import CommonVoiceKaldiTransformer

dataset_url = 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/es.tar.gz'
dataset_name = 'CommonVoiceSpanish'
eg_dir = Path('/espnet/egs/spanish_common_voice/asr1')
raw_data_folder = Path(eg_dir, 'raw_data')

if __name__ == '__main__':
    dataset_location = download_and_extract_data(
        dataset_urls=[dataset_url],
        dataset_name=dataset_name,
        download_folder=raw_data_folder)

    print("Dataset location:", dataset_location)

    transformer = CommonVoiceKaldiTransformer()
    transformer.transform(
        raw_data_path=dataset_location,
        espnet_kaldi_eg_directory=eg_dir)
