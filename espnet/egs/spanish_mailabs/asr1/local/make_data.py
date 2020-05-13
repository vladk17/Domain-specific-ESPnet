from pathlib import Path
from dataset_utils.dataset_downloader import download_and_extract_data
from dataset_utils.transformers.spanish_mailabs import MailabsKaldiTransformer

dataset_url = 'http://www.caito.de/data/Training/stt_tts/es_ES.tgz'
dataset_name = 'Mailabs'
eg_dir = Path('/espnet/egs/spanish_mailabs/asr1')
raw_data_folder = Path(eg_dir, 'raw_data')

if __name__ == '__main__':
    dataset_location = download_and_extract_data(
        dataset_urls=[dataset_url],
        dataset_name=dataset_name,
        download_folder=raw_data_folder)

    print("Dataset location:", dataset_location)

    transformer = MailabsKaldiTransformer()
    transformer.transform(
        raw_data_path=dataset_location,
        espnet_kaldi_eg_directory=eg_dir)
