from dataset_downloader import download_and_extract_data
from transformers.spanish_mailabs import MailabsKaldiTransformer

if __name__ == '__main__':
    dataset_location = download_and_extract_data(
        dataset_url='http://www.caito.de/data/Training/stt_tts/es_ES.tgz',
        dataset_name='Mailabs', download_folder="/home/stanislav/y-data/industry-project/Domain-specific-ESPnet/data")

    print("Dataset location:", dataset_location)

    transformer = MailabsKaldiTransformer()
    transformer.transform(
        raw_data_path=dataset_location,
        espnet_kaldi_eg_directory='/home/stanislav/y-data/industry-project/Domain-specific-ESPnet/'
                                  'espnet_emulation/egs/mailabs/asr1')