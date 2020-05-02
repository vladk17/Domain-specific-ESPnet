from .dataset_downloader import download_and_extract_data
from .transformators import TEDxSpanish2KaldiTransformer

if __name__ == '__main__':
    dataset_location = download_and_extract_data(
        dataset_url='https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/es.tar.gz',
        dataset_name='CommonVoiceSpanish',
        download_folder="data")
    print("Dataset location:", dataset_location)

    # transformer = TEDxSpanish2KaldiTransformer()
    # transformer.transform(
    #     raw_data_path=dataset_location,
    #     espnet_kaldi_eg_directory='/home/stanislav/y-data/gong/Domain-specific-ESPnet/espnet/egs/tedx_spanish/asr1')

