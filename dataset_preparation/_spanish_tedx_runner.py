from dataset_downloader import download_and_extract_data
from transformers.spanish_tedx import TEDxSpanish2KaldiTransformer

if __name__ == '__main__':
    dataset_location = download_and_extract_data(
        dataset_url='http://www.openslr.org/resources/67/tedx_spanish_corpus.tgz',
        dataset_name='TEDxSpanish', download_folder="/home/stanislav/y-data/gong/Domain-specific-ESPnet/data")
    print("Dataset location:", dataset_location)

    transformer = TEDxSpanish2KaldiTransformer()
    transformer.transform(
        raw_data_path=dataset_location,
        espnet_kaldi_eg_directory='/home/stanislav/y-data/gong/Domain-specific-ESPnet/espnet/egs/spanish_tedx/asr1')
