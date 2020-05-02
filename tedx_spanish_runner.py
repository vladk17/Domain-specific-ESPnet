from dataset_downloader import download_and_extract_data
from tedx_spanish import TEDxSpanish2KaldiTransformer

if __name__ == '__main__':
    dataset_location = download_and_extract_data(
        dataset_url='http://www.openslr.org/resources/67/tedx_spanish_corpus.tgz',
        dataset_name='TEDxSpanish', download_folder='')
    print("Dataset location:", dataset_location)

    transformer = TEDxSpanish2KaldiTransformer()
    transformer.transform(
        raw_data_path=dataset_location,
        espnet_kaldi_eg_directory='/home/stanislav/y-data/gong/Domain-specific-ESPnet/espnet/egs/tedx_spanish/asr1')
