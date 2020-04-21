from .dataset_downloader import download_and_extract_data
from .transformators import TEDxSpanish2KaldiTransformer

espnet_kaldi_eg_directory = '/home/stanislav/y-data/gong/Domain-specific-ESPnet/espnet/egs/tedx_spanish/asr1'
dataset_url = 'http://www.openslr.org/resources/67/tedx_spanish_corpus.tgz'
dataset_name = 'TEDxSpanish'
download_folder = '../temp_downloads'

if __name__ == '__main__':

    dataset_location = download_and_extract_data(
        dataset_url=dataset_url,
        dataset_name=dataset_name,
        download_folder=download_folder)
    print("Dataset location:", dataset_location)

    transformer = TEDxSpanish2KaldiTransformer()
    transformer.transform(
        raw_data_path=dataset_location,
        espnet_kaldi_eg_directory=espnet_kaldi_eg_directory,
        train_test_size=2000)
