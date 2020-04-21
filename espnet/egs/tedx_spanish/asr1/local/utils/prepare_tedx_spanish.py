from .dataset_downloader import download_and_extract_data
from .transformators import TEDxSpanish2KaldiTransformer

espnet_kaldi_eg_directory = '..'
dataset_url = 'http://www.openslr.org/resources/67/tedx_spanish_corpus.tgz'
dataset_name = 'TEDxSpanish'
download_folder = '../raw_data'
subset = 2000

if __name__ == '__main__':

    
    dataset_location = download_and_extract_data(
        dataset_url=dataset_url,
        dataset_name=dataset_name,
        download_folder=download_folder)
    print("Dataset location:", dataset_location)

    print("Init data transformer")
    transformer = TEDxSpanish2KaldiTransformer()
    transformer.transform(
        raw_data_path=dataset_location,
        espnet_kaldi_eg_directory=espnet_kaldi_eg_directory,
        train_test_size=subset)
