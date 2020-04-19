from dataset_preparation.dataset_downloader import download_and_extract_data
from dataset_preparation.transformators import TEDxSpanish2KaldiTransformer

if __name__ == '__main__':
    dataset_location = download_and_extract_data(
        dataset_url='http://www.openslr.org/resources/67/tedx_spanish_corpus.tgz',
        dataset_name='TEDxSpanish')
    print("Dataset location:", dataset_location)

    transformer = TEDxSpanish2KaldiTransformer()
    transformer.transform(
        raw_data_path=dataset_location,
        kaldi_data_dir='/home/stanislav/y-data/gong/Domain-specific-ESPnet/kaldi_eg/data',
        kaldi_audio_files_dir='/home/stanislav/y-data/gong/Domain-specific-ESPnet/kaldi_eg/downloads')
