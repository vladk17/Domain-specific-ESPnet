import os
from dataset_utils.dataset_downloader import download_and_extract_data_from_kaggle_datasets
from dataset_utils.transformers.spanish_kaggle_120h import Kaggle120hSpanish2KaldiTransformer

if __name__ == '__main__':

    common_folder_prefix = '/media/vk/volume1/Domain-specific-ESPnet'
    dataset_location = download_and_extract_data_from_kaggle_datasets(
        kuggle_dataset_name=r'carlfm01/120h-spanish-speech',
        kuggle_archive_name=r'120h-spanish-speech.zip',
        dataset_name='kaggle_120h_spanish_speech',
        download_folder=os.path.join(common_folder_prefix,"raw_data"))
    print("Dataset location:", dataset_location)

    transformer = Kaggle120hSpanish2KaldiTransformer()
    transformer.transform(
       raw_data_path=dataset_location,
       espnet_kaldi_eg_directory=os.path.join(common_folder_prefix, 'espnet/egs/spanish_kaggle_120h/asr1'))
