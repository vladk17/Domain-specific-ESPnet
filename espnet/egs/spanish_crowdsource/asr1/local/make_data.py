from pathlib import Path
from dataset_utils.dataset_downloader import download_and_extract_data
from dataset_utils.transformers.spanish_crowdsource_openasr import CrowdsourcedOpenASR

dataset_urls = ['http://www.openslr.org/resources/71/es_cl_female.zip',
                'http://www.openslr.org/resources/71/es_cl_male.zip'
                ]
dataset_name = 'crowdsource_chilean'
eg_dir = Path('/espnet/egs/spanish_crowdsource/asr1')
raw_data_folder = Path(eg_dir, 'raw_data')

if __name__ == '__main__':
    dataset_location = download_and_extract_data(
        dataset_urls=dataset_urls,
        dataset_name=dataset_name,
        download_folder=raw_data_folder,
        force_decompress=True)

    print("Dataset location:", dataset_location)

    transformer = CrowdsourcedOpenASR()
    transformer.transform(
        raw_data_path=dataset_location,
        espnet_kaldi_eg_directory=eg_dir)
