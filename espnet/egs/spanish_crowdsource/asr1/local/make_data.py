from pathlib import Path
from dataset_utils.dataset_downloader import download_and_extract_data
from dataset_utils.transformers.spanish_crowdsource_openasr import CrowdsourcedOpenASR

EG_NAME = 'spanish_crowdsource'
dataset_urls = ['http://www.openslr.org/resources/71/es_cl_female.zip',
                'http://www.openslr.org/resources/71/es_cl_male.zip',
                'http://www.openslr.org/resources/72/es_co_female.zip',
                'http://www.openslr.org/resources/72/es_co_male.zip',
                'http://www.openslr.org/resources/73/es_pe_female.zip',
                'http://www.openslr.org/resources/73/es_pe_male.zip',
                'http://www.openslr.org/resources/74/es_pr_female.zip',
                'http://www.openslr.org/resources/75/es_ve_female.zip',
                'http://www.openslr.org/resources/75/es_ve_male.zip'
                ]
eg_dir = Path(f'/espnet/egs/{EG_NAME}/asr1')
raw_data_folder = Path(eg_dir, 'raw_data')

if __name__ == '__main__':
    dataset_location = download_and_extract_data(
        dataset_urls=dataset_urls,
        dataset_name=EG_NAME,
        download_folder=raw_data_folder)

    print("Dataset location:", dataset_location)

    transformer = CrowdsourcedOpenASR()
    transformer.transform(
        raw_data_path=dataset_location,
        espnet_kaldi_eg_directory=eg_dir)
