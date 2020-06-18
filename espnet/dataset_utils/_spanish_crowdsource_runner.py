from dataset_utils.dataset_downloader import download_and_extract_data
from dataset_utils.transformers.spanish_crowdsource_openasr import CrowdsourcedOpenASR

if __name__ == '__main__':
    dataset_location = download_and_extract_data(
        dataset_urls=['http://www.openslr.org/resources/73/es_pe_male.zip',
                      'http://www.openslr.org/resources/74/es_pr_female.zip',
                      'http://www.openslr.org/resources/75/es_ve_female.zip'
                      ],
        dataset_name='Crowdsource',
        download_folder=r"C:\Workspace\y-data\industry-project\Domain-specific-ESPnet\espnet\egs\spanish_crowdsource\asr1\raw_data")

    print("Dataset location:", dataset_location)

    transformer = CrowdsourcedOpenASR()
    transformer.transform(
        raw_data_path=dataset_location,
        espnet_kaldi_eg_directory=r'C:\Workspace\y-data\industry-project\Domain-specific-ESPnet\espnet\egs\spanish_crowdsource\asr1')
