from dataset_utils.dataset_downloader import download_and_extract_data
from dataset_utils.transformers.spanish_crowdsource_openasr import CrowdsourcedOpenASR

if __name__ == '__main__':
    dataset_location = download_and_extract_data(
        dataset_urls=['http://www.openslr.org/resources/71/es_cl_female.zip',
                      'http://www.openslr.org/resources/71/es_cl_male.zip'
                      ],
        dataset_name='crowdsource_chilean',
        download_folder="/home/stanislav/y-data/industry-project/Domain-specific-ESPnet/data",
        force_decompress=False)

    print("Dataset location:", dataset_location)

    transformer = CrowdsourcedOpenASR()
    transformer.transform(
        raw_data_path=dataset_location,
        espnet_kaldi_eg_directory='/home/stanislav/y-data/industry-project/Domain-specific-ESPnet/'
                                  'espnet_emulation/egs/crowdsource_chilean/asr1')
