import logging
import os
from pathlib import Path
import pandas as pd
from dataset_utils.base_transformer import AbstractDataTransformer

logger = logging.root
SUBSET_SIZE = os.environ.get("ESPNET_SUBSET_SIZE", None)


class CrowdsourcedOpenASR(AbstractDataTransformer):

    def __init__(self):
        super().__init__()
        self._prefix = 'crowdsource'
        if SUBSET_SIZE:
            self.SUBSET_SIZE = int(SUBSET_SIZE)

    def transform(self, raw_data_path, espnet_kaldi_eg_directory, *args, **kwargs):

        self.kaldi_data_dir = os.path.join(espnet_kaldi_eg_directory, 'data')
        kaldi_audio_files_dir = os.path.join(espnet_kaldi_eg_directory, 'downloads')
        logger.info(raw_data_path)
        subdirs = list(os.walk(raw_data_path))[0][1]

        audio_dirs = [os.path.join(raw_data_path, subdir) for subdir in subdirs]
        self.copy_audio_files_to_kaldi_dir(audio_dirs, kaldi_audio_files_dir)

        dfs = [pd.read_csv(Path(audio_dir, 'line_index.tsv'), delimiter='\t', header=None) for audio_dir in audio_dirs]
        data = pd.concat(dfs, axis=0)
        data.columns = ['path', 'transcript']
        dataset_size = data.shape[0]

        logger.info(f"Total dataset size {dataset_size}")

        if self.SUBSET_SIZE:
            logger.info(f"Subset size: {self.SUBSET_SIZE}")
            if dataset_size < self.SUBSET_SIZE:
                logger.info(
                    f"ATTENTION! Provided self.SUBSET_SIZE size ({self.SUBSET_SIZE}) is less "
                    f"than overall dataset size ({dataset_size}). "
                    f"Taking all dataset")
            self.SUBSET_SIZE = self.SUBSET_SIZE
            data = data[:self.SUBSET_SIZE]

        logger.info("Generating train and test files")

        wavscp, text, utt2spk = self.generate_arrays(data)

        wavscp_train, wavscp_test, text_train, text_test, utt2spk_train, utt2spk_test = \
            self.split_train_test(wavscp,
                                  text,
                                  utt2spk,
                                  test_proportion=0.2)

        self.create_files(wavscp_train, text_train, utt2spk_train, 'train')
        self.create_files(wavscp_test, text_test, utt2spk_test, 'test')

    def generate_arrays(self, data: pd.DataFrame):

        wavscp = list()
        text = list()
        utt2spk = list()

        data['path'] = data['path'].apply(lambda x: "downloads/" + x + '.wav')

        for idx, row in data.iterrows():
            transcript = self.clean_text(row['transcript'])
            file_path = row['path']

            utt_id = idx+1
            speaker_id = f"{self.prefix}sp{utt_id}"
            utterance_id = f'{speaker_id}-{self.prefix}{utt_id}'
            wavscp.append(f'{utterance_id} {file_path}')
            utt2spk.append(f'{utterance_id} {speaker_id}')
            text.append(f'{utterance_id} {transcript}')

        return wavscp, text, utt2spk
