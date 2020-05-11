import os
from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm
import pandas as pd

from base_transformer import AbstractDataTransformer


class CrowdsourcedOpenASR(AbstractDataTransformer):

    def transform(self, raw_data_path, espnet_kaldi_eg_directory, subset_size=None, *args, **kwargs):

        kaldi_data_dir = os.path.join(espnet_kaldi_eg_directory, 'data')
        kaldi_audio_files_dir = os.path.join(espnet_kaldi_eg_directory, 'downloads')

        subdirs = ["decompressed_1", "decompressed_2"]

        audio_dirs = [os.path.join(raw_data_path, subdir) for subdir in subdirs]
        self.copy_audio_files_to_kaldi_dir(audio_dirs, kaldi_audio_files_dir)

        dfs = [pd.read_csv(Path(audio_dir, 'line_index.tsv'), delimiter='\t', header=None) for audio_dir in audio_dirs]
        data = pd.concat(dfs, axis=0)
        data.columns = ['path', 'transcript']
        dataset_size = data.shape[0]

        print("Total dataset size", dataset_size)

        if subset_size:
            print("Subset size:", subset_size)
            if dataset_size < subset_size:
                print(
                    f"ATTENTION! Provided subset size ({subset_size}) is less "
                    f"than overall dataset size ({dataset_size}). "
                    f"Taking all dataset")
            self.SUBSET_SIZE = subset_size
            data = data[:subset_size]

        print("Generating train and test files")

        wavscp, text, utt2spk = self.generate_arrays(data)

        wavscp_train, wavscp_test, text_train, text_test, utt2spk_train, utt2spk_test = \
            self.split_train_test(wavscp,
                                  text,
                                  utt2spk,
                                  test_proportion=0.2)

        self.create_files(wavscp_train, text_train, utt2spk_train, os.path.join(kaldi_data_dir, 'train'))
        self.create_files(wavscp_test, text_test, utt2spk_test, os.path.join(kaldi_data_dir, 'test'))

    def generate_arrays(self, data: pd.DataFrame):

        wavscp = list()
        text = list()
        utt2spk = list()

        data['path'] = data['path'].apply(lambda x: "downloads/" + x[:-4] + '.wav')

        for idx, row in data.iterrows():
            transcript = self.clean_text(row['sentence'])
            file_path = row['path']
            speaker_id = row['client_id']
            segment_id = idx
            utterance_id = f'{speaker_id}-{segment_id}'
            wavscp.append(f'{utterance_id} {file_path}')
            utt2spk.append(f'{utterance_id} {speaker_id}')
            text.append(f'{utterance_id} {transcript}')

        return wavscp, text, utt2spk
