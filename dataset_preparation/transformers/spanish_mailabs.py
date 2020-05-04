import os
from distutils.dir_util import copy_tree
from pathlib import Path
from typing import List

from pydub import AudioSegment
from tqdm import tqdm
import pandas as pd
from base_transformer import AbstractDataTransformer


class MailabsKaldiTransformer(AbstractDataTransformer):

    def transform(self, raw_data_path, espnet_kaldi_eg_directory, subset_size=None, *args, **kwargs):

        kaldi_data_dir = os.path.join(espnet_kaldi_eg_directory, 'data')
        kaldi_audio_files_dir = os.path.join(espnet_kaldi_eg_directory, 'downloads')
        root_dir = os.path.join(raw_data_path, 'es_ES', 'by_book')

        audio_dirs, transcript_paths, speakers = self.get_data_dirs(root_dir)

        dfs = list()
        for idx, transcript_path in enumerate(transcript_paths):
            df = pd.read_csv(transcript_path, delimiter="|", error_bad_lines=False, engine='python',
                             warn_bad_lines=False)
            df.columns = ['path','transcript_1','transcript_2']
            df['speaker'] = speakers[idx]
            dfs.append(df)

        data = pd.concat(dfs)
        dataset_size = data.shape[0]
        print("Total dataset size", dataset_size)

        # self.copy_audio_files_to_kaldi_dir(origin_paths=audio_dirs, destination_path=kaldi_audio_files_dir)

        if subset_size:
            print("Subset size:", subset_size)
            if dataset_size < subset_size:
                print(
                    f"ATTENTION! Provided subset size ({subset_size}) is less "
                    f"than overall dataset size ({dataset_size}). "
                    f"Taking all dataset")
            self.SUBSET_SIZE = subset_size
            data = data[:subset_size]

        print("Generating data")
        wavscp, text, utt2spk = self.generate_arrays(data)
        self.create_files(wavscp, text, utt2spk, os.path.join(kaldi_data_dir, 'train_test'))

    def copy_audio_files_to_kaldi_dir(self, origin_paths: List[str], destination_path):

        print("Copying audio files to kaldi downloads directory...")
        if os.path.exists(destination_path):
            pass
        else:
            os.makedirs(destination_path)
        for path in origin_paths:
            copy_tree(path, destination_path)

    def get_data_dirs(self, root_dir):
        gender_dirs = ['female', 'male']

        audio_dirs = []
        transcript_paths = []
        speakers = []

        speaker_id = 0
        for gender_dir in gender_dirs:
            curdir = Path(root_dir, gender_dir)
            speaker_dirs = list(os.walk(curdir))[0][1]
            for idx, speaker_dir in enumerate(speaker_dirs):
                speaker_id += 1
                book_dirs = list(os.walk(Path(curdir, speaker_dir)))[0][1]
                for book_dir in book_dirs:
                    final_abs_path = Path(curdir, speaker_dir, book_dir).absolute()
                    audio_path = Path(final_abs_path, 'wavs')
                    transcript_path = Path(final_abs_path, 'metadata.csv')
                    audio_dirs.append(audio_path)
                    transcript_paths.append(transcript_path)
                    speakers.append(speaker_id)

        mix_dir = Path(root_dir, "mix")
        mix_book_dirs = list(os.walk(mix_dir))[0][1]

        for book_dir in mix_book_dirs:
            final_abs_path = Path(mix_dir, book_dir).absolute()
            audio_path = Path(final_abs_path, 'wavs')
            transcript_path = Path(final_abs_path, 'metadata.csv')
            audio_dirs.append(audio_path)
            transcript_paths.append(transcript_path)
            speakers.append(999)

        return audio_dirs, transcript_paths, speakers

    def generate_arrays(self, data: pd.DataFrame):

        wavscp = list()
        text = list()
        utt2spk = list()

        data['path'] = data['path'].apply(lambda x: "downloads/" + x)

        for idx, row in tqdm(data.iterrows(),total=data.shape[0]):
            transcript = self.clean_text(row['transcript_1'])
            file_path = row['path']
            speaker_id = f"speaker_{row['speaker']}"
            segment_id = f"segment_{idx}"
            utterance_id = f'{speaker_id}-{segment_id}'
            wavscp.append(f'{utterance_id} {file_path}')
            utt2spk.append(f'{utterance_id} {speaker_id}')
            text.append(f'{utterance_id} {transcript}')

        return wavscp, text, utt2spk
