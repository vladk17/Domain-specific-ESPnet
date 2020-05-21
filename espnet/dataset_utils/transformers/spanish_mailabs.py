import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from dataset_utils.base_transformer import AbstractDataTransformer
import logging

logger = logging.root
SUBSET_SIZE = os.environ.get("ESPNET_SUBSET_SIZE", None)


class MailabsKaldiTransformer(AbstractDataTransformer):

    def __init__(self):
        super().__init__()
        self._prefix = 'mailabs'
        if SUBSET_SIZE:
            self.SUBSET_SIZE = int(SUBSET_SIZE)

    def transform(self, raw_data_path, espnet_kaldi_eg_directory, *args, **kwargs):

        self.kaldi_data_dir = os.path.join(espnet_kaldi_eg_directory, 'data')
        kaldi_audio_files_dir = os.path.join(espnet_kaldi_eg_directory, 'downloads')
        root_dir = os.path.join(raw_data_path, 'es_ES', 'by_book')

        audio_dirs, transcript_paths, speakers = self.get_data_dirs(root_dir)

        dfs = list()
        for idx, transcript_path in enumerate(transcript_paths):
            df = pd.read_csv(transcript_path, delimiter="|", error_bad_lines=False, engine='python',
                             warn_bad_lines=False)
            df.columns = ['path', 'transcript_1', 'transcript_2']
            df['speaker'] = speakers[idx]
            dfs.append(df)

        data = pd.concat(dfs)
        dataset_size = data.shape[0]
        logger.info(f"Total dataset size: {dataset_size}")

        self.copy_audio_files_to_kaldi_dir(origin_paths=audio_dirs, destination_path=kaldi_audio_files_dir)

        if self.SUBSET_SIZE:
            logger.info(f"Subset size: {self.SUBSET_SIZE}")
            if dataset_size < self.SUBSET_SIZE:
                logger.info(
                    f"ATTENTION! Provided subset size ({self.SUBSET_SIZE}) is less "
                    f"than overall dataset size ({dataset_size}). "
                    f"Taking all dataset")
            data = data[:self.SUBSET_SIZE]

        logger.info("Generating data")
        wavscp, text, utt2spk = self.generate_arrays(data)

        wavscp_train, wavscp_test, text_train, text_test, utt2spk_train, utt2spk_test = \
            self.split_train_test(wavscp,
                                  text,
                                  utt2spk)

        self.create_files(wavscp_train, text_train, utt2spk_train, 'train')
        self.create_files(wavscp_test, text_test, utt2spk_test, 'test')

    def get_data_dirs(self, root_dir):
        gender_dirs = list(os.walk(root_dir))[0][1]
        gender_dirs.remove('mix')

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
            speaker_id = -1
            final_abs_path = Path(mix_dir, book_dir).absolute()
            audio_path = Path(final_abs_path, 'wavs')
            transcript_path = Path(final_abs_path, 'metadata.csv')
            audio_dirs.append(audio_path)
            transcript_paths.append(transcript_path)
            speakers.append(speaker_id)

        return audio_dirs, transcript_paths, speakers

    def generate_arrays(self, data: pd.DataFrame):

        wavscp = list()
        text = list()
        utt2spk = list()

        data['path'] = data['path'].apply(lambda x: "downloads/" + x + '.wav')
        data = data.reset_index()

        for idx, row in tqdm(data.iterrows(), total=data.shape[0]):
            transcript = self.clean_text(row['transcript_1'])
            file_path = row['path']
            speaker_id = row['speaker']

            utt_id = idx+1
            utterance_id = f'{self.prefix}-{self.prefix}_{utt_id}'
            wavscp.append(f'{utterance_id} {file_path}')
            if row['speaker'] == -1:
                utt2spk.append(f'{utterance_id} {utterance_id}')
            else:
                utt2spk.append(f'{utterance_id} {speaker_id}')
            text.append(f'{utterance_id} {transcript}')

        return wavscp, text, utt2spk
