import logging
import os
import pandas as pd
import subprocess

from dataset_utils.base_transformer import AbstractDataTransformer

SUBSET_SIZE = os.environ.get("ESPNET_SUBSET_SIZE", None)
logger = logging.root

class Kaggle120hSpanish2KaldiTransformer(AbstractDataTransformer):

    def __init__(self):
        super().__init__()
        self._prefix = 'kaggle_120h'
        if SUBSET_SIZE:
            self.SUBSET_SIZE = int(SUBSET_SIZE)

    def transform(self, raw_data_path, espnet_kaldi_eg_directory, *args, **kwargs):

        raw_data_path = os.path.join(raw_data_path, r'asr-spanish-v1-carlfm01')
        self.kaldi_data_dir = os.path.join(espnet_kaldi_eg_directory, 'data')
        kaldi_audio_files_dir = os.path.join(espnet_kaldi_eg_directory, 'downloads')

        # copy audio files to separate directory according to kaldi directory conventions
        logger.info("Copying files to kaldi download directory")

        origin_audio_dir = [os.path.join(raw_data_path, 'audios')]
        destination_audio_dir = os.path.join(kaldi_audio_files_dir, self.prefix)

        should_cpy = False
        if not os.path.exists(destination_audio_dir):
            should_cpy = True
        else:
            proc = subprocess.Popen([f'ls -l {destination_audio_dir} | wc -l'], stdout=subprocess.PIPE, shell=True)
            (out, _) = proc.communicate()
            print (f"{destination_audio_dir} has {int(out[:-1])} entries")
            if 112846 != int(out[:-1]):
                should_cpy = True

        if should_cpy:
            print(f'copying audio files to {destination_audio_dir}')
            self.copy_audio_files_to_kaldi_dir(origin_paths=origin_audio_dir,
                                               destination_path=destination_audio_dir)
        else:
            print('all audio files are in place')

        wavscp, text, utt2spk = self.generate_arrays(raw_data_path)

        logger.info(f"Total dataset size: {len(text)}")
        if len(text) < self.SUBSET_SIZE:
            logger.info(
                f"ATTENTION! Provided subset size ({self.SUBSET_SIZE}) is more than overall dataset size ({len(text)}). "
                f"Taking all dataset")
        if self.SUBSET_SIZE:
            logger.info(f"Subset size: {self.SUBSET_SIZE}")
            wavscp = wavscp[:self.SUBSET_SIZE]
            text = text[:self.SUBSET_SIZE]
            utt2spk = utt2spk[:self.SUBSET_SIZE]

        logger.info("Splitting train-test")
        wavscp_train, wavscp_test, text_train, text_test, utt2spk_train, utt2spk_test = \
            self.split_train_test(wavscp,
                                  text,
                                  utt2spk)

        self.create_files(wavscp_train, text_train, utt2spk_train, 'train')
        self.create_files(wavscp_test, text_test, utt2spk_test, 'test')

    def generate_arrays(self, path):
        wavscp = list()
        text = list()
        utt2spk = list()

        transcripts_path = os.path.join(path, 'files.csv')
        df = pd.read_csv(transcripts_path, index_col='wav_filename')

        the_text_series = df['transcript'].apply(lambda x: x.lower())
        # the_text_series = the_text_series.apply(lambda x: x.lower())
        the_text_series.index = [(lambda x: x+'_'+x)(ent.split('.')[0].split('/')[1]) for ent in df.index]
        the_text_series_sorted = the_text_series.sort_index()

        text = [index+' '+the_text_series_sorted.loc[index] for index in the_text_series_sorted.index]
        wavscp = [index+' '+os.path.join('downloads',self._prefix, index.split('_')[1]+'.wav') for index in the_text_series_sorted.index]
        utt2spk = [index+' '+index.split('_')[1] for index in the_text_series_sorted.index]

        return wavscp, text, utt2spk
