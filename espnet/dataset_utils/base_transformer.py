import os
from abc import ABC, abstractmethod
from copy import copy
from distutils.dir_util import copy_tree
from typing import List

from pydub import AudioSegment
from sklearn.model_selection import train_test_split
import logging
import pandas as pd

logger = logging.root


class AbstractDataTransformer(ABC):

    def __init__(self):
        self._prefix: str = None
        self.SUBSET_SIZE: int = 999999999999999
        self.TESTSET_PROPORTION: float = 0.1
        self.kaldi_data_dir: str = None
        self.kaldi_eg_dir: str = None

    @property
    def prefix(self):
        if not self._prefix:
            raise ValueError("No prefix specified for current Data Transformer")
        return self._prefix

    @abstractmethod
    def transform(self, raw_data_path, espnet_kaldi_eg_directory, *args, **kwargs):
        pass

    def create_files(self, wavscp, text, utt2spk, dataset_part):
        if dataset_part == 'train':
            directory = os.path.join(self.kaldi_data_dir, f'train_{self.prefix}')
            self._create_files(directory, wavscp, text, utt2spk)
        if dataset_part == 'test':
            directory = os.path.join(self.kaldi_data_dir, f'test_{self.prefix}')
            self._create_files(directory, wavscp, text, utt2spk)

    def _create_files(self, directory, wavscp, text, utt2spk):
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(os.path.join(directory, 'wav.scp'), 'w', encoding="utf-8") as f1:
            f1.write('\n'.join(wavscp))
            f1.write('\n')
        with open(os.path.join(directory, 'text', ), 'w', encoding="utf-8") as f2:
            f2.write('\n'.join(text))
            f2.write('\n')
        with open(os.path.join(directory, 'utt2spk'), 'w', encoding="utf-8") as f3:
            f3.write('\n'.join(utt2spk))
            f3.write('\n')

    def split_train_test(self, wavscp, text, utt2spk, test_proportion=None, shuffle=False):
        df = pd.DataFrame(data=[wavscp, text, utt2spk]).T
        df.columns = ['wavscp', 'text', 'utt2spk']
        df['speaker'] = df['utt2spk'].apply(lambda x: x.split(' ')[1])
        df = df.sort_values('speaker')
        wavscp, text, utt2spk = df['wavscp'].to_list(), df['text'].to_list(), df['utt2spk'].to_list()
        train_test_args = train_test_split(wavscp,
                                           text,
                                           utt2spk, test_size=test_proportion or self.TESTSET_PROPORTION,
                                           random_state=42,
                                           shuffle=shuffle)
        return train_test_args

    def clean_text(self, text):
        # text = text.lower()
        # text = re.sub("[.,:;¡!?¿\-]+", ' ', text).strip()
        return text

    def copy_audio_files_to_kaldi_dir(self, origin_paths: List[str], destination_path):

        logger.info("Copying audio files to kaldi downloads directory...")
        if os.path.exists(destination_path):
            pass
        else:
            os.makedirs(destination_path)
        for path in origin_paths:
            copy_tree(path, destination_path)

    def downsample_audio(self, source_path: str, frequency=16, channels=1):
        downsampled_path = list(copy(source_path))
        downsampled_path[-4:] = '.out.wav'
        downsampled_path = ''.join(downsampled_path)
        os.system("sox %s -r 16000 -b 16 -c 1 %s" % (source_path, downsampled_path))
        os.remove(source_path)
        os.rename(downsampled_path, source_path)

    def cut_audio_to_chunks(self, base_dir: str, wav_path: str, unprocessed_dir_prefix,
                            processed_dir_prefix, chunks=List[tuple]):
        """
        Takes as input audio file and chunks in format [(0,1.2),(1.2,4.5),...] and returns list of new audio file paths
        accordingly to chunks length
        """
        cut_audio_dir = os.path.join(base_dir, processed_dir_prefix)
        if not os.path.exists(cut_audio_dir):
            os.makedirs(cut_audio_dir)
        sound = AudioSegment.from_wav(os.path.join(base_dir, unprocessed_dir_prefix, wav_path))
        created_files: List[str] = []
        for idx, chunk in enumerate(chunks):
            start = chunk[0] * 1000
            end = chunk[1] * 1000
            chunk_sound = sound[start:end]
            created_files.append(f"{wav_path}".replace(".wav", f"{idx + 1}.wav"))
            new_file_path = f"{os.path.join(base_dir, processed_dir_prefix, wav_path)}".replace(".wav",
                                                                                                f"{idx + 1}.wav")
            chunk_sound.export(new_file_path, format='wav')
        return created_files
