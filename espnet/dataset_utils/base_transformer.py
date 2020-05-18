import os
import re
from abc import ABC, abstractmethod
from distutils.dir_util import copy_tree
from typing import List
from sklearn.model_selection import train_test_split


class AbstractDataTransformer(ABC):

    def __init__(self):
        self._prefix: str = None
        self.SUBSET_SIZE: int = None
        self.TESTSET_PROPORTION: float = 0.2
        self.kaldi_data_dir: str = None

    @property
    def prefix(self):
        if not self._prefix:
            raise ValueError("No prefix specified for current Data Tranformer")
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
        with open(os.path.join(directory, 'wav.scp'), 'w') as f1:
            f1.write('\n'.join(wavscp))
            f1.write('\n')
        with open(os.path.join(directory, 'text'), 'w') as f2:
            f2.write('\n'.join(text))
            f2.write('\n')
        with open(os.path.join(directory, 'utt2spk'), 'w') as f3:
            f3.write('\n'.join(utt2spk))
            f3.write('\n')

    def split_train_test(self, *args, test_proportion=None):
        train_test_args = train_test_split(*args, test_size=test_proportion or self.TESTSET_PROPORTION, random_state=42,
                                           shuffle=True)
        return train_test_args

    def clean_text(self, text):
        text = text.lower()
        # for now only removing punctuation, should add number2word later and other cleansing if relevant
        text = re.sub("[.,:;¡!?¿\-]+", ' ', text).strip()
        return text

    def copy_audio_files_to_kaldi_dir(self, origin_paths: List[str], destination_path):

        print("Copying audio files to kaldi downloads directory...")
        if os.path.exists(destination_path):
            pass
        else:
            os.makedirs(destination_path)
        for path in origin_paths:
            copy_tree(path, destination_path)
