import os
from abc import ABC, abstractmethod
import re

from sklearn.model_selection import train_test_split
from num2words import num2words


class AbstractDataTransformer(ABC):

    def __init__(self):
        self.SUBSET_SIZE: int = 999999999999999

    @abstractmethod
    def transform(self, raw_data_path, espnet_kaldi_eg_directory, *args, **kwargs):
        pass

    def create_files(self, wavscp, text, utt2spk, directory):
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

    def split_train_test(self, *args, test_proportion=0.2):
        train_test_args = train_test_split(*args, test_size=test_proportion, random_state=42, shuffle=False)
        return train_test_args

    def clean_text(self, text):
        text = text.lower()
        # for now only removing punctuation, should add number2word later and other cleansing if relevant
        clean = re.sub("\W+", ' ', text)
        return clean
