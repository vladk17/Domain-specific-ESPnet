import os
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from distutils.dir_util import copy_tree
import re

class AbstractDataTransformer(ABC):

    @abstractmethod
    def transform(self, raw_data_path, *args, **kwargs):
        pass


class TEDxSpanish2KaldiTransformer(AbstractDataTransformer):

    def transform(self, raw_data_path, espnet_kaldi_eg_directory, subset_size=None):

        kaldi_data_dir = os.path.join(espnet_kaldi_eg_directory, 'data')
        kaldi_audio_files_dir = os.path.join(espnet_kaldi_eg_directory, 'downloads')

        # copy audio files to separate directory according to kaldi directory conventions
        print("Copying files to kaldi download directory")
        fromDirectory = os.path.join(raw_data_path, 'speech')
        toDirectory = kaldi_audio_files_dir
        copy_tree(fromDirectory, toDirectory) 
        print("Generating train and test files")
        wavscp, text, utt2spk = self.generate_arrays(raw_data_path, kaldi_audio_files_dir)

        if subset_size:
            print("Subset size:", subset_size)
            wavscp = wavscp[:subset_size]
            text = text[:subset_size]
            utt2spk = utt2spk[:subset_size]

        wavscp_train, wavscp_test, text_train, text_test, utt2spk_train,  utt2spk_test = \
            self.split_train_test(wavscp,
                                  text,
                                  utt2spk,
                                  test_proportion=0.25)

        self.create_files(wavscp_train, text_train, utt2spk_train, os.path.join(kaldi_data_dir, 'train'))
        self.create_files(wavscp_test, text_test, utt2spk_test, os.path.join(kaldi_data_dir, 'test'))

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

    def generate_arrays(self, path, audio_files_dir):
        wavscp = list()
        text = list()
        utt2spk = list()

        files_path = os.path.join(path, 'files', 'TEDx_Spanish.paths')
        transcripts_path = os.path.join(path, 'files', 'TEDx_Spanish.transcription')

        with open(files_path) as f1:
            files = list()
            _files = f1.read().splitlines()
            for _ in _files:
                files.append(os.path.join('downloads', _[9:]))

        with open(transcripts_path) as f2:
            transcripts = f2.read().splitlines()
        assert len(files) == len(transcripts), "Number of files is not "

        for idx, transcript in enumerate(transcripts):
            tokens = transcript.lower().split(' ')
            transcript = ' '.join(tokens[:-1])
            file_path = files[idx]
            utterance_tokens = tokens[-1][5:].split('_')
            speaker_id = '_'.join(utterance_tokens[:2])
            segment_id = '_'.join(utterance_tokens[2:])
            utterance_id = f'{speaker_id}-{segment_id}'
            wavscp.append(f'{utterance_id} {file_path}')
            utt2spk.append(f'{utterance_id} {speaker_id}')
            text.append(f'{utterance_id} {transcript}')

        return wavscp, text, utt2spk

    def split_train_test(self, *args, test_proportion):
        train_test_args = train_test_split(*args, test_size=test_proportion, random_state=42, shuffle=False)
        return train_test_args