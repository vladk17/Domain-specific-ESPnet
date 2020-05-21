import logging
import os
from distutils.dir_util import copy_tree

from dataset_utils.base_transformer import AbstractDataTransformer

SUBSET_SIZE = os.environ.get("ESPNET_SUBSET_SIZE", None)
logger = logging.root

class TEDxSpanish2KaldiTransformer(AbstractDataTransformer):

    def __init__(self):
        super().__init__()
        self._prefix = 'tedx'
        if SUBSET_SIZE:
            self.SUBSET_SIZE = int(SUBSET_SIZE)

    def transform(self, raw_data_path, espnet_kaldi_eg_directory, *args, **kwargs):

        raw_data_path = os.path.join(raw_data_path, 'tedx_spanish_corpus')
        self.kaldi_data_dir = os.path.join(espnet_kaldi_eg_directory, 'data')
        kaldi_audio_files_dir = os.path.join(espnet_kaldi_eg_directory, 'downloads')

        # copy audio files to separate directory according to kaldi directory conventions
        logger.info("Copying files to kaldi download directory")
        fromDirectory = os.path.join(raw_data_path, 'speech')
        toDirectory = kaldi_audio_files_dir
        self.copy_audio_files_to_kaldi_dir([fromDirectory], toDirectory)

        wavscp, text, utt2spk = self.generate_arrays(raw_data_path)

        logger.info(f"Total dataset size: {len(text)}")
        if len(text) < self.SUBSET_SIZE:
            logger.info(
                f"ATTENTION! Provided subset size ({self.SUBSET_SIZE}) is less than overall dataset size ({len(text)}). "
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
                                  utt2spk,
                                  test_proportion=0.25)

        self.create_files(wavscp_train, text_train, utt2spk_train, 'train')
        self.create_files(wavscp_test, text_test, utt2spk_test, 'test')

    def generate_arrays(self, path):
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
            # utt_id = '_'.join(utterance_tokens[2:])

            utt_id = idx+1
            utterance_id = f'{self.prefix}{speaker_id}-{self.prefix}_{utt_id}'
            wavscp.append(f'{utterance_id} {file_path}')
            utt2spk.append(f'{utterance_id} {speaker_id}')
            text.append(f'{utterance_id} {transcript}')

        return wavscp, text, utt2spk
