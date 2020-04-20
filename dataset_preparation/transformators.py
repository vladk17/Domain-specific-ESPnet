import os
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from distutils.dir_util import copy_tree
import re

# from tools.sph2wav import SPHFile
from settings import TRAIN_TEST_SIZE


class AbstractDataTransformer(ABC):

    @abstractmethod
    def transform(self, raw_data_path, *args, **kwargs):
        pass


class TEDxSpanish2KaldiTransformer(AbstractDataTransformer):

    def transform(self, raw_data_path, espnet_kaldi_eg_directory):

        kaldi_data_dir = os.path.join(espnet_kaldi_eg_directory, 'data')
        kaldi_audio_files_dir = os.path.join(espnet_kaldi_eg_directory, 'downloads')

        # copy audio files to separate directory according to kaldi directory conventions
        print("Copying files to kaldi download directory")
        fromDirectory = os.path.join(raw_data_path, 'speech')
        toDirectory = kaldi_audio_files_dir
        copy_tree(fromDirectory, toDirectory)
        print("Generating train and test files")
        wavscp, text, utt2spk = self.generate_arrays(raw_data_path, kaldi_audio_files_dir)

        wavscp = wavscp[:TRAIN_TEST_SIZE]
        text = text[:TRAIN_TEST_SIZE]
        utt2spk = utt2spk[:TRAIN_TEST_SIZE]

        wavscp_train, text_test, utt2spk_train, wavscp_test, text_train, utt2spk_test = \
            self.split_train_test(wavscp,
                                  text,
                                  utt2spk,
                                  test_proportion=0.25)

        self.create_files(wavscp_train, utt2spk_train, text_train, os.path.join(kaldi_data_dir, 'train'))
        self.create_files(wavscp_test, utt2spk_test, text_test, os.path.join(kaldi_data_dir, 'test'))

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

#
# class Tedlium2NeMoDataTransformer(AbstractDataTransformer):
#
#     def __init__(self, sample_rate):
#         self.sample_rate: int = sample_rate
#
#     def transform(self, path):
#
#         transformed_audio_path = os.path.join(path, 'transformed_audio')
#
#         if not os.path.exists(transformed_audio_path):
#             os.mkdir(transformed_audio_path)
#
#         transformed_data = list()
#
#         audio_dir = os.path.join(path, 'sph')
#         transripts_dir = os.path.join(path, 'stm')
#
#         for audio_path, transcript_path in zip(os.listdir(audio_dir),
#                                                os.listdir(transripts_dir)):
#
#             sph_audio = SPHFile(os.path.join(audio_dir, audio_path)).open()
#             text_segments = self.get_segments_from_transcript(os.path.join(transripts_dir, transcript_path))
#             audio_segments = self.get_audio_segments(os.path.join(audio_dir, audio_path), text_segments)
#
#             for idx, audio_segment in enumerate(audio_segments):
#                 audio_segment_path = os.path.join(transformed_audio_path, audio_path[:-4])
#                 if not os.path.exists(audio_segment_path):
#                     os.mkdir(audio_segment_path)
#                 audio_segment_chunk_path = os.path.join(audio_segment_path, f'{str(idx)}.wav')
#
#                 sph_audio.write_wav(audio_segment_chunk_path, audio_segment['start'], audio_segment['end'])
#                 transformed_data.append({'transcript': audio_segment['transcript'],
#                                          'audio': audio_segment_chunk_path})
#         return transformed_data
#
#     def get_segments_from_transcript(self, transcript_path):
#         with open(transcript_path) as transcript_fd:
#             transcripts = transcript_fd.read()
#             segments = [ent for ent in transcripts.split('\n')
#                         if '' != ent]
#         return segments
#
#     def get_audio_segments(self, audio_path, text_segments):
#         _segments = list()
#         for idx, segment in enumerate(text_segments):
#             the_md, the_transcript = re.split(r' <.+> ', segment)
#             if the_transcript.strip() == 'ignore_time_segment_in_scoring':
#                 continue
#             left_timestamp, right_timestamp = map(float, the_md.split(' ')[-2:])
#             left_idx = left_timestamp
#             right_idx = right_timestamp
#             _segments.append({'audio_path': audio_path,
#                               'start': left_idx,
#                               'end': right_idx,
#                               'transcript': the_transcript})
#         return _segments
