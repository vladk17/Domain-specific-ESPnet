import os
from pathlib import Path

from pydub import AudioSegment
from tqdm import tqdm

from transformators import AbstractDataTransformer


class CommonVoiceKaldiTransformer(AbstractDataTransformer):

    def transform(self, raw_data_path, espnet_kaldi_eg_directory, subset_size=None, *args, **kwargs):
        kaldi_data_dir = os.path.join(espnet_kaldi_eg_directory, 'data')
        kaldi_audio_files_dir = os.path.join(espnet_kaldi_eg_directory, 'downloads')

        # copy audio files to separate directory according to kaldi directory conventions
        print("Transforming audio to wav and copying to eg directory")
        origin_audiofiles_dir = os.path.join(raw_data_path, 'clips')
        files = list(os.walk(origin_audiofiles_dir))[2]

        print("Total dataset size", len(files))
        if subset_size:
            print("Subset size:", subset_size)
            if len(files) < subset_size:
                print(
                    f"ATTENTION! Provided subset size ({subset_size}) is less than overall dataset size ({len(files)}). "
                    f"Taking all dataset")
            self.SUBSET_SIZE = subset_size

        for file in tqdm(files[:self.SUBSET_SIZE]):
            joined_path = os.path.join(origin_audiofiles_dir, file)
            self.convert_to_wav_from_mp3(joined_path, kaldi_audio_files_dir)

        # print("Generating train and test files")
        #
        # wavscp, text, utt2spk = self.generate_arrays(raw_data_path, kaldi_audio_files_dir)
        #
        # wavscp = wavscp[:self.SUBSET_SIZE]
        # text = text[:self.SUBSET_SIZE]
        # utt2spk = utt2spk[:self.SUBSET_SIZE]
        #
        # wavscp_train, wavscp_test, text_train, text_test, utt2spk_train, utt2spk_test = \
        #     self.split_train_test(wavscp,
        #                           text,
        #                           utt2spk,
        #                           test_proportion=0.25)
        #
        # self.create_files(wavscp_train, text_train, utt2spk_train, os.path.join(kaldi_data_dir, 'train'))
        # self.create_files(wavscp_test, text_test, utt2spk_test, os.path.join(kaldi_data_dir, 'test'))

    def convert_to_wav_from_mp3(self, source_path: str, destination_folder: str):
        new_file_name = source_path.split("/")[-1][:-4] + '.wav'
        destination_path = Path(destination_folder,new_file_name)
        sound = AudioSegment.from_mp3(source_path)
        sound.export(destination_path, format="wav")

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
            segment_id = '_'.join(utterance_tokens[2:])
            utterance_id = f'{speaker_id}-{segment_id}'
            wavscp.append(f'{utterance_id} {file_path}')
            utt2spk.append(f'{utterance_id} {speaker_id}')
            text.append(f'{utterance_id} {transcript}')

        return wavscp, text, utt2spk