import logging
import os
from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm
import pandas as pd
from sklearn import preprocessing


from dataset_utils.base_transformer import AbstractDataTransformer

SUBSET_SIZE = os.environ.get("ESPNET_SUBSET_SIZE", None)
logger = logging.root

class CommonVoiceKaldiTransformer(AbstractDataTransformer):

    def __init__(self):
        super().__init__()
        self._prefix = 'comvoice'
        if SUBSET_SIZE:
            self.SUBSET_SIZE = int(SUBSET_SIZE)

    def transform(self, raw_data_path, espnet_kaldi_eg_directory, force_transform_audio=False, *args, **kwargs):

        self.kaldi_data_dir = os.path.join(espnet_kaldi_eg_directory, 'data')
        kaldi_audio_files_dir = os.path.join(espnet_kaldi_eg_directory, 'downloads')
        self.kaldi_preprocessed_audio_folder = kaldi_audio_files_dir

        origin_audiofiles_dir = os.path.join(raw_data_path, 'clips')
        data = pd.read_csv(Path(raw_data_path, 'validated.tsv'), delimiter='\t')
        dataset_size = data.shape[0]

        logger.info(f"Total dataset size : {dataset_size}")

        if self.SUBSET_SIZE:
            logger.info(f"Subset size: {self.SUBSET_SIZE}")
            if dataset_size < self.SUBSET_SIZE:
                logger.info(
                    f"ATTENTION! Provided subset size ({self.SUBSET_SIZE}) is less "
                    f"than overall dataset size ({dataset_size}). "
                    f"Taking all dataset")
            data = data[:self.SUBSET_SIZE]

        # self.transform_audio(origin_audiofiles_dir, kaldi_audio_files_dir, data, force_transform_audio)

        logger.info("Generating train and test files")

        wavscp, text, utt2spk = self.generate_arrays(data)

        wavscp_train, wavscp_test, text_train, text_test, utt2spk_train, utt2spk_test = \
            self.split_train_test(wavscp,
                                  text,
                                  utt2spk,
                                  test_proportion=0.2)

        self.create_files(wavscp_train, text_train, utt2spk_train, 'train')
        self.create_files(wavscp_test, text_test, utt2spk_test, 'test')

    def transform_audio(self, origin_audiofiles_dir, kaldi_audio_files_dir, data, force_transform_audio):
        logger.info("Transforming audio to .wav and copying to eg directory")
        audio_files = [os.path.join(origin_audiofiles_dir, audio_path) for audio_path in data['path'].tolist()]
        if os.path.exists(kaldi_audio_files_dir):
            logger.info("Data directory already exists")
            if force_transform_audio:
                for a_path in tqdm(audio_files):
                    self.convert_to_wav_from_mp3(a_path)
        else:
            logger.info("Creating data directory")
            os.makedirs(kaldi_audio_files_dir)
            for a_path in tqdm(audio_files):
                self.convert_to_wav_from_mp3(a_path)

    def convert_to_wav_from_mp3(self, source_path: str):
        new_file_name = source_path.split("/")[-1][:-4] + '.wav'
        upsampled_wav_path = Path(self.kaldi_preprocessed_audio_folder, new_file_name + '.upsample')
        downsampled_wav_path = Path(self.kaldi_preprocessed_audio_folder, new_file_name)
        sound = AudioSegment.from_mp3(source_path)
        sound.export(upsampled_wav_path, format="wav")
        os.system("sox '%s' -r 16000 -b 16 -c 1 %s" % (upsampled_wav_path, downsampled_wav_path))
        os.remove(upsampled_wav_path)

    def generate_arrays(self, data: pd.DataFrame):

        wavscp = list()
        text = list()
        utt2spk = list()

        data['path'] = data['path'].apply(lambda x: "downloads/" + x[:-4] + '.wav')
        le = preprocessing.LabelEncoder()
        data['client_id'] = le.fit_transform(data['client_id'])
        for idx, row in data.iterrows():
            transcript = self.clean_text(row['sentence'])
            file_path = row['path']

            utt_id = idx + 1
            speaker_id = f"self.prefix{row['client_id']}"
            utterance_id = f'{speaker_id}-{self.prefix}{utt_id}'

            wavscp.append(f'{utterance_id} {file_path}')
            utt2spk.append(f'{utterance_id} {speaker_id}')
            text.append(f'{utterance_id} {transcript}')

        return wavscp, text, utt2spk
