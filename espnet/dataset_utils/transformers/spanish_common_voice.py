import os
from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm
import pandas as pd
from sklearn import preprocessing
import multiprocessing

from dataset_utils.base_transformer import AbstractDataTransformer

SUBSET_SIZE = os.environ.get("ESPNET_SUBSET_SIZE", None)


class CommonVoiceKaldiTransformer(AbstractDataTransformer):

    def __init__(self):
        super().__init__()
        if self.SUBSET_SIZE:
            self.SUBSET_SIZE = int(self.SUBSET_SIZE)

    def transform(self, raw_data_path, espnet_kaldi_eg_directory, *args, **kwargs):

        kaldi_data_dir = os.path.join(espnet_kaldi_eg_directory, 'data')
        kaldi_audio_files_dir = os.path.join(espnet_kaldi_eg_directory, 'downloads')
        self.kaldi_preprocessed_audio_folder = kaldi_audio_files_dir
        origin_audiofiles_dir = os.path.join(raw_data_path, 'clips')

        data = pd.read_csv(Path(raw_data_path, 'validated.tsv'), delimiter='\t')
        dataset_size = data.shape[0]

        print("Total dataset size", dataset_size)

        if self.SUBSET_SIZE:
            print("Subset size:", self.SUBSET_SIZE)
            if dataset_size < self.SUBSET_SIZE:
                print(
                    f"ATTENTION! Provided subset size ({self.SUBSET_SIZE}) is less "
                    f"than overall dataset size ({dataset_size}). "
                    f"Taking all dataset")
            data = data[:self.SUBSET_SIZE]

        audio_files = data['path'].tolist()
        if os.path.exists(kaldi_audio_files_dir):
            pass
        else:
            print("Transforming audio to .wav and copying to eg directory")
            os.makedirs(kaldi_audio_files_dir)

        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        for _ in tqdm(pool.map(self.convert_to_wav_from_mp3(audio_files)), total=len(audio_files)):
            pass

        # for file in tqdm(audio_files):
        #     joined_path = os.path.join(origin_audiofiles_dir, file)
        #     self.convert_to_wav_from_mp3(joined_path)
        print("Generating train and test files")

        wavscp, text, utt2spk = self.generate_arrays(data)

        wavscp_train, wavscp_test, text_train, text_test, utt2spk_train, utt2spk_test = \
            self.split_train_test(wavscp,
                                  text,
                                  utt2spk,
                                  test_proportion=0.2)

        self.create_files(wavscp_train, text_train, utt2spk_train, os.path.join(kaldi_data_dir, 'train'))
        self.create_files(wavscp_test, text_test, utt2spk_test, os.path.join(kaldi_data_dir, 'test'))

    def convert_to_wav_from_mp3(self, source_path: str):

        new_file_name = source_path.split("/")[-1][:-4] + '.wav'
        destination_path = Path(self.kaldi_preprocessed_audio_folder, new_file_name)
        sound = AudioSegment.from_mp3(source_path)
        sound.export(destination_path, format="wav")

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
            speaker_id = row['client_id']
            segment_id = idx
            utterance_id = f'speaker{speaker_id}-segmemt{segment_id}'
            wavscp.append(f'{utterance_id} {file_path}')
            utt2spk.append(f'{utterance_id} {speaker_id}')
            text.append(f'{utterance_id} {transcript}')

        return wavscp, text, utt2spk
