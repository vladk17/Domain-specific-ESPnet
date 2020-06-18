import json
import logging
import os

from pydub import AudioSegment
from tqdm import tqdm

from dataset_utils.base_transformer import AbstractDataTransformer
from dataset_utils.conf import UTTERANCE_MIN_LENGTH

SUBSET_SIZE = os.environ.get("ESPNET_SUBSET_SIZE", None)
logger = logging.root


class GongSpanish2KaldiTransformer(AbstractDataTransformer):

    def __init__(self):
        super().__init__()
        self._prefix = 'gong'
        self.overall_duration = 0
        if SUBSET_SIZE:
            self.SUBSET_SIZE = int(SUBSET_SIZE)

    def transform(self, raw_data_path, espnet_kaldi_eg_directory, *args, **kwargs):
        self.kaldi_eg_dir = espnet_kaldi_eg_directory
        raw_data_path = os.path.join(raw_data_path, 'to-y-data', 'spanish_test_set_second_pass')
        self.kaldi_data_dir = os.path.join(espnet_kaldi_eg_directory, 'data')
        kaldi_audio_files_dir = os.path.join(espnet_kaldi_eg_directory, 'downloads')

        # copy audio files to separate directory according to kaldi directory conventions

        transcripts_dir = os.path.join(raw_data_path)
        transcript_paths = list(os.walk(transcripts_dir))[0][2]
        texts = []
        chunks = []
        cut_audio_paths = []
        for transcript_path in tqdm(transcript_paths):
            try:
                cur_texts, cur_chunks, cur_cut_audio_paths = self.cut_audio_to_monologues(raw_data_path,
                                                                                          transcript_path)
                texts.extend(cur_texts)
                chunks.extend(cur_chunks)
                cut_audio_paths.extend(cur_cut_audio_paths)
            except Exception as e:
                logger.error(f"EXCEPTION {e}")
        logger.info(f'Total {self.__class__.__name__} dataset duration: {round(self.overall_duration/3600, 2)} hours')

        best_monologue_indexes = [idx for idx, chunk in enumerate(chunks) if chunk[2] > UTTERANCE_MIN_LENGTH
                                  and "<unk>" not in texts[idx]
                                  and "+" not in chunk[3]
                                  and len(texts[idx]) > 0]
        texts = [texts[idx] for idx in best_monologue_indexes]
        cut_audio_paths = [cut_audio_paths[idx] for idx in best_monologue_indexes]
        speakers = [chunks[idx][3] for idx in best_monologue_indexes]
        origin_audio_dir = [os.path.join(raw_data_path, 'cut_audio')]
        destination_audio_dir = os.path.join(kaldi_audio_files_dir, self.prefix)
        self.copy_audio_files_to_kaldi_dir(origin_paths=origin_audio_dir,
                                           destination_path=destination_audio_dir)

        wavscp, text, utt2spk = self.generate_arrays(texts, speakers, cut_audio_paths)

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

    def cut_audio_to_monologues(self, relative_path, transcript_path):
        json_path = os.path.join(relative_path, transcript_path)
        wav_path = f"{transcript_path[:-5]}.raw-audio.wav"
        with open(json_path, 'r', encoding="utf8") as f:
            data = json.load(f)
            self.overall_duration += data['monologues'][-1]['end']
            chunks = [
                (
                    utterance['start'], utterance['end'], utterance['end'] - utterance['start'],
                    utterance['speaker']['id'])
                for utterance in
                data['monologues']]
            texts = []
            for idx, utterance in enumerate(data['monologues']):
                text = " ".join([_['text'] for _ in utterance['terms']])
                texts.append(text)
            assert len(chunks) == len(
                texts), "Length of texts is not equal to length of chunks in the transcript file"
            cut_audio_paths = self.cut_audio_to_chunks(base_dir=relative_path,
                                                       unprocessed_dir_prefix='audio',
                                                       processed_dir_prefix='cut_audio',
                                                       wav_path=wav_path, chunks=chunks)
            return texts, chunks, cut_audio_paths

    def generate_arrays(self, origin_texts, speakers, audio_paths):
        wavscp = list()
        text = list()
        utt2spk = list()

        for idx, transcript in enumerate(origin_texts):

            file_path = "downloads/" + self.prefix + "/" + audio_paths[idx]
            absolute_path = os.path.join(self.kaldi_eg_dir, file_path)
            if os.path.exists(absolute_path):
                audio = AudioSegment.from_file(absolute_path)
                duration = audio.duration_seconds
                if duration > UTTERANCE_MIN_LENGTH:
                    utt_id = idx + 1
                    speaker_id = self.prefix + 'sp' + ''.join(speakers[idx]).replace(' ', '')
                    utterance_id = f'{speaker_id}-{self.prefix}{utt_id}'
                    wavscp.append(f'{utterance_id} {file_path}')
                    utt2spk.append(f'{utterance_id} {speaker_id}')
                    text.append(f'{utterance_id} {transcript}')

        return wavscp, text, utt2spk
