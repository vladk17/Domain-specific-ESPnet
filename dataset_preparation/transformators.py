from abc import ABC, abstractmethod
from tools.sph2wav import SPHFile

class AbstractDataTransformer(ABC):

    @abstractmethod
    def transform(self, path, *args, **kwargs):
        pass

class Tedlium2NeMoDataTransformer(AbstractDataTransformer):
    
    def __init__(self, sample_rate):
        self.sample_rate: int = sample_rate
    
    def transform(self, path):
        
        transformed_audio_path = os.path.join(path, 'transformed_audio')
        
        if not os.path.exists(transformed_audio_path):
            os.mkdir(transformed_audio_path)
                
        transformed_data = list()
        
        audio_dir = os.path.join(path,'sph')
        transripts_dir = os.path.join(path,'stm')
        
        for audio_path, transcript_path in zip(os.listdir(audio_dir),
                                  os.listdir(transripts_dir)):
            
            sph_audio = SPHFile(os.path.join(audio_dir, audio_path)).open()
            text_segments = self.get_segments_from_transcript(os.path.join(transripts_dir,transcript_path))
            audio_segments = self.get_audio_segments(os.path.join(audio_dir,audio_path), text_segments)
            
            for idx, audio_segment in enumerate(audio_segments):
                audio_segment_path = os.path.join(transformed_audio_path, audio_path[:-4])
                if not os.path.exists(audio_segment_path):
                    os.mkdir(audio_segment_path)
                audio_segment_chunk_path = os.path.join(audio_segment_path, f'{str(idx)}.wav')

                sph_audio.write_wav(audio_segment_chunk_path, audio_segment['start'], audio_segment['end'])
                transformed_data.append({'transcript':audio_segment['transcript'],
                                         'audio':audio_segment_chunk_path})
        return transformed_data
            
            
    def get_segments_from_transcript(self, transcript_path):
        with open(transcript_path) as transcript_fd:
            transcripts = transcript_fd.read()
            segments = [ent for ent in transcripts.split('\n') 
                        if '' != ent]
        return segments

    def get_audio_segments(self, audio_path, text_segments):
        _segments = list()
        for idx, segment in enumerate(text_segments):
            the_md, the_transcript = re.split(r' <.+> ',segment)
            if the_transcript.strip() == 'ignore_time_segment_in_scoring':
                continue
            left_timestamp, right_timestamp = map(float,the_md.split(' ')[-2:])
            left_idx = left_timestamp
            right_idx = right_timestamp
            _segments.append({'audio_path':audio_path,
                              'start': left_idx,
                              'end': right_idx,
                              'transcript':the_transcript})
        return _segments
    
class TEDxSpanish2KaldiTransformer(AbstractDataTransformer):
    
    def transform(path):
        pass