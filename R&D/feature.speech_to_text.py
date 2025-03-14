'''
Requirements:
"qai-hub-models[whisper-tiny-en]"
sounddevice
numpy
'''

# AI Hub information
from qai_hub_models.models.whisper_tiny_en.model import WhisperTinyEn
from qai_hub_models.models._shared.whisper.app import WhisperApp
from qai_hub_models.models._shared.whisper.model import Whisper

import sounddevice as sd    # listening from microphone
import numpy as np          # turning audio into numpy arrays

DURATION : int = 2          # length of recording (adjust for transcription length)
SAMPLE_RATE : int = 16000   # the model computes in 16kHz (do not change)

def main():
    audio_file_name = record_audio(DURATION, SAMPLE_RATE)
    transcription = run_model(WhisperTinyEn, audio_file_name)
    print(transcription)


# runs a transcription an input voice
def run_model(model_cls : type[Whisper], audio_file : str):
    app = WhisperApp(model_cls.from_pretrained())

    audio, audio_sample_rate = load_audio(audio_file)

    return app.transcribe(audio, audio_sample_rate)


# returns the audio in numpy array format with the sample rate
def load_audio(audio_file : str) -> tuple[np.ndarray, int]:
    with np.load(audio_file + '.npz') as f:
        return f['audio'], SAMPLE_RATE


def record_audio(duration, sample_rate):
    file_name = 'speech_to_text'
    print('Recording for',duration,'seconds...')

    # duration * sample_rate is number of snapshots total over the duration
    # record in float32 for the model
    # channels = 1 is just mono audio
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait() # wait for the recording to end
    
    print('Recording complete.')

    # needs to turned into one-dimensional array for the model
    np.savez(file_name, audio=audio.flatten())

    return file_name

if __name__ == "__main__":
    main()