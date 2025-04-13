'''
Requirements:
"qai-hub-models[whisper-tiny-en]"
sounddevice
numpy
pyautogui
'''

# Whisper Model information
from qai_hub_models.models.whisper_tiny_en.model import WhisperTinyEn
from qai_hub_models.models._shared.whisper.app import WhisperApp
from qai_hub_models.models._shared.whisper.model import Whisper

# Listening from microphone
import sounddevice as sd
import numpy as np

# Interpretting words into commands
import queue
import string
import pyautogui

# Allows for mulithreading purposes
import threading
import time

from collections.abc import Callable

class SpeechToCommand:
    SAMPLE_RATE = 16000                         # Whisper model works off 16kHz
    DURATION = 3                                # Length of chunks fed into model
    MODEL_CLS : type[Whisper] = WhisperTinyEn   # Model classifiation

    isActive : bool = False                     # Determines whether commands are executed or not
    isMousePaused : bool = False                # Determines if mouse is paused by voice commands

    def __init__(self, cb_calibrate : Callable, debugMode : bool = False):
        # Queue of input words to be read
        self.audio_queue = queue.Queue()

        # Used to remove non-alphanumeric characters in transcription
        self.translator = str.maketrans('', '', string.punctuation)

        # Instance of the transcribing app
        self.app = WhisperApp(SpeechToCommand.MODEL_CLS.from_pretrained())

        # Turns on/off the print debug
        self.debugMode = debugMode

        # Dictionary for all commands
        self.commands = {}

        # Default commands
        self.commands['left'] = lambda : self.__perform_if_active(lambda: pyautogui.click(button='left'))
        self.commands['right'] = lambda : self.__perform_if_active(lambda: pyautogui.click(button='right'))

        self.commands['hold'] = lambda : self.__perform_if_active(pyautogui.mouseDown)
        self.commands['release'] = lambda : self.__perform_if_active(pyautogui.mouseUp)

        self.commands['pause mouse'] = lambda : self.__perform_if_active(lambda: setattr(self, 'isMousePaused', True))
        self.commands['resume mouse'] = lambda : self.__perform_if_active(lambda: setattr(self, 'isMousePaused', False))

        self.commands['start listening'] = lambda: (print('Commands enabled ✔️'), setattr(self, 'isActive', True))
        self.commands['stop listening'] = lambda: (print('Commands disabled ❌'), setattr(self, 'isActive', False))

        # Callbacks for recalibrating
        self.commands['calibrate'] = lambda : self.__perform_if_active(cb_calibrate)
        
    # Only allows a command to activate if the voice module is on
    def __perform_if_active(self, action : Callable):
        if self.isActive:
            return action()
        return lambda : None # no-op (no operation)

    # Method to be used on a separate thread for constant audio input
    def __record_audio(self):

        # This function is called each time a block segment is finished recording (DURATION)
        def callback(indata : np.ndarray, frames, time, status):
            # Turn audio into 1D array, then queue for reading
            self.audio_queue.put(indata.flatten())

        if self.debugMode:
            print('Listening for speech 🗣')

        # duration * sample_rate is number of snapshots total over the duration
        # record in float32 for the model
        # channels = 1 is just mono audio
        with sd.InputStream(
                callback=callback, 
                channels=1, 
                samplerate=self.SAMPLE_RATE,
                blocksize=(self.SAMPLE_RATE * self. DURATION),
                dtype='float32'
            ):
            
            # Keeps the thread alive
            while True:
                time.sleep(0.0)

    # Reads audio sent from recording thread and transcribes it
    def __transcribe_audio(self):

        while True:
            # Grab from the queue if something in there
            if not self.audio_queue.empty():
                audio_chunk = self.audio_queue.get()

                # Removes non-alphanumeric characters
                transcription = self.app.transcribe(audio_chunk, self.SAMPLE_RATE)
                clean_text = transcription.translate(self.translator).lower()

                print(clean_text)

                # Check commands; Only one command can activate at a time
                for key, command in self.commands.items():
                    if key in clean_text:
                        command()
                        break

            # Keeps thread alive with 10ms delay for performance
            time.sleep(0.01)

    # Starts all the threads
    def start(self):

        if self.debugMode:
            print('STC booting up... 🔴')

        record_thread = threading.Thread(target=self.__record_audio, daemon=True)
        transcribe_thread = threading.Thread(target=self.__transcribe_audio, daemon=True)

        record_thread.start()
        transcribe_thread.start()

        if self.debugMode:
            print('STC online 🟢.')