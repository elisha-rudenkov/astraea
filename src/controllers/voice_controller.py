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


from collections.abc import Callable    # Function type hinting
import json                             # Loading in custom commands
from jsonschema import validate, ValidationError

# TODO: Validate incoming keys and make PR.

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

        # Load the commands in
        self.__loadDefaultCommands(cb_calibrate)
        self.__loadAdditionalJSONCommands()

    # Returns a dict of the function and description together for storage
    def __formatCommand(self, func : Callable, desc : str) -> dict:
        return { "action" : func, "desc" : desc }

    # Returns if the provided JSON written command info is formatted correctly
    def __isCommandFormatted(self, info : dict) -> bool:
       
       # Format for the JSON commands
        schema = {
            "type" : "object",
            "properties" : {
                "needsActivation": {"type" : "boolean"},
                "type" : {"type" : "string"},
                "keys" : {"type" : "array", "items" : {"type" : "string"}},
                "desc" : {"type" : "string"}
            },
            "required": ["needsActivation", "type", "keys", "desc"],
            "additionalProperties" : False
        }

        # Check if information matches correctly
        try:
            validate(instance=info, schema=schema)
        except ValidationError:
            return False
        
        # Passed all tests
        return True
    
    def __loadDefaultCommands(self, cb_calibrate : Callable):
        # Default commands - Always in Astrea
        command_definitions = {
            'left': (
                lambda: self.__perform_if_active(lambda: pyautogui.click(button='left')),
                "mouse left click"
            ),
            'right': (
                lambda: self.__perform_if_active(lambda: pyautogui.click(button='right')),
                "mouse right click"
            ),
            'hold': (
                lambda: self.__perform_if_active(pyautogui.mouseDown),
                "mouse hold down left click"
            ),
            'release': (
                lambda: self.__perform_if_active(pyautogui.mouseUp),
                "mouse release left click"
            ),
            'pause mouse': (
                lambda: self.__perform_if_active(lambda: setattr(self, 'isMousePaused', True)),
                "pause head-mouse movement"
            ),
            'resume mouse': (
                lambda: self.__perform_if_active(lambda: setattr(self, 'isMousePaused', False)),
                "resume head-mouse movement"
            ),
            'start listening': (
                lambda: (print('Commands enabled ✔️'), setattr(self, 'isActive', True)),
                "activate voice commands"
            ),
            'stop listening': (
                lambda: (print('Commands disabled ❌'), setattr(self, 'isActive', False)),
                "deactivate voice commands"
            ),
            'calibrate': (
                lambda: self.__perform_if_active(cb_calibrate),
                "calibrate/recalibrate head position for mouse movement"
            )
        }

        # Adds the default commands to the dictionary
        for name, (action, description) in command_definitions.items():
            self.commands[name] = self.__formatCommand(action, description)

    def __loadAdditionalJSONCommands(self):
        '''
        Commands are stored in JSON with the format:

        <word/phrase> :
            "needsActivation": true or false
            "type": shortcut or macro
            "keys": list of pyautogui keys
            "desc"
        
        word/phrase: what needs to be said to be activated
        needsActivation: whether or not 'start listening' needs to be said first
        keys: the keystrokes of the command 
        '''

        # Load additional commands - User's local commands
        with open('src\\commands.json', 'r') as f:
            customCommands : dict = json.load(f)

            for newCommand, info in customCommands.items():
                
                # Check if there's no command under that phrase
                if newCommand not in self.commands:

                    # Do not attempt to read the command info if it's incomplete
                    if not self.__isCommandFormatted(info):
                        continue

                    # Get command information
                    needsActivation : bool = info['needsActivation']
                    inputType : str = info['type']
                    inputKeys : list[str] = info['keys']
                    desc : str = info['desc']

                    

                    # Loads the command correctly based on information
                    commandMap = {
                        (True, 'shortcut') : lambda k=inputKeys : self.__perform_if_active(lambda: pyautogui.hotkey(k)),
                        (True, 'macro') : lambda k=inputKeys : self.__perform_if_active(lambda: pyautogui.press(k)),
                        (False, 'shortcut') : lambda k=inputKeys: pyautogui.hotkey(k),
                        (False, 'macro'): lambda k=inputKeys : pyautogui.press(k)
                    }

                    commandFunc = commandMap.get((needsActivation, inputType), None)
                    
                    # Skip command if anything if can't match with a command format
                    if commandFunc is None:
                        continue

                    self.commands[newCommand] = self.__formatCommand(commandFunc, desc)

                else:
                    # Skip commands that already have the same name
                    continue

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
                for phrase, command_info in self.commands.items():
                    if phrase in clean_text:
                        command_info['action']()
                        break

            # Keeps thread alive with 10ms delay for performance
            time.sleep(0.01)

    # Returns a dictionary of all the commands in astrea
    def getVoiceCommands(self) -> dict:
        
        # Get the phrase to description only
        stripped_commands = {
            phrase : {key: value for key, value in details if key == 'desc'}
            for phrase, details in self.commands.items()
        }

        return stripped_commands

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