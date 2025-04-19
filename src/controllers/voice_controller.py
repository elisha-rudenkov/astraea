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

import command_maker as cm

'''
Make a button that opens an overlay
The overlay asks 'How many words for your phrase?'
    User says 'number __'
    'Confirm'
Phrase?
    Gets the (#) of words in the beginning of a phrase
    'Confirm'
Needs activation?
    'Yes' or 'No'
    'Confirm'
Type?
    'shortcut' or 'macro'
    'Confirm'
Keys?
    'control' 'Confirm' etc.
    'Done'

'''



class SpeechToCommand:
    SAMPLE_RATE = 16000                         # Whisper model works off 16kHz
    DURATION = 3                                # Length of chunks fed into model
    MODEL_CLS : type[Whisper] = WhisperTinyEn   # Model classifiation

    isActive : bool = False                     # Determines whether commands are executed or not
    isMakingCommand : bool = False              # Determines whether a command making process is going
    isMousePaused : bool = False                # Determines if mouse is paused by voice commands

    def __init__(self, cb_calibrate : Callable = None, debugMode : bool = False):
        # Queue of input words to be read
        self.audio_queue = queue.Queue()

        # Used to remove non-alphanumeric characters in transcription
        self.translator = str.maketrans('', '', string.punctuation)

        # Instance of the transcribing app
        self.app = WhisperApp(SpeechToCommand.MODEL_CLS.from_pretrained())

        # Instance of the command maker
        self.commandMaker = cm.CommandMaker()

        # Turns on/off the print debug
        self.debugMode = debugMode

        # Dictionary for all commands
        self.commands = {}

        # Default commands - Always in Astrea
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

        self.commands['command'] = lambda : self.__speechMakeCommand()

        '''
        Commands are stored in JSON with the format:

        <word/phrase> :
            "needsActivation": true or false
            "type": shortcut or macro
            "keys": list of pyautogui keys
        
        word/phrase: what needs to be said to be activated
        needsActivation: whether or not 'start listening' needs to be said first
        keys: the keystrokes of the command 

        '''

        # Load additional commands - User's local commands
        with open('src\\commands.json', 'r') as f:
            customCommands : dict = json.load(f)

            for newCommand, info in customCommands.items():
                self.__loadInCommand(newCommand, info)

    
    # Returns if the provided JSON written command info is formatted correctly
    def __isCommandFormatted(self, info : dict) -> bool:
       
       # Format for the JSON commands
        schema = {
            "type" : "object",
            "properties" : {
                "needsActivation": {"type" : "boolean"},
                "type" : {"type" : "string"},
                "keys" : {"type" : "array", "items" : {"type" : "string"}}
            },
            "required": ["needsActivation", "type", "keys"],
            "additionalProperties" : False
        }

        # Check if information matches correctly
        try:
            validate(instance=info, schema=schema)
        except ValidationError:
            return False
        
        # Passed all tests
        return True
        
    # Only allows a command to activate if the voice module is on
    def __perform_if_active(self, action : Callable):
        if self.isActive:
            return action()
        return lambda : None # no-op (no operation)

    def __loadInCommand(self, newCommand : str, info : dict):
        # Check if there's no command under that phrase
        if newCommand not in self.commands:

            # Do not attempt to read the command info if it's incomplete
            if not self.__isCommandFormatted(info):
                return

            # Get command information
            needsActivation : bool = info['needsActivation']
            inputType : str = info['type']
            inputKeys : list[str] = info['keys']


            # Loads the command correctly based on information
            commandMap = {
                (True, 'shortcut') : lambda k=inputKeys : self.__perform_if_active(lambda: pyautogui.hotkey(k)),
                (True, 'macro') : lambda k=inputKeys : self.__perform_if_active(lambda: pyautogui.press(k)),
                (False, 'shortcut') : lambda k=inputKeys: pyautogui.hotkey(k),
                (False, 'macro'): lambda k=inputKeys : pyautogui.press(k)
            }

            formattedCommand = commandMap.get((needsActivation, inputType), None)
            
            # Skip command if anything if can't match with a command format
            if formattedCommand is None:
                return

            self.commands[newCommand] = formattedCommand

        else:
            # Skip commands that already have the same name
            return


    # TODO: Remove this once GUI is ready. Used for function testing.
    def __speechMakeCommand(self):
        self.isMakingCommand = True
        print('NOTE: GUI isnt ready if youre seeing this')



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

                # Only activate commands if we're not making one
                if not self.isMakingCommand:
                    self.__checkForCommandUsage(clean_text)
                else:
                    self.__checkCommandCreation(clean_text)
  

            # Keeps thread alive with 10ms delay for performance
            time.sleep(0.01)

    # Check commands; Only one command can activate at a time
    def __checkForCommandUsage(self, clean_text : str):
        print(clean_text)

        for phrase, command in self.commands.items():
            if phrase in clean_text:
                command()
                break   

    # Walks through the command process
    def __checkCommandCreation(self, clean_text : str):
        # Adds the command that was created once done
        newCommand = self.commandMaker.makerHandler(clean_text)
        if newCommand is not None:
            cleanCommand = next(iter(newCommand.items()))
            self.__loadInCommand(cleanCommand[0], cleanCommand[1])

            with open('src\\commands.json', 'r+') as f:
                commands : dict
                commands = json.load(f)
                commands.update(newCommand)

                f.seek(0)
                json.dump(commands, f, indent=4, separators=(', ', ': '))
                f.truncate()

            self.isMakingCommand = False



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


def main():
    stc = SpeechToCommand()
    stc.start()

    while True:
        time.sleep(0.0)

if __name__ == "__main__":
    main()