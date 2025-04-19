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

phrase_to_key = {
    # Punctuation and symbols
    "tab": "\t",
    "new line": "\n",
    "return": "\r",
    "space": " ",
    "exclamation mark": "!",
    "double quote": '"',
    "hash": "#",
    "dollar": "$",
    "percent": "%",
    "and sign": "&",
    "single quote": "'",
    "left parenthesis": "(",
    "right parenthesis": ")",
    "asterisk": "*",
    "plus": "+",
    "comma": ",",
    "dash": "-",
    "period": ".",
    "slash": "/",
    "backslash": "\\",
    "colon": ":",
    "semicolon": ";",
    "less than": "<",
    "equal": "=",
    "greater than": ">",
    "question mark": "?",
    "at sign": "@",
    "left bracket": "[",
    "right bracket": "]",
    "hat": "^",
    "underscore": "_",
    "back tick": "`",
    "grave": "`",
    "left brace": "{",
    "pipe": "|",
    "right brace": "}",
    "tilda": "~",
    
    # Arrow keys
    "up arrow": "up",
    "down arrow": "down",
    "left arrow": "left",
    "right arrow": "right",

    # Other keys
    "enter": "enter",
    "escape": "esc",
    "backspace": "backspace",
    "delete": "delete",
    "insert": "insert",
    "home": "home",
    "end": "end",
    "page up": "pageup",
    "page down": "pagedown",
    "control": "ctrl",
    "control left": "ctrlleft",
    "control right": "ctrlright",
    "alt": "alt",
    "alternative": "alt",
    "alt left": "altleft",
    "alternative left": "altleft",
    "alt right": "altright",
    "alternative right": "altright",
    "shift": "shift",
    "shift left": "shiftleft",
    "shift right": "shiftright",
    "caps lock": "capslock",
    "print screen": "printscreen",
    "printscreen": "printscreen",
    "volume up": "volumeup",
    "volume down": "volumedown",
    "mute": "volumemute",
    "play pause": "playpause",
    "next track": "nexttrack",
    "previous track": "prevtrack",
    "stop": "stop",
    "command": "command",
    "option": "option",
    "option left": "optionleft",
    "option right": "optionright",
    "windows": "win",
    "windows left": "winleft",
    "windows right": "winright",

    "confirm": "confirm",
    "cancel": "cancel",
    "done": "done",

    # Function keys
    **{f"f {i}": f"f{i}" for i in range(1, 25)},

    # Numbers
    "number zero": "0",
    "number one": "1",
    "number two": "2",
    "number to": "2",   # misrecognition-safe
    "number too": "2",
    "number three": "3",
    "number four": "4",
    "number for": "4",
    "number five": "5",
    "number six": "6",
    "number seven": "7",
    "number eight": "8",
    "number ate": "8",
    "number nine": "9",
    **{f"number {num}": num for num in "0123456789"},

    # Letters - Some keys need to be written out phonetically
    **{f"letter {char}": char for char in "abcdefghijklmnopqrstuvwxyz"},
    "let her see" : "c",
    "let her are" : "r",
    "let her you" : "u"
}


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

        # Turns on/off the print debug
        self.debugMode = debugMode

        # Dictionary for all commands
        self.commands = {}
        self.loaded : str = ''
        self.command = []

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
                
                # Check if there's no command under that phrase
                if newCommand not in self.commands:

                    # Do not attempt to read the command info if it's incomplete
                    if not self.__isCommandFormatted(info):
                        continue

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
                        continue

                    self.commands[newCommand] = formattedCommand

                else:
                    # Skip commands that already have the same name
                    continue

    
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

    def __speechMakeCommand(self):
        self.isMakingCommand = True
        print('> Welcome to Making a Command <')
        print('How many words for command?')
        return


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

                if not self.isMakingCommand:
                    print(clean_text)
                    # Check commands; Only one command can activate at a time
                    for phrase, command in self.commands.items():
                        if phrase in clean_text:
                            command()
                            break
                else:
                    
                    for phrase, key in phrase_to_key.items():
                        if phrase in clean_text:

                            if phrase == 'confirm':
                                self.command.append(self.loaded)
                                print(self.loaded, '✔️')
                                self.loaded = None
                            elif phrase == 'done':
                                new_command = self.command
                                print('New Command :)')
                                print(new_command)
                                self.isMakingCommand = False
                                pyautogui.hotkey(new_command)

                                self.command.clear()
                            else:
                                self.loaded = key
                                print('Set?: ', key)
                            
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


def main():
    stc = SpeechToCommand()
    stc.start()

    while True:
        time.sleep(0.0)

if __name__ == "__main__":
    main()