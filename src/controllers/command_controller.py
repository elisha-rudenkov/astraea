from enum import Enum, auto
from collections.abc import Callable    # Function type hinting

control_map = {
    "confirm": "confirm",
    "cancel": "cancel",
    "done": "done",
}

symbol_key_map = {
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

    # Function keys
    **{f"f {i}": f"f{i}" for i in range(1, 25)},

    # Letters - Some keys need to be written out phonetically
    **{f"letter {char}": char for char in "abcdefghijklmnopqrstuvwxyz"},
    "let her see" : "c",
    "let her are" : "r",
    "let her you" : "u"
}

number_map = {
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
}

full_phrase_map = {
    **control_map,
    **symbol_key_map,
    **number_map
}


class CommandMakerState(Enum):
    IDLE = auto()
    ASK_WORD_COUNT = auto()
    ASK_PHRASE = auto()
    ASK_ACTIVATION = auto()
    ASK_TYPE = auto()
    KEY_INPUT_LOOP = auto()
    DONE = auto()
CMSTATE = CommandMakerState

class CommandMaker:

    def __init__(self, command_questions : Callable, command_answers : Callable):
        
        self.command_questions = command_questions
        self.command_answers = command_answers
        
        self.state : CommandMakerState = CommandMakerState.IDLE
        
        self.phrase : str = None
        self.phraseLength : int = None

        self.needsActivation : bool = False
        self.type : str = None

        self.keys : list[str] = []
        self.keyLoader : str = None

        self.finalized_command : dict = None

    def __switchToState(self, to : CommandMakerState):
        self.state = to

        match to:
            case CMSTATE.ASK_WORD_COUNT:
                self.command_questions("How many words in your command?" \
                "\n\nSay 'number' followed a digit (1-9)." \
                "\n\ne.g. 'number 2', 'number 3'")
            case CMSTATE.ASK_PHRASE:
                self.command_questions("What will your activation phrase be?" \
                "\n\nSay 'say' followed by your phrase." \
                "\n\ne.g., 'say select all', 'say copy'")
            case CMSTATE.ASK_ACTIVATION:
                self.command_questions("Does this command require Astrea to be listening?" \
                "\n\nSay 'yes' or 'no'" \
                "\n\nIf you say 'yes', Astrea needs to hear 'start listening' before the command is used")
            case CMSTATE.ASK_TYPE:
                self.command_questions("Is this a shortcut or macro?" \
                "\n\nSay 'shortcut' or 'macro'" \
                "\n\nShortcuts are triggered together, macros are executed sequentially")
            case CMSTATE.KEY_INPUT_LOOP:
                self.command_questions("Say the name of the key to add" \
                "\n\ne.g. 'enter', 'control', 'letter a', 'number 9'")
            case CMSTATE.DONE:
                self.command_questions("⚙️ Finalizing...")
    
    def __clearCommandMaker(self):
        self.phrase = None
        self.phraseLength = None

        self.needsActivation = False
        self.type = None

        self.keys = []
        self.keyLoader = None
        self.finalized_command = None

    def makerHandler(self, clean_text : str) -> dict:

        if self.state == CMSTATE.IDLE:
            self.__switchToState(CMSTATE.ASK_WORD_COUNT)

        if self.state == CMSTATE.ASK_WORD_COUNT:

            for phrase, key in number_map.items():
                if phrase in clean_text:
                    self.phraseLength = int(key)

                    if 1 <= self.phraseLength <= 9:
                        self.command_answers('📝 Word count set to ' + key)
                        self.__switchToState(CMSTATE.ASK_PHRASE)
                        break
                    
        elif self.state == CMSTATE.ASK_PHRASE:
            
            # 'Say' is the delimiter
            if 'say' in clean_text:
                word_list = clean_text.split(' ')
                index = word_list.index('say')

                pre_phrase = word_list[index + 1: index + 1 + self.phraseLength]
                self.phrase = ' '.join(pre_phrase)
                self.command_questions("🗣️  You said: " + self.phrase +
                                       "\n\nSay 'confirm' to continue"
                                       "\n\nSay 'say' followed by your phrase to edit")
            
            # Check for confirmation if a phrase was set
            if self.phrase is not None:
                if 'confirm' in clean_text:
                    self.command_answers('✅ Phrase confirmed: ' + self.phrase)
                    self.__switchToState(CMSTATE.ASK_ACTIVATION)
        
        elif self.state == CMSTATE.ASK_ACTIVATION:

            if 'yes' in clean_text:
                self.needsActivation = True
                self.command_answers('🟢 Activation Needed')
                self.__switchToState(CMSTATE.ASK_TYPE)
            elif 'no' in clean_text:
                self.needsActivation = False
                self.command_answers('🔴 Activation Not Needed')
                self.__switchToState(CMSTATE.ASK_TYPE)
        
        elif self.state == CMSTATE.ASK_TYPE:

            if 'shortcut' in clean_text:
                self.type = 'shortcut'
                self.command_answers('🔑 Shortcut Selected')
                self.__switchToState(CMSTATE.KEY_INPUT_LOOP)
            elif 'short cut' in clean_text:
                self.type = 'shortcut'
                self.command_answers('🔑 Shortcut Selected')
                self.__switchToState(CMSTATE.KEY_INPUT_LOOP)
            elif 'macro' in clean_text:
                self.type = 'macro'
                self.command_answers('⌨️ Macro Selected')
                self.__switchToState(CMSTATE.KEY_INPUT_LOOP)

        elif self.state == CMSTATE.KEY_INPUT_LOOP:

            for phrase_in_map, key in full_phrase_map.items():
                if phrase_in_map in clean_text:
                    
                    if phrase_in_map == 'confirm':

                        if self.keyLoader is not None:
                            self.keys.append(self.keyLoader)
                            self.command_answers('➕ ' + self.keyLoader)
                            self.command_questions("Say 'done' if you are finished or the name of a different key" \
                                                   "\n\ne.g. 'enter', 'control', 'letter a', 'number 9'")
                            self.keyLoader = None
                        else:
                            self.command_questions('✖️ No key specified' \
                            '\n\nSay the name of the key to add' \
                            "\n\ne.g. 'enter', 'control', 'letter a', 'number 9'")

                    elif phrase_in_map == 'done':

                        new_command = {
                            self.phrase : {
                                "needsActivation": self.needsActivation,
                                "type": self.type,
                                'keys' : self.keys
                            }
                        }

                        self.finalized_command = new_command
                        
                        self.__switchToState(CMSTATE.DONE)
                    
                    else:
                        self.keyLoader = key
                        self.command_questions('🗣️ You said: ' + key + \
                                               "\n\nSay 'confirm' to add or the name of a different key"
                                               "\n\nSay 'done' if you are finished")
                    
                    break

        elif self.state == CMSTATE.DONE:
            self.state = CMSTATE.IDLE
            returned = self.finalized_command.copy()
            self.__clearCommandMaker()
            return returned

        return None