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

class SpeechToCommand:
    SAMPLE_RATE = 16000                         # Whisper model works off 16kHz
    DURATION = 3                                # Length of chunks fed into model
    MODEL_CLS : type[Whisper] = WhisperTinyEn   # Model classifiation

    def __init__(self, text_callback, debugMode : bool = False):
        # Queue of input words to be read
        self.audio_queue = queue.Queue()

        # Used to remove non-alphanumeric characters in transcription
        self.translator = str.maketrans('', '', string.punctuation)

        # Instance of the transcribing app
        self.app = WhisperApp(SpeechToCommand.MODEL_CLS.from_pretrained())

        # Turns on/off the print debug
        self.debugMode = debugMode

        self.text_callback = text_callback

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

        # Determines whether commands are executed or not
        isActive : bool = False 

        while True:
            # Grab from the queue if something in there
            if not self.audio_queue.empty():
                audio_chunk = self.audio_queue.get()

                # Removes non-alphanumeric characters
                transcription = self.app.transcribe(audio_chunk, self.SAMPLE_RATE)
                clean_text = transcription.translate(self.translator).lower()

                self.text_callback('… ' + clean_text)

                # List of commands; Only one command can activate at a time
                # TODO: This is a temporary implement for showcase purposes
                if isActive:
                    if 'right' in clean_text:
                        pyautogui.click(button='right')
                    elif 'left' in clean_text:
                        pyautogui.click(button='left')
                    elif 'hold' in clean_text:
                        pyautogui.mouseDown()
                    elif 'release' in clean_text:
                        pyautogui.mouseUp()
                    elif 'stop listening' in clean_text:
                        print('Commands disabled ❌')
                        isActive = False

                else:
                    if 'start listening' in clean_text:
                        print('Commands enabled ✔️')
                        isActive = True

                
            # Keeps thread alive
            time.sleep(0.0)

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


import sys
from PyQt6.QtCore import Qt, QTimer, QRect
from PyQt6.QtGui import QPainter, QColor, QPen
from PyQt6.QtWidgets import QApplication, QWidget
from collections import deque

class TextDisplayWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple Text Display")
        self.setGeometry(100, 100, 400, 300)  # Set the window size
        self.text_lines = deque(maxlen=10)
        
        # Set window flags
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint | # Always on top of other window
            Qt.WindowType.FramelessWindowHint |  # Removes window frame
            Qt.WindowType.Tool |                 # Window not shown on taskbar
            Qt.WindowType.BypassWindowManagerHint # Don't let window manager handle this window
        )

        # Set background as transparent
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Make the cursor able to click through the overlay
        self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, True)

        self.setGeometry(0, 50, 400, 80)
        desktop = QApplication.primaryScreen().geometry()
        self.move(desktop.width() - self.width() - 20, 20)

    def update_text(self, new_text):
        """This function updates the text to display a different message."""
        self.text_lines.append(new_text)
        self.update()  # Call the update method to trigger a repaint

    def paintEvent(self, event):
        """Override paintEvent to draw the text."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw semi-transparent background
        painter.setBrush(QColor(40, 40, 40, 180))  # Dark gray with alpha
        painter.setPen(QPen(QColor(60, 60, 220), 2))  # Blue border
        painter.drawRoundedRect(0, 0, self.width() - 1, self.height() - 1, 10, 10)

        # Draw the updated text lines, starting from the bottom of the widget
        painter.setPen(QColor(255, 255, 255))  # White text
        font = painter.font()
        font.setPointSize(12)
        painter.setFont(font)

        # Draw the text at the center of the widget
        y_offset = self.height() - 20  # Start at the bottom of the widget
        for text in reversed(self.text_lines):  # Draw from the most recent to the oldest
            painter.drawText(10, y_offset, text)
            y_offset -= 20  # Adjust vertical spacing between lines


app = QApplication(sys.argv)
tdw = TextDisplayWidget()
tdw.show()

stc = SpeechToCommand(tdw.update_text)
stc.start()

sys.exit(app.exec())