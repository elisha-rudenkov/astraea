import cv2
import logging
import keyboard  
import argparse 
from src.detectors.face_detector import FaceDetector, extract_face_roi
from src.analyzers.landmark_analyzer import FaceLandmarkAnalyzer
from src.controllers.mouse_controller import MouseController
from src.controllers.voice_controller import SpeechToCommand
from src.utils.onnx_utils import get_available_providers, check_provider_performance  # Import utility functions

from src.ui.settings import MainWindow
import threading
import time
import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QPixmap
import sys

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    stc = SpeechToCommand(
        window.transcription_box.update_text,
        window.command_maker.update_walkthrough,
        window.command_maker.update_answers,
        window.command_maker.show_overlay
    )
    stc.start()

    sys.exit(app.exec())

if __name__ == "__main__":
    main() 