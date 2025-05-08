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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Head-based mouse controller")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration for ONNX models")
    parser.add_argument("--benchmark", action="store_true", help="Run a performance benchmark of all available providers")
    args = parser.parse_args()
    
    # Determine whether to use GPU acceleration
    use_gpu = not args.no_gpu
    
    # Log available providers
    available_providers = get_available_providers()
    logger.info(f"Available ONNX Runtime providers: {available_providers}")
    if use_gpu:
        logger.info("GPU acceleration is enabled (use --no-gpu to disable)")
        if not any(p for p in available_providers if p not in ['CPUExecutionProvider', 'AzureExecutionProvider']):
            logger.warning("No GPU acceleration providers detected! Only CPU will be used.")
            logger.warning("To enable GPU acceleration, install onnxruntime-gpu or follow setup instructions in README.")
    else:
        logger.info("GPU acceleration is disabled")
    
    # If the benchmark flag is set, run performance benchmarks
    if args.benchmark:
        logger.info("Running performance benchmark...")
        # Use the first model for benchmarking
        check_provider_performance("models/face_det_lite.onnx")
        return  # Exit after benchmarking
    
    def calibrate():
        nonlocal calibrated
        if last_analysis_result is not None:
            mouse_controller.calibrate(last_analysis_result)
            logger.info("Calibration complete!")
            window.update_calibration_status(True, last_analysis_result)
            calibrated = True
        else:
            logger.warning("No analysis data available for calibration.")


    logger.info("Initializing application...")

    # Initialize UI
    app = QApplication(sys.argv)
    window = MainWindow()

    face_detector = FaceDetector(use_gpu=use_gpu)
    landmark_analyzer = FaceLandmarkAnalyzer(use_gpu=use_gpu)
    mouse_controller = MouseController()
    voice_controller = SpeechToCommand(
        window.transcription_box.update_text,
        window.command_maker.update_walkthrough,
        window.command_maker.update_answers,
        window.command_maker.show_overlay,
        cb_calibrate=calibrate,
    )
    voice_controller.start()    

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open camera")
        return
    logger.info("Camera initialized")

    # Initialize state variables
    last_analysis_result = None
    calibrated = False

    window.register_mouse_controller(mouse_controller)

    # Register calibration function to button
    window.register_calibrate_callback(calibrate)

    def update_frame():
        nonlocal last_analysis_result

        ret, calculation_frame = cap.read()
        if not ret:
            logger.error("Failed to grab frame")
            return
        
        # Check if spacebar is pressed using the keyboard module
        mouse_controller.movement_paused = keyboard.is_pressed('space') or voice_controller.isMousePaused
        
        # Flip the the display for more intuitive use
        display_frame = calculation_frame
        display_frame = cv2.flip(display_frame, 1)

        # Detect faces in the frame
        detections = face_detector.detect_faces(calculation_frame)
        
        if detections:
            detection = max(detections, key=lambda x: x[4])
            face_roi = extract_face_roi(calculation_frame, detection)
            
            if face_roi.size > 0:
                analysis_result = landmark_analyzer.analyze_face(face_roi)
                
                if analysis_result is not None:
                    # Store latest analysis result
                    last_analysis_result = analysis_result

                    # Draw rectangle around face (flipped for correct coordinates)
                    x, y, w, h = detection[:4]
                    display_frame = cv2.flip(display_frame, 1)
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    display_frame = cv2.flip(display_frame, 1)
                    
                    # Check for calibration key press
                    key = cv2.waitKey(1) & 0xFF  
                    if key == ord('c'):
                        calibrate()

                    # Update mouse position if calibrated
                    if calibrated:
                        mouse_controller.update(analysis_result)
                    try:
                        pitch = analysis_result['pitch']
                        yaw = analysis_result['yaw']
                        roll = analysis_result['roll']
                        
                        # Display information on frame
                        cv2.putText(display_frame, f"Pitch: {pitch:>6.1f}deg", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(display_frame, f"Yaw: {yaw:>6.1f}deg", (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(display_frame, f"Roll: {roll:>6.1f}deg", (10, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Display pause status
                        if mouse_controller.movement_paused:
                            cv2.putText(display_frame, "MOVEMENT PAUSED", (10, 130),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            

                    except Exception as e:
                        logger.error(f"Error displaying results: {str(e)}")

            # Convert frame to RGB for Qt
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            # Update GUI with camera frame
            window.scene_video_label.setPixmap(QPixmap.fromImage(q_img))

    # Set up timer for frame updates
    timer = QTimer()
    timer.timeout.connect(update_frame)
    timer.start(30)  # ~30 fps

    # Start Qt event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 