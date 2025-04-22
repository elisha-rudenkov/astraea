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
    
    logger.info("Initializing application...")
    face_detector = FaceDetector(use_gpu=use_gpu)
    landmark_analyzer = FaceLandmarkAnalyzer(use_gpu=use_gpu)
    mouse_controller = MouseController()
    voice_controller = SpeechToCommand()
    voice_controller.start()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open camera")
    logger.info("Camera initialized")

    # Initialize state variables
    last_analysis_result = None
    calibrated = False
    calibration_text = "Press 'C' when ready to calibrate"
    cam_error_shown = False

    # Initialize UI
    app = QApplication(sys.argv)
    window = MainWindow()

    window.register_mouse_controller(mouse_controller)
        
    def calibrate():
        nonlocal calibrated, calibration_text
        if last_analysis_result is not None:
            mouse_controller.calibrate(last_analysis_result)
            logger.info("Calibration complete!")
            window.update_calibration_status(True, last_analysis_result)
            calibrated = True
            calibration_text = "Calibration complete!"
        else:
            logger.warning("No analysis data available for calibration.")

    # Register calibration function to button
    window.register_calibrate_callback(calibrate)

    def update_frame():
        nonlocal last_analysis_result, calibration_text, cam_error_shown

        if not cap.isOpened():
            if not cam_error_shown:
                window.no_cam_err_msg()
                cam_error_shown = True
            return
        else:
            if cam_error_shown:
                cam_error_shown = False

            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to grab frame")
                return
            
            # Check if spacebar is pressed using the keyboard module
            mouse_controller.movement_paused = keyboard.is_pressed('space')
            
            # Detect faces in the frame
            detections = face_detector.detect_faces(frame)
            
            if detections:
                detection = max(detections, key=lambda x: x[4])
                face_roi = extract_face_roi(frame, detection)
                
                if face_roi.size > 0:
                    analysis_result = landmark_analyzer.analyze_face(face_roi)
                    
                    if analysis_result is not None:
                        # Store latest analysis result
                        last_analysis_result = analysis_result

                        # Draw rectangle around face
                        x, y, w, h = detection[:4]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
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
                            cv2.putText(frame, calibration_text, (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            cv2.putText(frame, f"Pitch: {pitch:>6.1f}deg", (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.putText(frame, f"Yaw: {yaw:>6.1f}deg", (10, 80),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.putText(frame, f"Roll: {roll:>6.1f}deg", (10, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            # Display pause status
                            if mouse_controller.movement_paused:
                                cv2.putText(frame, "MOVEMENT PAUSED", (10, 130),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        except Exception as e:
                            logger.error(f"Error displaying results: {str(e)}")

            # Convert frame to RGB for Qt
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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