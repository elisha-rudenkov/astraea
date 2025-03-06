import cv2
import logging
from src.detectors.face_detector import FaceDetector, extract_face_roi
from src.analyzers.landmark_analyzer import FaceLandmarkAnalyzer
from src.controllers.mouse_controller import MouseController

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing application...")
    face_detector = FaceDetector()
    landmark_analyzer = FaceLandmarkAnalyzer()
    mouse_controller = MouseController()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open camera")
        return
    logger.info("Camera initialized")
    
    calibration_mode = True
    calibration_text = "Press 'C' when ready to calibrate"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to grab frame")
            break
            
        detections = face_detector.detect_faces(frame)
        
        if detections:
            detection = max(detections, key=lambda x: x[4])
            face_roi = extract_face_roi(frame, detection)
            
            if face_roi.size > 0:
                analysis_result = landmark_analyzer.analyze_face(face_roi)
                
                if analysis_result is not None:
                    x, y, w, h = detection[:4]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if calibration_mode:
                        if key == ord('c'):
                            mouse_controller.calibrate(analysis_result)
                            calibration_mode = False
                            calibration_text = "Calibration complete!"
                    else:
                        mouse_controller.update(analysis_result)
                    
                    try:
                        pitch = analysis_result['pitch']
                        yaw = analysis_result['yaw']
                        roll = analysis_result['roll']
                        
                        cv2.putText(frame, calibration_text, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"Pitch: {pitch:>6.1f}deg", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(frame, f"Yaw: {yaw:>6.1f}deg", (10, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(frame, f"Roll: {roll:>6.1f}deg", (10, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    except Exception as e:
                        logger.error(f"Error displaying results: {str(e)}")
        
        cv2.imshow('Face Analysis', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 