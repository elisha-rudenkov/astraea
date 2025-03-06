import cv2
import numpy as np
import onnxruntime
import logging
import pyautogui
import time
from collections import deque

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Disable pyautogui's fail-safe
pyautogui.FAILSAFE = False

def angle_difference(new_angle, calibration_angle):
    """
    Computes the minimal difference between two angles (in degrees)
    while properly handling the wrap-around at ±180deg.
    Returns a value in the range [-180, 180].
    """
    diff = new_angle - calibration_angle
    diff = (diff + 180) % 360 - 180
    return diff

class FaceDetector:
    def __init__(self, model_path='face_det_lite.onnx'):
        logger.info(f"Initializing FaceDetector with model: {model_path}")
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        logger.debug(f"Face detector input shape: {self.input_shape}")

    def preprocess_image(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (640, 480))
        img = img.astype('float32') / 255.0
        img = (img - 0.442) / 0.280
        img = np.expand_dims(np.expand_dims(img, axis=0), axis=0)
        return img.astype(np.float32)

    def detect_faces(self, img, conf_threshold=0.55):
        input_tensor = self.preprocess_image(img)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        heatmap, bbox_reg, landmarks = outputs
        heatmap = 1 / (1 + np.exp(-heatmap))
        
        detections = []
        confidence_mask = heatmap[0, 0] > conf_threshold
        y_indices, x_indices = np.where(confidence_mask)
        
        for y, x in zip(y_indices, x_indices):
            confidence = heatmap[0, 0, y, x]
            dx1, dy1, dx2, dy2 = bbox_reg[0, :, y, x]
            
            stride = 8
            x1 = (x - dx1) * stride
            y1 = (y - dy1) * stride
            x2 = (x + dx2) * stride
            y2 = (y + dy2) * stride
            
            w = x2 - x1
            h = y2 - y1
            
            x1 = x1 - w * 0.05
            y1 = y1 - h * 0.05
            w = w * 1.1
            h = h * 1.1
            
            x1 = max(0, min(x1, 640))
            y1 = max(0, min(y1, 480))
            w = min(w, 640 - x1)
            h = min(h, 480 - y1)
            
            detections.append([int(x1), int(y1), int(w), int(h), float(confidence)])
        
        return detections

class FaceLandmarkAnalyzer:
    def __init__(self, model_path='mediapipe_face-mediapipefacelandmarkdetector.onnx'):
        logger.info(f"Initializing FaceLandmarkAnalyzer with model: {model_path}")
        self.session = onnxruntime.InferenceSession(model_path)
        
        self.input_details = self.session.get_inputs()
        self.output_details = self.session.get_outputs()
        self.input_name = self.input_details[0].name

    def preprocess_face(self, face_img):
        face_img = cv2.resize(face_img, (192, 192))
        face_img = face_img.astype('float32') / 255.0
        face_img = np.transpose(face_img, (2, 0, 1))
        face_img = np.expand_dims(face_img, axis=0)
        return face_img

    def analyze_face(self, face_img):
        try:
            processed_img = self.preprocess_face(face_img)
            outputs = self.session.run(None, {self.input_name: processed_img})
            score, landmarks = outputs
            
            # Extract key landmarks.
            # (Indices may vary based on the model. Here we assume:
            #  nose_tip at index 1, left_eye at index 33, right_eye at index 263)
            nose_tip = landmarks[0, 1, :]
            left_eye = landmarks[0, 33, :]
            right_eye = landmarks[0, 263, :]
            
            # Use the midpoint of the eyes as a stable center.
            eye_center = (left_eye + right_eye) / 2.0
            
            # Calculate head pose from landmarks using the eye center as reference.
            pitch = np.arctan2(nose_tip[1] - eye_center[1], nose_tip[2] - eye_center[2])
            yaw   = np.arctan2(nose_tip[0] - eye_center[0], nose_tip[2] - eye_center[2])
            roll  = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
            
            # Convert to degrees.
            pitch = np.degrees(pitch)
            yaw   = np.degrees(yaw)
            roll  = np.degrees(roll)
            
            # Removed additional smoothing here to avoid conflicting with MouseController smoothing.
            
            return {
                'score': score[0],
                'pitch': pitch,
                'yaw': yaw,
                'roll': roll,
                'landmarks': landmarks[0]
            }
            
        except Exception as e:
            logger.error(f"Error in analyze_face: {str(e)}")
            return None

def extract_face_roi(frame, detection):
    x, y, w, h = detection[:4]
    return frame[y:y+h, x:x+w]

class MouseController:
    def __init__(self):
        self.calibration = None
        self.movement_threshold = 8.0    # Threshold for movement detection
        self.click_threshold = 20.0      # Threshold for detecting click gestures
        self.click_cooldown = 0.5
        self.last_click_time = 0
        
        # Movement smoothing
        self.smooth_window = 5
        self.yaw_history = deque(maxlen=self.smooth_window)
        self.pitch_history = deque(maxlen=self.smooth_window)
        
        # Screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Speed parameters
        self.base_speed = 2             # Base speed
        self.max_speed = 20             # Maximum speed
        self.exp_factor = 2.0           # Exponential acceleration factor

    def calibrate(self, angles):
        self.calibration = {
            'pitch': angles['pitch'],
            'yaw': angles['yaw'],
            'roll': angles['roll']
        }
        # Clear history on calibration.
        self.yaw_history.clear()
        self.pitch_history.clear()
        logger.info(f"Calibrated at: Pitch={angles['pitch']:.1f}, Yaw={angles['yaw']:.1f}, Roll={angles['roll']:.1f}")

    def get_movement_speed(self, angle_diff):
        """Calculate exponential movement speed based on angle difference."""
        if abs(angle_diff) < self.movement_threshold:
            return 0
        
        # Normalize the angle difference.
        normalized_diff = (abs(angle_diff) - self.movement_threshold) / 45.0
        normalized_diff = min(max(normalized_diff, 0), 1)
        
        # Calculate exponential speed.
        speed = self.base_speed + (self.max_speed - self.base_speed) * (normalized_diff ** self.exp_factor)
        return speed * np.sign(angle_diff)

    def update(self, angles):
        if self.calibration is None:
            return

        # Use the angle_difference helper to account for wrap-around.
        yaw_diff = angle_difference(angles['yaw'], self.calibration['yaw'])
        pitch_diff = angle_difference(angles['pitch'], self.calibration['pitch'])
        
        # Add to history for smoothing.
        self.yaw_history.append(yaw_diff)
        self.pitch_history.append(pitch_diff)
        
        # Only proceed if we have enough history.
        if len(self.yaw_history) == self.smooth_window:
            # Calculate smoothed differences.
            yaw_diff_smooth = np.mean(list(self.yaw_history))
            pitch_diff_smooth = np.mean(list(self.pitch_history))
            
            # Calculate movement speeds.
            x_speed = self.get_movement_speed(yaw_diff_smooth)
            # Invert pitch to correct the up/down inversion.
            y_speed = self.get_movement_speed(-pitch_diff_smooth)
            
            # Move mouse if speed is non-zero.
            if x_speed != 0 or y_speed != 0:
                current_x, current_y = pyautogui.position()
                new_x = min(max(current_x + x_speed, 0), self.screen_width)
                new_y = min(max(current_y + y_speed, 0), self.screen_height)
                pyautogui.moveTo(new_x, new_y)
        
        # Handle clicks based on roll difference.
        roll_diff = angle_difference(angles['roll'], self.calibration['roll'])
        current_time = time.time()
        
        if current_time - self.last_click_time > self.click_cooldown:
            if roll_diff > self.click_threshold:
                pyautogui.click(button='right')
                self.last_click_time = current_time
            elif roll_diff < -self.click_threshold:
                pyautogui.click(button='left')
                self.last_click_time = current_time

def main():
    logger.info("Initializing application...")
    face_detector = FaceDetector()
    landmark_analyzer = FaceLandmarkAnalyzer()
    mouse_controller = MouseController()
    
    cap = cv2.VideoCapture(0)
    logger.info("Camera initialized")
    
    calibration_mode = True
    calibration_text = "Press 'C' when ready to calibrate"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to grab frame")
            break
            
        # Detect faces.
        detections = face_detector.detect_faces(frame)
        
        if detections:
            # Get the first face (highest confidence).
            detection = max(detections, key=lambda x: x[4])
            
            # Extract face ROI.
            face_roi = extract_face_roi(frame, detection)
            
            if face_roi.size > 0:
                # Analyze face.
                analysis_result = landmark_analyzer.analyze_face(face_roi)
                
                if analysis_result is not None:
                    # Draw the face detection rectangle.
                    x, y, w, h = detection[:4]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Handle calibration and mouse control.
                    key = cv2.waitKey(1) & 0xFF
                    if calibration_mode:
                        if key == ord('c'):
                            mouse_controller.calibrate(analysis_result)
                            calibration_mode = False
                            calibration_text = "Calibration complete!"
                    else:
                        mouse_controller.update(analysis_result)
                    
                    # Display results on the frame.
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
