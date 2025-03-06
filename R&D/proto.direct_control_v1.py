import cv2
import numpy as np
import onnxruntime
import logging
import pyautogui
import time
from dataclasses import dataclass
from enum import Enum, auto

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Disable pyautogui's fail-safe
pyautogui.FAILSAFE = False

@dataclass
class CalibrationPoint:
    x: int
    y: int
    yaw: float = None
    pitch: float = None
    
    def is_calibrated(self):
        return self.yaw is not None and self.pitch is not None

class CalibrationState(Enum):
    CENTER = auto()
    TOP_LEFT = auto()
    TOP_RIGHT = auto()
    BOTTOM_LEFT = auto()
    BOTTOM_RIGHT = auto()
    COMPLETED = auto()

class DirectMouseController:
    def __init__(self):
        # Get the actual full screen dimensions
        self.screen_width = pyautogui.size()[0]
        self.screen_height = pyautogui.size()[1]
        
        # Adjust calibration points to use screen edges
        margin = 10  # Small margin from screen edges
        self.calibration_points = {
            CalibrationState.CENTER: CalibrationPoint(
                self.screen_width // 2, 
                self.screen_height // 2
            ),
            CalibrationState.TOP_LEFT: CalibrationPoint(
                margin, 
                margin
            ),
            CalibrationState.TOP_RIGHT: CalibrationPoint(
                self.screen_width - margin, 
                margin
            ),
            CalibrationState.BOTTOM_LEFT: CalibrationPoint(
                margin, 
                self.screen_height - margin
            ),
            CalibrationState.BOTTOM_RIGHT: CalibrationPoint(
                self.screen_width - margin, 
                self.screen_height - margin
            )
        }
        self.current_state = CalibrationState.CENTER
        self.click_threshold = 10.0  # Threshold for detecting click gestures (in degrees)
        self.click_cooldown = 0.5    # Minimum time between clicks (in seconds)
        self.last_click_time = 0
        self.base_roll = None
        
        # Add smoothing parameters
        self.position_history = []
        self.smoothing_window = 5  # Number of frames to average
        self.dead_zone = 0.02  # 2% dead zone around center (adjust as needed)
        
    def get_current_target(self):
        return self.calibration_points[self.current_state]
    
    def calibrate_point(self, angles):
        current_point = self.calibration_points[self.current_state]
        current_point.yaw = angles['yaw']
        current_point.pitch = angles['pitch']
        
        if self.base_roll is None and self.current_state == CalibrationState.CENTER:
            self.base_roll = angles['roll']
        
        # Move to next state
        states = list(CalibrationState)
        current_idx = states.index(self.current_state)
        if current_idx < len(states) - 1:
            self.current_state = states[current_idx + 1]
            
        return self.current_state == CalibrationState.COMPLETED
    
    def is_calibrated(self):
        return self.current_state == CalibrationState.COMPLETED
    
    def interpolate_position(self, angles):
        """
        Interpolate mouse position based on current head angles and calibration points
        """
        if not self.is_calibrated():
            return None
            
        # Get calibration points
        center = self.calibration_points[CalibrationState.CENTER]
        top_left = self.calibration_points[CalibrationState.TOP_LEFT]
        top_right = self.calibration_points[CalibrationState.TOP_RIGHT]
        bottom_left = self.calibration_points[CalibrationState.BOTTOM_LEFT]
        bottom_right = self.calibration_points[CalibrationState.BOTTOM_RIGHT]
        
        # Calculate relative position
        yaw_range = (
            min(top_left.yaw, bottom_left.yaw),
            max(top_right.yaw, bottom_right.yaw)
        )
        pitch_range = (
            min(top_left.pitch, top_right.pitch),
            max(bottom_left.pitch, bottom_right.pitch)
        )
        
        # Normalize current angles to 0-1 range
        x_ratio = (angles['yaw'] - yaw_range[0]) / (yaw_range[1] - yaw_range[0])
        y_ratio = (angles['pitch'] - pitch_range[0]) / (pitch_range[1] - pitch_range[0])
        
        # Clamp values to screen bounds
        x_ratio = max(0, min(1, x_ratio))
        y_ratio = max(0, min(1, y_ratio))
        
        # Add dead zone
        x_ratio = self.apply_dead_zone(x_ratio)
        y_ratio = self.apply_dead_zone(y_ratio)
        
        # Calculate screen position
        x = int(x_ratio * self.screen_width)
        y = int(y_ratio * self.screen_height)
        
        return x, y
    
    def apply_dead_zone(self, value):
        """Apply dead zone to a normalized value"""
        if abs(value - 0.5) < self.dead_zone:
            return 0.5
        return value
    
    def update(self, angles):
        if not self.is_calibrated():
            return
            
        # Update mouse position
        position = self.interpolate_position(angles)
        if position:
            # Add to position history
            self.position_history.append(position)
            if len(self.position_history) > self.smoothing_window:
                self.position_history.pop(0)
            
            # Average the positions
            if self.position_history:
                smoothed_x = int(sum(p[0] for p in self.position_history) / len(self.position_history))
                smoothed_y = int(sum(p[1] for p in self.position_history) / len(self.position_history))
                pyautogui.moveTo(smoothed_x, smoothed_y)
        
        # Handle clicks based on head roll
        if self.base_roll is not None:
            roll_diff = angles['roll'] - self.base_roll
            current_time = time.time()
            
            if current_time - self.last_click_time > self.click_cooldown:
                if roll_diff > self.click_threshold:
                    pyautogui.click(button='right')
                    self.last_click_time = current_time
                    logger.debug(f"Right click triggered (roll_diff: {roll_diff:.1f}deg)")
                elif roll_diff < -self.click_threshold:
                    pyautogui.click(button='left')
                    self.last_click_time = current_time
                    logger.debug(f"Left click triggered (roll_diff: {roll_diff:.1f}deg)")

def draw_calibration_overlay(frame, controller):
    """Draw calibration target and instructions on the frame"""
    h, w = frame.shape[:2]
    
    # Draw calibration instructions
    if not controller.is_calibrated():
        target = controller.get_current_target()
        text = f"Look at the {controller.current_state.name} point and press 'C'"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
        
        # Draw target position indicator
        screen_x_ratio = target.x / controller.screen_width
        screen_y_ratio = target.y / controller.screen_height
        target_x = int(w * screen_x_ratio)
        target_y = int(h * screen_y_ratio)
        
        # Draw crosshair
        size = 20
        color = (0, 255, 0)
        thickness = 2
        cv2.line(frame, (target_x - size, target_y), (target_x + size, target_y), 
                 color, thickness)
        cv2.line(frame, (target_x, target_y - size), (target_x, target_y + size), 
                 color, thickness)
    else:
        cv2.putText(frame, "Calibration complete!", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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
        
        # Log model details
        self.input_details = self.session.get_inputs()
        self.output_details = self.session.get_outputs()
        
        logger.debug("Input details:")
        for input in self.input_details:
            logger.debug(f"Name: {input.name}, Shape: {input.shape}, Type: {input.type}")
        
        logger.debug("Output details:")
        for output in self.output_details:
            logger.debug(f"Name: {output.name}, Shape: {output.shape}, Type: {output.type}")
        
        self.input_name = self.input_details[0].name

    def preprocess_face(self, face_img):
        logger.debug(f"Preprocessing face image, input shape: {face_img.shape}")
        
        # Resize to 192x192
        face_img = cv2.resize(face_img, (192, 192))
        
        # Convert to float32 and normalize
        face_img = face_img.astype('float32') / 255.0
        
        # Transpose from HWC to CHW format (move channels to front)
        face_img = np.transpose(face_img, (2, 0, 1))
        
        # Add batch dimension
        face_img = np.expand_dims(face_img, axis=0)
        
        logger.debug(f"Preprocessed shape: {face_img.shape}")
        return face_img

    def analyze_face(self, face_img):
        try:
            processed_img = self.preprocess_face(face_img)
            logger.debug(f"Running inference with input shape: {processed_img.shape}")
            
            outputs = self.session.run(None, {self.input_name: processed_img})
            score, landmarks = outputs
            
            logger.debug(f"Detection score: {score[0]:.3f}")
            logger.debug(f"Landmarks shape: {landmarks.shape}")
            
            # Calculate head pose from landmarks
            # Using central face landmarks for pose estimation
            nose_tip = landmarks[0, 1, :]  # example landmark index
            left_eye = landmarks[0, 33, :]  # example landmark index
            right_eye = landmarks[0, 263, :]  # example landmark index
            
            # Basic pose estimation (this is a simplified version)
            # You might want to use a more sophisticated method
            face_center = landmarks[0].mean(axis=0)
            
            # Calculate basic angles
            pitch = np.arctan2(nose_tip[1] - face_center[1], nose_tip[2] - face_center[2])
            yaw = np.arctan2(nose_tip[0] - face_center[0], nose_tip[2] - face_center[2])
            roll = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
            
            # Convert to degrees and normalize to [0, 360] range
            pitch = (np.degrees(pitch) + 360) % 360
            yaw = (np.degrees(yaw) + 360) % 360
            roll = (np.degrees(roll) + 360) % 360
            
            # Convert to more intuitive ranges:
            # Yaw: 0 is center, negative is left, positive is right (-180 to +180)
            yaw = yaw if yaw <= 180 else yaw - 360
            
            # Pitch: 0 is center, negative is down, positive is up (-180 to +180)
            pitch = pitch if pitch <= 180 else pitch - 360
            
            # Roll: 0 is level, normalize to -180 to +180
            roll = roll if roll <= 180 else roll - 360
            
            # Optional: add smoothing to prevent jitter
            if hasattr(self, 'last_angles'):
                smooth_factor = 0.7  # Adjust this value (0-1) to change smoothing amount
                pitch = smooth_factor * self.last_angles['pitch'] + (1 - smooth_factor) * pitch
                yaw = smooth_factor * self.last_angles['yaw'] + (1 - smooth_factor) * yaw
                roll = smooth_factor * self.last_angles['roll'] + (1 - smooth_factor) * roll
            
            # Store angles for next frame smoothing
            self.last_angles = {'pitch': pitch, 'yaw': yaw, 'roll': roll}
            
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

def main():
    
    logger.info("Initializing application...")
    face_detector = FaceDetector()
    landmark_analyzer = FaceLandmarkAnalyzer()
    mouse_controller = DirectMouseController()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open camera")
        return
    logger.info("Camera initialized")
    
    # Make the window fullscreen
    cv2.namedWindow('Face Analysis', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Face Analysis', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to grab frame")
            break
            
        # Detect faces
        detections = face_detector.detect_faces(frame)
        
        if detections:
            # Get the detection with the highest confidence
            detection = max(detections, key=lambda x: x[4])
            face_roi = extract_face_roi(frame, detection)
            
            if face_roi.size > 0:
                analysis_result = landmark_analyzer.analyze_face(face_roi)
                
                if analysis_result is not None:
                    # Draw face detection rectangle
                    x, y, w, h = detection[:4]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Handle calibration and updates
                    key = cv2.waitKey(1) & 0xFF
                    if not mouse_controller.is_calibrated():
                        if key == ord('c'):
                            mouse_controller.calibrate_point(analysis_result)
                    else:
                        mouse_controller.update(analysis_result)
                    
                    # Draw calibration overlay
                    draw_calibration_overlay(frame, mouse_controller)
        
        cv2.imshow('Face Analysis', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 