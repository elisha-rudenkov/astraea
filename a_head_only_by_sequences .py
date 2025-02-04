
import cv2
import numpy as np
import pyautogui
import time
from collections import deque
import onnxruntime

class FacePipeline:
    def __init__(self, face_model_path='face_det_lite.onnx'):
        # Initialize face detection model with optimizations
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        
        self.face_detector = onnxruntime.InferenceSession(
            face_model_path, 
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        
        self.face_input_name = self.face_detector.get_inputs()[0].name
        
        # Cache preprocessed frames
        self.last_frame = None
        self.last_gray = None
        
        # Performance metrics
        self.fps = 0
        self.last_time = time.time()
        self.frame_count = 0

    def update_fps(self):
        self.frame_count += 1
        if self.frame_count >= 30:
            current_time = time.time()
            self.fps = self.frame_count / (current_time - self.last_time)
            self.last_time = current_time
            self.frame_count = 0
        return self.fps

    def preprocess_face_detection(self, img):
        """Optimized preprocessing for face detection"""
        if img is self.last_frame:
            return self.last_gray
            
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        gray = cv2.resize(gray, (640, 480))
        processed = (gray.astype('float32') / 255.0 - 0.442) / 0.280
        processed = np.expand_dims(np.expand_dims(processed, axis=0), axis=0)
        
        self.last_frame = img
        self.last_gray = processed
        return processed

    def detect_faces(self, img, conf_threshold=0.55):
        """Optimized face detection"""
        input_tensor = self.preprocess_face_detection(img)
        outputs = self.face_detector.run(None, {self.face_input_name: input_tensor})
        
        heatmap, bbox_reg, _ = outputs
        heatmap = 1 / (1 + np.exp(-heatmap))
        
        confidence_mask = heatmap[0, 0] > conf_threshold
        y_indices, x_indices = np.where(confidence_mask)
        
        detections = []
        for y, x in zip(y_indices, x_indices):
            dx1, dy1, dx2, dy2 = bbox_reg[0, :, y, x]
            
            stride = 8
            x1 = int((x - dx1) * stride)
            y1 = int((y - dy1) * stride)
            w = int((dx1 + dx2) * stride)
            h = int((dy1 + dy2) * stride)
            
            # Add padding
            x1 = max(0, int(x1 - w * 0.05))
            y1 = max(0, int(y1 - h * 0.05))
            w = min(int(w * 1.1), 640 - x1)
            h = min(int(h * 1.1), 480 - y1)
            
            detections.append([x1, y1, w, h, float(heatmap[0, 0, y, x])])
            
        return detections

    def process_frame(self, frame):
        """Process a single frame"""
        fps = self.update_fps()
        face_detections = self.detect_faces(frame)
        return face_detections, fps

class HeadGestureController:
    def __init__(self, screen_width=1920, screen_height=1080, 
                 smoothing_window=5, sensitivity=20.0,
                 box_scale=0.3):
        # Screen parameters
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Calibration box parameters
        self.box_scale = box_scale
        self.is_calibrated = False
        self.center_position = None
        
        self.horizontal_dead_zone = 20
        self.vertical_dead_zone = 10  # Smaller dead zone for vertical movement
        self.vertical_sensitivity = 25.0  # Separate sensitivity for vertical movement
        self.sensitivity = sensitivity
        
        # Gesture detection parameters
        self.gesture_history = deque(maxlen=4)  # Shorter history for quicker detection
        self.gesture_times = deque(maxlen=4)
        self.gesture_timeout = 0.4  # Shorter timeout for more responsive clicks
        self.velocity_threshold = 200  # Pixels per second for click detection
        self.min_gesture_distance = 15
        
        # Cooldown for click actions
        self.last_click_time = 0
        self.click_cooldown = 0.3
        
        # Movement smoothing
        self.position_history = deque(maxlen=smoothing_window)
        self.velocity_history = deque(maxlen=3)  # For detecting quick movements
    
    def get_box_dimensions(self, frame_width, frame_height):
        """Calculate calibration box dimensions"""
        box_width = int(frame_width * self.box_scale)
        box_height = int(frame_height * self.box_scale)
        
        x1 = (frame_width - box_width) // 2
        y1 = (frame_height - box_height) // 2
        x2 = x1 + box_width
        y2 = y1 + box_height
        
        return x1, y1, x2, y2
    
    def calibrate(self, face_center):
        """Store reference face position"""
        self.center_position = face_center
        self.is_calibrated = True
        self.gesture_history.clear()
        self.gesture_times.clear()
        self.velocity_history.clear()
    
    def get_movement_vector(self, current_pos):
        """Calculate mouse movement with different vertical and horizontal scaling"""
        if not self.is_calibrated or not self.center_position:
            return 0, 0
            
        # Calculate deviation from center
        dx = current_pos[0] - self.center_position[0]
        dy = current_pos[1] - self.center_position[1]
        
        # Handle horizontal movement
        if abs(dx) < self.horizontal_dead_zone:
            final_dx = 0
        else:
            adjusted_dx = abs(dx) - self.horizontal_dead_zone
            speed_x = min((adjusted_dx / 100) ** 2 * self.sensitivity, self.sensitivity)
            final_dx = speed_x * (-1 if dx > 0 else 1)
        
        # Handle vertical movement with custom curve
        if abs(dy) < self.vertical_dead_zone:
            final_dy = 0
        else:
            # More sensitive initial movement, then accelerating curve
            adjusted_dy = abs(dy) - self.vertical_dead_zone
            base_speed = adjusted_dy * 0.3  # Linear component for fine control
            exp_speed = (adjusted_dy / 80) ** 1.5 * self.vertical_sensitivity  # Exponential for fast movement
            speed_y = min(base_speed + exp_speed, self.vertical_sensitivity * 1.5)
            final_dy = speed_y * (1 if dy > 0 else -1)
        
        # Add to position history for smoothing
        self.position_history.append((final_dx, final_dy))
        
        # Average recent movements
        if self.position_history:
            smooth_dx = sum(p[0] for p in self.position_history) / len(self.position_history)
            smooth_dy = sum(p[1] for p in self.position_history) / len(self.position_history)
            return int(smooth_dx), int(smooth_dy)
            
        return int(final_dx), int(final_dy)

    def detect_gesture(self, current_pos):
        """Detect gestures based on movement velocity"""
        current_time = time.time()
        
        # Add current position and time to history
        self.gesture_history.append(current_pos)
        self.gesture_times.append(current_time)
        
        # Remove old gestures
        while self.gesture_times and (current_time - self.gesture_times[0]) > self.gesture_timeout:
            self.gesture_history.popleft()
            self.gesture_times.popleft()
        
        if len(self.gesture_history) < 2:
            return None
            
        # Check for click cooldown
        if current_time - self.last_click_time < self.click_cooldown:
            return None
        
        # Calculate vertical velocity (pixels per second)
        dy = self.gesture_history[-1][1] - self.gesture_history[-2][1]
        dt = self.gesture_times[-1] - self.gesture_times[-2]
        if dt > 0:
            velocity = abs(dy / dt)
            self.velocity_history.append(velocity)
            
            # Check for quick vertical movement (nod)
            if velocity > self.velocity_threshold:
                # Verify it's a natural nod pattern by checking direction changes
                if len(self.gesture_history) >= 3:
                    dy1 = self.gesture_history[-2][1] - self.gesture_history[-3][1]
                    dy2 = self.gesture_history[-1][1] - self.gesture_history[-2][1]
                    if dy1 * dy2 < 0:  # Direction changed
                        self.last_click_time = current_time
                        self.gesture_history.clear()
                        self.gesture_times.clear()
                        return 'left_click'
            
            # Check for quick horizontal movement (right click)
            dx = self.gesture_history[-1][0] - self.gesture_history[-2][0]
            velocity_x = abs(dx / dt)
            if velocity_x > self.velocity_threshold * 1.2:  # Slightly higher threshold for horizontal
                if len(self.gesture_history) >= 3:
                    dx1 = self.gesture_history[-2][0] - self.gesture_history[-3][0]
                    dx2 = self.gesture_history[-1][0] - self.gesture_history[-2][0]
                    if dx1 * dx2 < 0:  # Direction changed
                        self.last_click_time = current_time
                        self.gesture_history.clear()
                        self.gesture_times.clear()
                        return 'right_click'
        
        return None

    def draw_calibration_box(self, frame):
        """Draw calibration guide on frame"""
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = self.get_box_dimensions(width, height)
        
        # Draw calibration box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw center crosshair
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        size = 20
        cv2.line(frame, (center_x - size, center_y), (center_x + size, center_y), (0, 255, 0), 2)
        cv2.line(frame, (center_x, center_y - size), (center_x, center_y + size), (0, 255, 0), 2)
        
        # Add calibration status and instructions
        status = "Calibrated" if self.is_calibrated else "Center face in box and press 'c'"
        cv2.putText(frame, status, (10, height - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add gesture instructions
        if self.is_calibrated:
            cv2.putText(frame, "Left click: Quick nod", (10, height - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, "Right click: Quick head shake", (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame

def main():
    try:
        print("Initializing...")
        pipeline = FacePipeline('face_det_lite.onnx')
        controller = HeadGestureController()
        
        pyautogui.FAILSAFE = False
        
        print("Opening camera...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise RuntimeError("Failed to open camera")
            
        print("Camera opened successfully")
        
        ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to read first frame")
            
        print(f"Frame dimensions: {frame.shape}")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
                
            frame_count += 1
            
            # Process frame
            face_detections, fps = pipeline.process_frame(frame)
            
            # Draw calibration box
            frame = controller.draw_calibration_box(frame)
            
            if face_detections:
                face_rect = face_detections[0]  # Use largest/closest face
                x, y, w, h = face_rect[:4]
                
                # Get face center
                face_center = (x + w//2, y + h//2)
                
                # Handle calibration
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    controller.calibrate(face_center)
                    print("Calibrated to position:", face_center)
                elif key == ord('q'):
                    print("Quit command received")
                    break
                    
                # Process gestures and move mouse if calibrated
                if controller.is_calibrated:
                    # Check for gestures
                    gesture = controller.detect_gesture(face_center)
                    if gesture == 'left_click':
                        pyautogui.click(button='left')
                        print("Left click")
                    elif gesture == 'right_click':
                        pyautogui.click(button='right')
                        print("Right click")
                    
                    # Move mouse based on head position
                    dx, dy = controller.get_movement_vector(face_center)
                    if dx != 0 or dy != 0:
                        current_x, current_y = pyautogui.position()
                        pyautogui.moveTo(current_x + dx, current_y + dy)
            
            # Display frame
            cv2.imshow('Astraea Head Control', frame)
            
            # Check if window was closed
            if cv2.getWindowProperty('Astraea Head Control', cv2.WND_PROP_VISIBLE) < 1:
                break
                
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        if 'cap' in locals():
            cap.release()
            cv2.destroyAllWindows()
        print("Cleanup complete")

if __name__ == "__main__":
    main()