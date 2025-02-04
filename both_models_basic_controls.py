import cv2
import numpy as np
import pyautogui
import time
from collections import deque

import cv2
import numpy as np
import onnxruntime
import time

class FacePipeline:
    def __init__(self, face_model_path='face_det_lite.onnx', attr_model_path='face_attrib_net.onnx'):
        # Initialize models with optimizations
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4  # Adjust based on your CPU
        
        self.face_detector = onnxruntime.InferenceSession(
            face_model_path, 
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        self.attr_detector = onnxruntime.InferenceSession(
            attr_model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        
        self.face_input_name = self.face_detector.get_inputs()[0].name
        self.attr_input_name = self.attr_detector.get_inputs()[0].name
        
        # Cache preprocessed frames
        self.last_frame = None
        self.last_gray = None
        
        # Eye cascade with cached parameters
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Performance metrics
        self.fps = 0
        self.last_time = time.time()
        self.frame_count = 0

    def update_fps(self):
        self.frame_count += 1
        if self.frame_count >= 30:  # Update FPS every 30 frames
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

    def preprocess_attributes(self, img):
        """Optimized preprocessing for attribute detection"""
        img = cv2.resize(img, (128, 128))
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = img.astype('float32') / 255.0
        img = (img - 0.5) / 0.50196078
        img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
        return img

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

    def analyze_face(self, img, face_rect):
        """Analyze face attributes"""
        x, y, w, h = face_rect
        face_img = img[y:y+h, x:x+w]
        
        # Run attribute detection
        attr_input = self.preprocess_attributes(face_img)
        outputs = self.attr_detector.run(None, {self.attr_input_name: attr_input})
        
        # Process outputs
        results = {}
        for name, output in zip(['id_feature', 'liveness_feature', 'eye_closeness', 
                               'glasses', 'mask', 'sunglasses'], outputs):
            if name in ['eye_closeness', 'glasses', 'mask', 'sunglasses']:
                results[name] = float(1 / (1 + np.exp(-output[0][0])))
            else:
                results[name] = output[0]
        
        return results

    def process_frame(self, frame):
        """Process a single frame"""
        fps = self.update_fps()
        face_detections = self.detect_faces(frame)
        
        results = []
        for face_rect in face_detections[:1]:  # Process only the largest/closest face
            x, y, w, h, conf = face_rect
            attributes = self.analyze_face(frame, [x, y, w, h])
            results.append({
                'face_rect': face_rect,
                'attributes': attributes
            })
            
        return results, fps

class MouseController:
    def __init__(self, screen_width=1920, screen_height=1080, 
                 smoothing_window=5, sensitivity=2.0,
                 box_scale=0.3):
        """
        Initialize mouse controller with calibration box
        
        Args:
            screen_width: Display width in pixels
            screen_height: Display height in pixels 
            smoothing_window: Number of frames for movement smoothing
            sensitivity: Mouse movement multiplier
            box_scale: Size of calibration box relative to frame
        """
        # Screen parameters
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Calibration box parameters
        self.box_scale = box_scale
        self.is_calibrated = False
        self.center_position = None
        
        # Mouse control parameters
        self.sensitivity = sensitivity
        self.position_history = deque(maxlen=smoothing_window)
        
        # Click detection parameters
        self.blink_threshold = 0.7  # Eye closeness threshold
        self.click_duration = 1.0  # Seconds for left click
        self.double_blink_window = 0.5  # Seconds between blinks for right click
        
        # Click state tracking
        self.eyes_closed_start = None
        self.last_blink_time = None
        self.blink_count = 0
    
    def get_box_dimensions(self, frame_width, frame_height):
        """Calculate calibration box dimensions"""
        box_width = int(frame_width * self.box_scale)
        box_height = int(frame_height * self.box_scale)
        
        # Center box coordinates
        x1 = (frame_width - box_width) // 2
        y1 = (frame_height - box_height) // 2
        x2 = x1 + box_width
        y2 = y1 + box_height
        
        return x1, y1, x2, y2
    
    def calibrate(self, face_center):
        """Store reference face position"""
        self.center_position = face_center
        self.is_calibrated = True
    
    def get_movement_vector(self, current_pos):
        """Calculate mouse movement based on face position"""
        if not self.is_calibrated or not self.center_position:
            return 0, 0
            
        # Calculate deviation from center
        dx = (current_pos[0] - self.center_position[0]) * self.sensitivity
        dy = (current_pos[1] - self.center_position[1]) * self.sensitivity
        
        # Add to position history
        self.position_history.append((dx, dy))
        
        # Average recent movements for smoothing
        if len(self.position_history) > 0:
            dx = sum(p[0] for p in self.position_history) / len(self.position_history)
            dy = sum(p[1] for p in self.position_history) / len(self.position_history)
            
        return int(dx), int(dy)
    
    def process_eye_gestures(self, eye_closeness):
        """Handle eye-based click detection"""
        current_time = time.time()
        
        # Check if eyes are closed
        if eye_closeness > self.blink_threshold:
            if self.eyes_closed_start is None:
                self.eyes_closed_start = current_time
                
            # Left click - sustained closure
            if (current_time - self.eyes_closed_start) >= self.click_duration:
                pyautogui.click(button='left')
                self.eyes_closed_start = None
                return
                
        # Eyes opened
        else:
            if self.eyes_closed_start is not None:
                blink_duration = current_time - self.eyes_closed_start
                
                # Detect quick blink
                if blink_duration < self.click_duration:
                    if self.last_blink_time and \
                       (current_time - self.last_blink_time) < self.double_blink_window:
                        # Double blink detected - right click
                        pyautogui.click(button='right')
                        self.last_blink_time = None
                    else:
                        self.last_blink_time = current_time
                        
            self.eyes_closed_start = None

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
        
        # Add calibration status
        status = "Calibrated" if self.is_calibrated else "Center face in box and press 'c'"
        cv2.putText(frame, status, (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame

def main():
    # Initialize face pipeline and mouse controller
    pipeline = FacePipeline('face_det_lite.onnx', 'face_attrib_net.onnx')
    controller = MouseController()
    
    # Disable pyautogui fail-safe
    pyautogui.FAILSAFE = False
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        results, fps = pipeline.process_frame(frame)
        
        # Draw calibration box
        frame = controller.draw_calibration_box(frame)
        
        if results:
            face_rect = results[0]['face_rect']
            attributes = results[0]['attributes']
            x, y, w, h = face_rect[:4]
            
            # Get face center
            face_center = (x + w//2, y + h//2)
            
            # Handle calibration
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                controller.calibrate(face_center)
            elif key == ord('q'):
                break
                
            # Move mouse if calibrated
            if controller.is_calibrated:
                dx, dy = controller.get_movement_vector(face_center)
                current_x, current_y = pyautogui.position()
                pyautogui.moveTo(current_x + dx, current_y + dy)
                
            # Process eye gestures
            if 'eye_closeness' in attributes:
                controller.process_eye_gestures(attributes['eye_closeness'])
        
        # Display frame
        cv2.imshow('Astraea Face Control', frame)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()