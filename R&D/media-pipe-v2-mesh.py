import cv2
import numpy as np
import pyautogui
import time
from collections import deque
import mediapipe as mp

class FacePipeline:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
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

    def detect_faces(self, frame):
        """Detect face and calculate head pose using MediaPipe"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return []
            
        face_landmarks = results.multi_face_landmarks[0]
        
        # Get image dimensions
        image_height, image_width = frame.shape[:2]
        
        # Get key landmarks for pose estimation
        nose = face_landmarks.landmark[1]
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]
        
        # Convert landmarks to pixel coordinates
        nose_2d = (int(nose.x * image_width), int(nose.y * image_height))
        left_eye_2d = (int(left_eye.x * image_width), int(left_eye.y * image_height))
        right_eye_2d = (int(right_eye.x * image_width), int(right_eye.y * image_height))
        
        # Calculate head pose angles
        eye_line = np.array(right_eye_2d) - np.array(left_eye_2d)
        roll = np.arctan2(eye_line[1], eye_line[0]) * 180 / np.pi
        pitch = (0.5 - nose.y) * 90
        yaw = (0.5 - nose.x) * 90
        
        # Calculate face bounding box
        x_coords = [int(lm.x * image_width) for lm in face_landmarks.landmark]
        y_coords = [int(lm.y * image_height) for lm in face_landmarks.landmark]
        
        x1, y1 = min(x_coords), min(y_coords)
        w = max(x_coords) - x1
        h = max(y_coords) - y1
        
        # Add visual feedback for debugging
        cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Yaw: {yaw:.1f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Roll: {roll:.1f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return [[x1, y1, w, h, 1.0, roll, pitch, yaw]]

    def process_frame(self, frame):
        """Process a single frame"""
        fps = self.update_fps()
        face_detections = self.detect_faces(frame)
        return face_detections, fps

class HeadGestureController:
    def __init__(self, screen_width=1920, screen_height=1080, 
                 smoothing_window=5, sensitivity=2.0,
                 box_scale=0.3):
        # Screen parameters
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Calibration box parameters
        self.box_scale = box_scale
        self.is_calibrated = False
        self.center_position = None
        
        self.horizontal_dead_zone = 5
        self.vertical_dead_zone = 5
        self.vertical_sensitivity = 2.5
        self.sensitivity = sensitivity
        
        # Movement smoothing
        self.position_history = deque(maxlen=smoothing_window)
        
        # Cooldown for click actions
        self.last_click_time = 0
        self.click_cooldown = 0.3
    
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
        self.position_history.clear()
    
    def get_movement_vector(self, current_pos, pitch, yaw):
        """Calculate mouse movement using pitch and yaw angles"""
        if not self.is_calibrated or not self.center_position:
            return 0, 0
            
        # Use yaw for horizontal movement (x-axis)
        if abs(yaw) < self.horizontal_dead_zone:
            final_dx = 0
        else:
            adjusted_dx = abs(yaw) - self.horizontal_dead_zone
            speed_x = min((adjusted_dx / 45) ** 1.5 * self.sensitivity, self.sensitivity * 2)
            final_dx = speed_x * (-1 if yaw > 0 else 1)
        
        # Use pitch for vertical movement (y-axis)
        if abs(pitch) < self.vertical_dead_zone:
            final_dy = 0
        else:
            adjusted_dy = abs(pitch) - self.vertical_dead_zone
            speed_y = min((adjusted_dy / 45) ** 1.5 * self.vertical_sensitivity, self.vertical_sensitivity * 2)
            final_dy = speed_y * (1 if pitch > 0 else -1)
        
        # Smoothing
        self.position_history.append((final_dx, final_dy))
        
        if self.position_history:
            smooth_dx = sum(p[0] for p in self.position_history) / len(self.position_history)
            smooth_dy = sum(p[1] for p in self.position_history) / len(self.position_history)
            return int(smooth_dx), int(smooth_dy)
            
        return int(final_dx), int(final_dy)

    def detect_gesture(self, current_pos, roll):
        """Detect clicks based on roll angle"""
        current_time = time.time()
        
        if current_time - self.last_click_time < self.click_cooldown:
            return None
            
        if abs(roll) > 25:
            self.last_click_time = current_time
            return 'left_click' if roll < 0 else 'right_click'
            
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
            cv2.putText(frame, "Left click: Tilt head left", (10, height - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, "Right click: Tilt head right", (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame

def main():
    try:
        print("Initializing...")
        pipeline = FacePipeline()
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
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # Process frame
            face_detections, fps = pipeline.process_frame(frame)
            
            # Draw calibration box
            frame = controller.draw_calibration_box(frame)
            
            if face_detections:
                face_rect = face_detections[0]  # Use largest/closest face
                x, y, w, h = face_rect[:4]
                roll, pitch, yaw = face_rect[5:8]  # Get pose angles
                
                # Get face center
                face_center = (x + w//2, y + h//2)
                
                # Draw FPS
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Handle calibration
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    controller.calibrate(face_center)
                    print("Calibrated to position:", face_center)
                    print(f"Initial angles - Pitch: {pitch:.1f}, Yaw: {yaw:.1f}")
                elif key == ord('q'):
                    print("Quit command received")
                    break
                    
                # Process gestures and move mouse if calibrated
                if controller.is_calibrated:
                    # Move mouse based on pitch and yaw
                    dx, dy = controller.get_movement_vector(face_center, pitch, yaw)
                    if dx != 0 or dy != 0:
                        current_x, current_y = pyautogui.position()
                        new_x = current_x + dx
                        new_y = current_y + dy
                        print(f"Moving mouse - dx: {dx}, dy: {dy}, New pos: ({new_x}, {new_y})")
                        pyautogui.moveTo(new_x, new_y)
                    
                    # Check for gestures using roll angle
                    gesture = controller.detect_gesture(face_center, roll)
                    if gesture == 'left_click':
                        pyautogui.click(button='left')
                        print("Left click")
                    elif gesture == 'right_click':
                        pyautogui.click(button='right')
                        print("Right click")
            
            # Display frame
            cv2.imshow('Head Mouse Control', frame)
            
            # Check if window was closed
            if cv2.getWindowProperty('Head Mouse Control', cv2.WND_PROP_VISIBLE) < 1:
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