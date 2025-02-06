import cv2
import numpy as np
import onnxruntime
import math

def calculate_angles(landmarks):
    """Calculate face angles using more stable landmark points"""
    # Using more stable landmarks
    nose = landmarks[1]    
    left_eye = np.mean([landmarks[33], landmarks[133]], axis=0)  # Average multiple points around eye
    right_eye = np.mean([landmarks[263], landmarks[362]], axis=0)
    left_ear = np.mean([landmarks[234], landmarks[227]], axis=0)  # Average multiple points around ear
    right_ear = np.mean([landmarks[454], landmarks[447]], axis=0)
    
    # Calculate face normal vector with smoothing
    face_normal = np.cross(right_eye - left_eye, nose - (left_eye + right_eye) / 2)
    face_normal = face_normal / (np.linalg.norm(face_normal) + 1e-6)
    
    # Calculate angles with clamping to avoid unstable values
    pitch = math.degrees(math.asin(np.clip(-face_normal[1], -0.9999, 0.9999)))
    yaw = math.degrees(math.atan2(face_normal[0], max(abs(face_normal[2]), 0.01)))
    
    # Calculate roll with stabilization
    eye_vector = right_eye - left_eye
    roll = math.degrees(math.atan2(eye_vector[1], eye_vector[0]))
    
    # Clamp angles to reasonable ranges
    pitch = np.clip(pitch, -90, 90)
    yaw = np.clip(yaw, -90, 90)
    roll = np.clip(roll, -90, 90)
    
    return pitch, yaw, roll

def preprocess_image(frame):
    # Adjust preprocessing for better stability
    input_size = (192, 192)
    
    # Calculate center crop
    h, w = frame.shape[:2]
    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2
    
    # Crop to square
    frame = frame[top:top+min_dim, left:left+min_dim]
    
    # Resize and normalize
    img = cv2.resize(frame, input_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to float32 and normalize
    img = img.astype(np.float32)
    img = img / 255.0
    
    # Make sure we're using float32 arrays
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    # Standardize
    img = (img - mean) / std
    
    # Ensure float32 type is maintained
    img = img.astype(np.float32)
    
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32)  # Final type check

def moving_average(new_value, history, alpha=0.5):
    """Simple exponential moving average for smoothing"""
    if history is None:
        return new_value
    return alpha * new_value + (1 - alpha) * history

class AngleTracker:
    def __init__(self):
        self.pitch_history = None
        self.yaw_history = None
        self.roll_history = None
        
    def update(self, pitch, yaw, roll):
        self.pitch_history = moving_average(pitch, self.pitch_history)
        self.yaw_history = moving_average(yaw, self.yaw_history)
        self.roll_history = moving_average(roll, self.roll_history)
        return self.pitch_history, self.yaw_history, self.roll_history

def main():
    session = onnxruntime.InferenceSession("mediapipe_face-mediapipefacelandmarkdetector.onnx")
    input_name = session.get_inputs()[0].name
    
    cap = cv2.VideoCapture(0)
    tracker = AngleTracker()
    
    # Store previous landmarks for smoothing
    prev_landmarks = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        input_data = preprocess_image(frame)
        outputs = session.run(None, {input_name: input_data})
        
        confidence = outputs[0][0]
        landmarks = outputs[1][0]
        
        # Smooth landmarks
        if prev_landmarks is not None:
            landmarks = 0.7 * landmarks + 0.3 * prev_landmarks
        prev_landmarks = landmarks.copy()
        
        if confidence > 0.5:
            # Calculate and smooth angles
            pitch, yaw, roll = calculate_angles(landmarks)
            pitch, yaw, roll = tracker.update(pitch, yaw, roll)
            
            # Draw face mesh (less points for clarity)
            h, w = frame.shape[:2]
            for i, (x, y, _) in enumerate(landmarks):
                if i % 5 == 0:  # Draw fewer points
                    x_px = int(x * w)
                    y_px = int(y * h)
                    cv2.circle(frame, (x_px, y_px), 1, (0, 255, 0), -1)
            
            # Draw key landmarks
            key_points = [1, 33, 263, 234, 454]
            for idx in key_points:
                x, y, _ = landmarks[idx]
                x_px = int(x * w)
                y_px = int(y * h)
                cv2.circle(frame, (x_px, y_px), 3, (0, 0, 255), -1)
            
            # Display smoothed angles
            cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Yaw: {yaw:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Roll: {roll:.1f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Conf: {confidence:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Face Angles Demo', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()