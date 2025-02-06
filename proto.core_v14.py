"""
a working face detection and landmark analysis pipeline

combines two ONNX models for real-time face analysis:
face_det_lite.onnx - A lightweight face detection model that processes full frames
mediapipe_face_landmark.onnx - mediapipes face landmark model that analyzes individual faces

1. Captures video from webcam
2. Detects faces in each frame
3. For the primary detected face:
   - Extracts and normalizes the face region
   - sends to mediapipe_face_landmark.onnx
   - Calculates head pose (pitch, yaw, roll) from landmarks
4. Displys results in real-time with visual overlays

"""

import cv2
import numpy as np
import onnxruntime
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
    
    cap = cv2.VideoCapture(0)
    logger.info("Camera initialized")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to grab frame")
            break
            
        # Detect faces
        detections = face_detector.detect_faces(frame)
        
        if detections:
            # Get the first face (highest confidence)
            detection = max(detections, key=lambda x: x[4])
            
            # Extract face ROI
            face_roi = extract_face_roi(frame, detection)
            
            if face_roi.size > 0:
                # Analyze face
                analysis_result = landmark_analyzer.analyze_face(face_roi)
                
                if analysis_result is not None:
                    # Draw the face detection
                    x, y, w, h = detection[:4]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Display results
                    try:
                        pitch = analysis_result['pitch']
                        yaw = analysis_result['yaw']
                        roll = analysis_result['roll']
                        score = analysis_result['score']
                        
                        # Display angles with direction indicators
                        yaw_dir = "Right" if yaw > 0 else "Left" if yaw < 0 else "Center"
                        pitch_dir = "Up" if pitch > 0 else "Down" if pitch < 0 else "Center"
                        roll_dir = "->" if roll > 0 else "<-" if roll < 0 else "Center"
                        
                        cv2.putText(frame, f"Pitch: {pitch:>6.1f}deg {pitch_dir}", (x, y - 40),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(frame, f"Yaw: {yaw:>6.1f}deg {yaw_dir}", (x, y - 25),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(frame, f"Roll: {roll:>6.1f}deg {roll_dir}", (x, y - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Draw some landmarks for visualization
                        landmarks = analysis_result['landmarks']
                        for i in range(0, landmarks.shape[0], 10):  # Draw every 10th landmark
                            lx = int(x + landmarks[i, 0] * w)
                            ly = int(y + landmarks[i, 1] * h)
                            cv2.circle(frame, (lx, ly), 1, (0, 0, 255), -1)
                    except Exception as e:
                        logger.error(f"Error displaying results: {str(e)}")
        
        cv2.imshow('Face Analysis', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()