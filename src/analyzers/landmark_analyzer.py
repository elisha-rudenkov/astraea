import cv2
import numpy as np
import onnxruntime
import logging

logger = logging.getLogger(__name__)

class FaceLandmarkAnalyzer:
    def __init__(self, model_path='models/mediapipe_face-mediapipefacelandmarkdetector.onnx'):
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
            
            nose_tip = landmarks[0, 1, :]
            left_eye = landmarks[0, 33, :]
            right_eye = landmarks[0, 263, :]
            
            eye_center = (left_eye + right_eye) / 2.0
            
            pitch = np.arctan2(nose_tip[1] - eye_center[1], nose_tip[2] - eye_center[2])
            yaw   = np.arctan2(nose_tip[0] - eye_center[0], nose_tip[2] - eye_center[2])
            roll  = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
            
            pitch = np.degrees(pitch)
            yaw   = np.degrees(yaw)
            roll  = np.degrees(roll)
            
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