import cv2
import numpy as np
import onnxruntime
import logging

logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self, model_path='models/face_det_lite.onnx'):
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

def extract_face_roi(frame, detection):
    x, y, w, h = detection[:4]
    return frame[y:y+h, x:x+w] 