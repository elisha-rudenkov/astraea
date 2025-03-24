import cv2
import numpy as np
import onnxruntime
import logging
from src.utils.onnx_utils import create_inference_session

logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self, model_path='models/face_det_lite.onnx', use_gpu=True):
        logger.info(f"Initializing FaceDetector with model: {model_path}")
        # Use the GPU-accelerated session creation utility
        self.session = create_inference_session(model_path, use_gpu)
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
        """
        Detect faces in the input image.
        
        Args:
            img: Input image (BGR or grayscale)
            conf_threshold: Confidence threshold for face detection (default: 0.55)
            
        Returns:
            List of face detections, each as [x, y, width, height, confidence]
        """
        # Preprocess the image for the neural network
        input_tensor = self.preprocess_image(img)
        
        # Run inference with the ONNX model
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Unpack the model outputs
        face_heatmap, bounding_box_regression, facial_landmarks = outputs
        
        # Apply sigmoid to convert logits to probability scores (0-1 range)
        # sigmoid(x) = 1 / (1 + e^(-x))
        face_heatmap = 1 / (1 + np.exp(-face_heatmap))
        
        # Initialize list to store detected faces
        face_detections = []
        
        # Create a mask of points where confidence exceeds the threshold
        confidence_mask = face_heatmap[0, 0] > conf_threshold
        
        # Get coordinates of points that passed the threshold
        y_coords, x_coords = np.where(confidence_mask)
        
        # Process each detection point
        for y_pos, x_pos in zip(y_coords, x_coords):
            # Get the confidence score at this position
            confidence_score = face_heatmap[0, 0, y_pos, x_pos]
            
            # Get the bounding box regression values (offsets)
            # dx1, dy1: offsets from the point to the top-left corner
            # dx2, dy2: offsets from the point to the bottom-right corner
            offset_left, offset_top, offset_right, offset_bottom = bounding_box_regression[0, :, y_pos, x_pos]
            
            # The model operates on a downsampled feature map with stride=8
            # We need to scale coordinates back to the original image space
            stride = 8
            
            # Calculate the absolute coordinates of the bounding box
            # The detection point (x_pos, y_pos) is treated as a reference point
            # from which we compute the actual box coordinates using the offsets
            left = (x_pos - offset_left) * stride
            top = (y_pos - offset_top) * stride
            right = (x_pos + offset_right) * stride
            bottom = (y_pos + offset_bottom) * stride
            
            # Convert to width and height format
            width = right - left
            height = bottom - top
            
            # Add a 5% margin around the face (expand the box by 10%)
            # This ensures we capture the full face with some context
            left = left - width * 0.05
            top = top - height * 0.05
            width = width * 1.1
            height = height * 1.1
            
            # Ensure the bounding box stays within image boundaries
            left = max(0, min(left, 640))
            top = max(0, min(top, 480))
            width = min(width, 640 - left)
            height = min(height, 480 - top)
            
            # Add the detection to our results list
            face_detections.append([int(left), int(top), int(width), int(height), float(confidence_score)])
        
        return face_detections

def extract_face_roi(frame, detection):
    x, y, w, h = detection[:4]
    return frame[y:y+h, x:x+w] 