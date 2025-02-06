import cv2
import numpy as np
import onnxruntime

class FaceDetector:
    def __init__(self, model_path='face_det_lite.onnx'):
        # Initialize ONNX Runtime session
        self.session = onnxruntime.InferenceSession(model_path)
        
        # Get input details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        print(f"Model expects input shape: {self.input_shape}")

    def preprocess_image(self, img):
        """Preprocess the image for model input"""
        # Convert to grayscale first if image is BGR
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        # Resize to exact input dimensions
        img = cv2.resize(img, (640, 480))
        
        # Convert to float32 and normalize
        img = img.astype('float32') / 255.0
        img = (img - 0.442) / 0.280
        
        # Add batch and channel dimensions to match [1, 1, 480, 640]
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = np.expand_dims(img, axis=0)  # Add channel dimension
        
        return img.astype(np.float32)  # Ensure float32 type

    def detect_faces(self, img, conf_threshold=0.55):
        """
        Detect faces in the image
        Returns: List of [x, y, width, height, confidence] for each detection
        """
        # Get original image dimensions
        orig_h, orig_w = img.shape[:2]

        # Preprocess image
        input_tensor = self.preprocess_image(img)
        
        # Print shape before inference
        print(f"Input tensor shape: {input_tensor.shape}")
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Post-process outputs
        heatmap, bbox_reg, landmarks = outputs
        
        # Process heatmap
        heatmap = 1 / (1 + np.exp(-heatmap))  # sigmoid
        
        # Find face detections
        detections = []
        confidence_mask = heatmap[0, 0] > conf_threshold
        y_indices, x_indices = np.where(confidence_mask)
        
        for y, x in zip(y_indices, x_indices):
            confidence = heatmap[0, 0, y, x]
            
            # Get bbox regression values
            dx1, dy1, dx2, dy2 = bbox_reg[0, :, y, x]
            
            # Convert to pixel coordinates (stride = 8)
            stride = 8
            x1 = (x - dx1) * stride
            y1 = (y - dy1) * stride
            x2 = (x + dx2) * stride
            y2 = (y + dy2) * stride
            
            # Calculate width and height
            w = x2 - x1
            h = y2 - y1
            
            # Add some padding (as in original implementation)
            x1 = x1 - w * 0.05
            y1 = y1 - h * 0.05
            w = w * 1.1
            h = h * 1.1
            
            # Clip to image boundaries
            x1 = max(0, min(x1, 640))
            y1 = max(0, min(y1, 480))
            w = min(w, 640 - x1)
            h = min(h, 480 - y1)
            
            # Add detection
            detections.append([int(x1), int(y1), int(w), int(h), float(confidence)])
        
        return detections

def draw_detections(image, detections):
    """Draw detected faces on the image"""
    img_copy = image.copy()
    for x, y, w, h, conf in detections:
        # Draw rectangle
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Draw confidence
        label = f"{conf:.2f}"
        cv2.putText(img_copy, label, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img_copy

def main():
    # Initialize detector
    detector = FaceDetector('face_det_lite.onnx')
    
    # Initialize video capture (0 for webcam)
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect faces
        detections = detector.detect_faces(frame)
        
        # Draw detections
        output_frame = draw_detections(frame, detections)
        
        # Display result
        cv2.imshow('Face Detection', output_frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()