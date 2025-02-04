import cv2
import numpy as np
import onnxruntime

class FaceAttributeDetector:
    def __init__(self, model_path='face_attrib_net.onnx'):
        # Initialize ONNX Runtime session
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def preprocess_image(self, img, target_size=(128, 128)):
        """Preprocess the image for model input"""
        # Resize image
        img = cv2.resize(img, target_size)
        
        # Convert BGR to RGB if needed
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to float32 and normalize to [0, 1]
        img = img.astype('float32') / 255.0
        
        # Apply model-specific normalization
        img = (img - 0.5) / 0.50196078
        
        # Transpose to NCHW format
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        return img

    def detect_attributes(self, img):
        """
        Detect facial attributes in the image
        Returns: Dictionary of facial attributes
        """
        # Preprocess image
        input_tensor = self.preprocess_image(img)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Process outputs
        attribute_names = ['id_feature', 'liveness_feature', 'eye_closeness', 
                         'glasses', 'mask', 'sunglasses']
        
        results = {}
        for name, output in zip(attribute_names, outputs):
            if name in ['eye_closeness', 'glasses', 'mask', 'sunglasses']:
                # Convert to probability using sigmoid for binary attributes
                prob = 1 / (1 + np.exp(-output))
                results[name] = float(prob[0][0])
            else:
                # Store feature vectors as is
                results[name] = output[0]
                
        return results

def main():
    # Initialize detector
    detector = FaceAttributeDetector('face_attrib_net.onnx')
    
    # Initialize video capture (0 for webcam)
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect attributes
        attributes = detector.detect_attributes(frame)
        
        # Create display frame
        display_frame = frame.copy()
        
        # Display results
        y_pos = 30
        for attr, value in attributes.items():
            if attr not in ['id_feature', 'liveness_feature']:  # Skip feature vectors
                text = f"{attr}: {value:.2f}"
                cv2.putText(display_frame, text, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_pos += 30
        
        # Display the frame
        cv2.imshow('Face Attributes', display_frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()