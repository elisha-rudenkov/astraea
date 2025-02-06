import cv2
import numpy as np
import onnxruntime

class FaceAttributeDetector:
    def __init__(self, model_path='face_attrib_net.onnx'):
        # Initialize ONNX Runtime session
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Load eye detector
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )

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

    def analyze_individual_eyes(self, gray_img):
        """Analyze each eye separately using CV techniques"""
        eyes = self.eye_cascade.detectMultiScale(
            gray_img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        
        left_eye = None
        right_eye = None
        
        if len(eyes) >= 2:
            # Sort eyes by x-coordinate
            eyes = sorted(eyes, key=lambda x: x[0])
            left_eye = eyes[0]
            right_eye = eyes[1]
            
        return left_eye, right_eye

    def calculate_eye_aspect_ratio(self, eye_roi):
        """Calculate eye aspect ratio as a measure of eye openness"""
        if eye_roi is None:
            return None
            
        # Convert ROI to grayscale if it's not already
        if len(eye_roi.shape) == 3:
            eye_roi = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to create a binary image
        _, thresh = cv2.threshold(eye_roi, 30, 255, cv2.THRESH_BINARY)
        
        # Calculate the white pixel ratio (indication of eye openness)
        white_pixels = np.sum(thresh == 255)
        total_pixels = thresh.size
        openness_ratio = white_pixels / total_pixels
        
        return openness_ratio

    def detect_attributes(self, img):
        """
        Detect facial attributes in the image
        Returns: Dictionary of facial attributes
        """
        # Get original image dimensions
        orig_h, orig_w = img.shape[:2]
        
        # Preprocess image for the main model
        input_tensor = self.preprocess_image(img)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Process outputs
        attribute_names = ['id_feature', 'liveness_feature', 'eye_closeness', 
                         'glasses', 'mask', 'sunglasses']
        
        results = {}
        for name, output in zip(attribute_names, outputs):
            if name in ['eye_closeness', 'glasses', 'mask', 'sunglasses']:
                prob = 1 / (1 + np.exp(-output))
                results[name] = float(prob[0][0])
            else:
                results[name] = output[0]
        
        # Add individual eye analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        left_eye, right_eye = self.analyze_individual_eyes(gray)
        
        results['left_eye'] = None
        results['right_eye'] = None
        
        if left_eye is not None:
            x, y, w, h = left_eye
            left_roi = gray[y:y+h, x:x+w]
            results['left_eye'] = {
                'bbox': left_eye,
                'openness': self.calculate_eye_aspect_ratio(left_roi)
            }
            
        if right_eye is not None:
            x, y, w, h = right_eye
            right_roi = gray[y:y+h, x:x+w]
            results['right_eye'] = {
                'bbox': right_eye,
                'openness': self.calculate_eye_aspect_ratio(right_roi)
            }
                
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
        #log arttributes
        print(attributes)
        
        # Create display frame
        display_frame = frame.copy()
        
        # Display results
        y_pos = 30
        # Display basic attributes
        for attr in ['glasses', 'mask', 'sunglasses']:
            if attr in attributes:
                text = f"{attr}: {attributes[attr]:.2f}"
                cv2.putText(display_frame, text, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_pos += 30
        
        # Display eye-specific results
        for eye_name in ['left_eye', 'right_eye']:
            if attributes[eye_name] is not None:
                eye_data = attributes[eye_name]
                if eye_data['openness'] is not None:
                    text = f"{eye_name} openness: {eye_data['openness']:.2f}"
                    cv2.putText(display_frame, text, (10, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_pos += 30
                
                # Draw eye bounding boxes
                x, y, w, h = eye_data['bbox']
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
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