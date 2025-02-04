import cv2
import numpy as np
import onnxruntime

class FaceAttributeDetector:
    def __init__(self, model_path='face_attrib_net.onnx'):
        # Initialize ONNX Runtime session
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Load face and eye detectors
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
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

    def detect_face_area(self, gray_img):
        """Detect face area to limit eye detection"""
        faces = self.face_cascade.detectMultiScale(
            gray_img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )
        
        if len(faces) > 0:
            # Return the largest face
            return max(faces, key=lambda rect: rect[2] * rect[3])
        return None

    def detect_eyes_in_face(self, gray_img, face_rect):
        """Detect eyes within face region with improved parameters"""
        x, y, w, h = face_rect
        
        # Define the upper half of face for eye detection
        upper_face_y = int(y + h * 0.1)  # Start 10% down from top
        upper_face_h = int(h * 0.4)      # Use only top 40% of face
        
        face_roi = gray_img[upper_face_y:upper_face_y + upper_face_h, x:x + w]
        
        # More strict eye detection parameters
        eyes = self.eye_cascade.detectMultiScale(
            face_roi,
            scaleFactor=1.05,
            minNeighbors=6,
            minSize=(int(w/8), int(w/8)),
            maxSize=(int(w/3), int(w/3))
        )
        
        if len(eyes) >= 2:
            # Adjust coordinates relative to original image
            eyes = [(ex + x, ey + upper_face_y, ew, eh) for ex, ey, ew, eh in eyes]
            
            # Sort by x-coordinate to separate left and right
            eyes = sorted(eyes, key=lambda x: x[0])
            
            # Take the leftmost and rightmost eyes if more than 2 detected
            left_eye = eyes[0]
            right_eye = eyes[-1]
            
            return left_eye, right_eye
        return None, None

    def calculate_eye_openness(self, gray_img, eye_rect):
        """Calculate eye openness using improved method"""
        if eye_rect is None:
            return None
            
        x, y, w, h = eye_rect
        eye_roi = gray_img[y:y+h, x:x+w]
        
        # Apply histogram equalization
        eye_roi = cv2.equalizeHist(eye_roi)
        
        # Calculate the average intensity in the eye region
        avg_intensity = np.mean(eye_roi)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            eye_roi,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        # Calculate ratio of white pixels
        white_ratio = np.sum(thresh == 255) / thresh.size
        
        # Combine intensity and ratio for final openness score
        openness = (white_ratio + (avg_intensity / 255)) / 2
        
        return float(openness)

    def detect_attributes(self, img):
        """Detect facial attributes in the image"""
        # Convert to grayscale for face/eye detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect face first
        face_rect = self.detect_face_area(gray)
        if face_rect is None:
            return None  # No face detected
            
        # Detect eyes within face region
        left_eye, right_eye = self.detect_eyes_in_face(gray, face_rect)
        
        # Run model inference
        input_tensor = self.preprocess_image(img)
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
        
        # Add individual eye data
        results['left_eye'] = {
            'bbox': left_eye,
            'openness': self.calculate_eye_openness(gray, left_eye) if left_eye is not None else None
        } if left_eye is not None else None
        
        results['right_eye'] = {
            'bbox': right_eye,
            'openness': self.calculate_eye_openness(gray, right_eye) if right_eye is not None else None
        } if right_eye is not None else None
        
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
        
        if attributes is not None:
            # Create display frame
            display_frame = frame.copy()
            
            # Display results
            y_pos = 30
            for attr, value in attributes.items():
                if attr in ['glasses', 'mask', 'sunglasses', 'eye_closeness']:
                    text = f"{attr}: {value:.2f}"
                    cv2.putText(display_frame, text, (10, y_pos), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_pos += 30
            
            # Draw eye regions and openness values
            for eye_name in ['left_eye', 'right_eye']:
                eye_data = attributes.get(eye_name)
                if eye_data and eye_data['bbox'] is not None:
                    x, y, w, h = eye_data['bbox']
                    openness = eye_data['openness']
                    
                    # Draw eye rectangle
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Draw openness value
                    if openness is not None:
                        cv2.putText(display_frame, 
                                  f"{eye_name} open: {openness:.2f}", 
                                  (x, y-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Face Attributes', display_frame)
        else:
            # If no face detected, show original frame
            cv2.imshow('Face Attributes', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()