import cv2
import numpy as np
import onnxruntime
import time

class FacePipeline:
    def __init__(self, face_model_path='face_det_lite.onnx', attr_model_path='face_attrib_net.onnx'):
        # Initialize models with optimizations
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4  # Adjust based on your CPU
        
        self.face_detector = onnxruntime.InferenceSession(
            face_model_path, 
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        self.attr_detector = onnxruntime.InferenceSession(
            attr_model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        
        self.face_input_name = self.face_detector.get_inputs()[0].name
        self.attr_input_name = self.attr_detector.get_inputs()[0].name
        
        # Cache preprocessed frames
        self.last_frame = None
        self.last_gray = None
        
        # Eye cascade with cached parameters
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Performance metrics
        self.fps = 0
        self.last_time = time.time()
        self.frame_count = 0

    def update_fps(self):
        self.frame_count += 1
        if self.frame_count >= 30:  # Update FPS every 30 frames
            current_time = time.time()
            self.fps = self.frame_count / (current_time - self.last_time)
            self.last_time = current_time
            self.frame_count = 0
        return self.fps

    def preprocess_face_detection(self, img):
        """Optimized preprocessing for face detection"""
        if img is self.last_frame:
            return self.last_gray
            
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        gray = cv2.resize(gray, (640, 480))
        processed = (gray.astype('float32') / 255.0 - 0.442) / 0.280
        processed = np.expand_dims(np.expand_dims(processed, axis=0), axis=0)
        
        self.last_frame = img
        self.last_gray = processed
        return processed

    def preprocess_attributes(self, img):
        """Optimized preprocessing for attribute detection"""
        img = cv2.resize(img, (128, 128))
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = img.astype('float32') / 255.0
        img = (img - 0.5) / 0.50196078
        img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
        return img

    def detect_faces(self, img, conf_threshold=0.55):
        """Optimized face detection"""
        input_tensor = self.preprocess_face_detection(img)
        outputs = self.face_detector.run(None, {self.face_input_name: input_tensor})
        
        heatmap, bbox_reg, _ = outputs
        heatmap = 1 / (1 + np.exp(-heatmap))
        
        confidence_mask = heatmap[0, 0] > conf_threshold
        y_indices, x_indices = np.where(confidence_mask)
        
        detections = []
        for y, x in zip(y_indices, x_indices):
            dx1, dy1, dx2, dy2 = bbox_reg[0, :, y, x]
            
            stride = 8
            x1 = int((x - dx1) * stride)
            y1 = int((y - dy1) * stride)
            w = int((dx1 + dx2) * stride)
            h = int((dy1 + dy2) * stride)
            
            # Add padding
            x1 = max(0, int(x1 - w * 0.05))
            y1 = max(0, int(y1 - h * 0.05))
            w = min(int(w * 1.1), 640 - x1)
            h = min(int(h * 1.1), 480 - y1)
            
            detections.append([x1, y1, w, h, float(heatmap[0, 0, y, x])])
            
        return detections

    def analyze_face(self, img, face_rect):
        """Analyze face attributes"""
        x, y, w, h = face_rect
        face_img = img[y:y+h, x:x+w]
        
        # Run attribute detection
        attr_input = self.preprocess_attributes(face_img)
        outputs = self.attr_detector.run(None, {self.attr_input_name: attr_input})
        
        # Process outputs
        results = {}
        for name, output in zip(['id_feature', 'liveness_feature', 'eye_closeness', 
                               'glasses', 'mask', 'sunglasses'], outputs):
            if name in ['eye_closeness', 'glasses', 'mask', 'sunglasses']:
                results[name] = float(1 / (1 + np.exp(-output[0][0])))
            else:
                results[name] = output[0]
        
        return results

    def process_frame(self, frame):
        """Process a single frame"""
        fps = self.update_fps()
        face_detections = self.detect_faces(frame)
        
        results = []
        for face_rect in face_detections[:1]:  # Process only the largest/closest face
            x, y, w, h, conf = face_rect
            attributes = self.analyze_face(frame, [x, y, w, h])
            results.append({
                'face_rect': face_rect,
                'attributes': attributes
            })
            
        return results, fps

def draw_results(frame, results, fps):
    """Draw results in corner of frame"""
    output = frame.copy()
    
    # Initialize text position in top-left corner
    text_x = 10
    text_y = 30
    
    # Display FPS
    cv2.putText(output, f"FPS: {fps:.1f}", (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    text_y += 25
    
    for result in results:
        face_rect = result['face_rect']
        attributes = result['attributes']
        x, y, w, h, conf = face_rect
        
        # Draw face rectangle only
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw all stats in corner
        cv2.putText(output, f"Face conf: {conf:.2f}", (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        text_y += 25
        
        # Draw attribute values
        for attr in ['eye_closeness', 'glasses', 'mask', 'sunglasses']:
            if attr in attributes:
                text = f"{attr}: {attributes[attr]:.2f}"
                cv2.putText(output, text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                text_y += 25
    
    return output

def main():
    pipeline = FacePipeline('face_det_lite.onnx', 'face_attrib_net.onnx')
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        results, fps = pipeline.process_frame(frame)
        
        # Draw results
        output_frame = draw_results(frame, results, fps)
        
        # Display
        cv2.imshow('Face Analysis', output_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()