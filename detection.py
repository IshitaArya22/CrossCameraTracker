import cv2
import numpy as np
import urllib.request
import os

class PlayerDetector:
    """Player detection using OpenCV DNN with YOLOv4"""
    
    def __init__(self, confidence_threshold=0.5):
        """
        Initialize the player detector
        
        Args:
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.person_class_id = 0  # 'person' class in COCO dataset
        self.net = None
        self.output_layers = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv3 model using OpenCV DNN"""
        try:
            # Use YOLOv3 which is more compatible and smaller
            weights_url = "https://pjreddie.com/media/files/yolov3.weights"
            config_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
            
            weights_path = "yolov3.weights"
            config_path = "yolov3.cfg"
            
            if not os.path.exists(weights_path):
                print("Downloading YOLOv3 weights (this may take a while)...")
                try:
                    urllib.request.urlretrieve(weights_url, weights_path)
                except Exception as e:
                    print(f"Failed to download weights: {e}")
                    self.net = None
                    return
            
            if not os.path.exists(config_path):
                print("Downloading YOLOv3 config...")
                try:
                    urllib.request.urlretrieve(config_url, config_path)
                except Exception as e:
                    print(f"Failed to download config: {e}")
                    self.net = None
                    return
            
            # Load the network
            self.net = cv2.dnn.readNet(weights_path, config_path)
            
            # Get output layer names
            layer_names = self.net.getLayerNames()
            output_layer_indices = self.net.getUnconnectedOutLayers()
            # Handle different OpenCV versions
            if isinstance(output_layer_indices, np.ndarray):
                if output_layer_indices.ndim > 1:
                    output_layer_indices = output_layer_indices.flatten()
                self.output_layers = [layer_names[i - 1] for i in output_layer_indices]
            else:
                self.output_layers = [layer_names[i - 1] for i in output_layer_indices]
            
            print("YOLOv3 model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # Use simpler detection method as fallback
            self.net = None
        
    def detect(self, frame):
        """
        Detect players in a frame
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            List of detections with format [x1, y1, x2, y2, confidence]
        """
        detections = []
        
        if self.net is None:
            # Fallback detection using simple blob detection
            return self._simple_detection(frame)
        
        try:
            # Get frame dimensions
            height, width = frame.shape[:2]
            
            # Create blob from frame
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            
            # Set input to the network
            self.net.setInput(blob)
            
            # Run forward pass
            outputs = self.net.forward(self.output_layers)
            
            # Process outputs
            boxes = []
            confidences = []
            class_ids = []
            
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    # Filter for person class and confidence threshold
                    if class_id == self.person_class_id and confidence >= self.confidence_threshold:
                        # Get bounding box coordinates
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        x1 = int(center_x - w / 2)
                        y1 = int(center_y - h / 2)
                        x2 = int(center_x + w / 2)
                        y2 = int(center_y + h / 2)
                        
                        # Validate bounding box
                        if self._is_valid_detection(x1, y1, x2, y2, frame.shape):
                            boxes.append([x1, y1, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)
            
            # Apply non-maximum suppression
            if boxes:
                indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.4)
                
                if len(indices) > 0:
                    for i in indices.flatten():
                        x1, y1, w, h = boxes[i]
                        x2, y2 = x1 + w, y1 + h
                        confidence = confidences[i]
                        detections.append([x1, y1, x2, y2, confidence])
            
        except Exception as e:
            print(f"Detection error: {e}")
            # Fallback to simple detection
            return self._simple_detection(frame)
        
        return np.array(detections) if detections else np.empty((0, 5))
    
    def _simple_detection(self, frame):
        """Simple fallback detection using background subtraction and contours"""
        # This is a very basic fallback that tries to detect moving objects
        # In a real application, you'd want more sophisticated detection
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Use adaptive threshold to find potential objects
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (assuming players have a reasonable size)
            if 500 < area < 50000:  # Adjust these values based on your video resolution
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio (people are usually taller than wide)
                aspect_ratio = h / w if w > 0 else 0
                if 1.2 < aspect_ratio < 4.0:
                    x1, y1, x2, y2 = x, y, x + w, y + h
                    confidence = 0.6  # Fixed confidence for simple detection
                    
                    if self._is_valid_detection(x1, y1, x2, y2, frame.shape):
                        detections.append([x1, y1, x2, y2, confidence])
        
        return np.array(detections) if detections else np.empty((0, 5))
    
    def _is_valid_detection(self, x1, y1, x2, y2, frame_shape):
        """
        Validate detection bounding box
        
        Args:
            x1, y1, x2, y2: Bounding box coordinates
            frame_shape: Shape of the input frame
            
        Returns:
            Boolean indicating if detection is valid
        """
        height, width = frame_shape[:2]
        
        # Check if coordinates are within frame bounds
        if x1 < 0 or y1 < 0 or x2 >= width or y2 >= height:
            return False
            
        # Check if box has reasonable dimensions
        box_width = x2 - x1
        box_height = y2 - y1
        
        # Minimum and maximum size constraints
        min_size = 20
        max_width_ratio = 0.8
        max_height_ratio = 0.9
        
        if (box_width < min_size or box_height < min_size or
            box_width > width * max_width_ratio or 
            box_height > height * max_height_ratio):
            return False
            
        # Check aspect ratio (humans are typically taller than wide)
        aspect_ratio = box_height / box_width
        if aspect_ratio < 0.5 or aspect_ratio > 5.0:
            return False
            
        return True
    
    def visualize_detections(self, frame, detections):
        """
        Draw detection bounding boxes on frame
        
        Args:
            frame: Input frame
            detections: Array of detections
            
        Returns:
            Frame with drawn bounding boxes
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2, conf = detection
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence score
            label = f"Player: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated_frame
