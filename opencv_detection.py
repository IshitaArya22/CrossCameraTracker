import cv2
import numpy as np

class OpenCVPlayerDetector:
    """Player detection using OpenCV with Haar Cascades or simple methods"""
    
    def __init__(self, confidence_threshold=0.5):
        """
        Initialize the OpenCV player detector
        
        Args:
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.person_cascade = None
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self._load_cascades()
        
    def _load_cascades(self):
        """Load Haar cascade for person detection"""
        try:
            # Try to load person detection cascade
            if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
                cascade_path = cv2.data.haarcascades + 'haarcascade_fullbody.xml'
                self.person_cascade = cv2.CascadeClassifier(cascade_path)
                if not self.person_cascade.empty():
                    print("Loaded Haar cascade for person detection")
                else:
                    print("Haar cascade file found but failed to load")
                    self.person_cascade = None
            else:
                print("OpenCV data path not available")
                self.person_cascade = None
        except Exception as e:
            print(f"Could not load person cascade: {e}")
            self.person_cascade = None
    
    def detect(self, frame):
        """
        Detect players in a frame using multiple methods
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            List of detections with format [x1, y1, x2, y2, confidence]
        """
        detections = []
        
        # Method 1: Haar Cascade Detection
        if self.person_cascade is not None:
            cascade_detections = self._detect_with_cascade(frame)
            detections.extend(cascade_detections)
        
        # Method 2: Background Subtraction (for moving objects)
        motion_detections = self._detect_with_motion(frame)
        detections.extend(motion_detections)
        
        # Remove overlapping detections
        detections = self._non_max_suppression(detections)
        
        # Limit to realistic number of soccer players (max 22 players on field)
        if len(detections) > 22:
            # Sort by confidence and keep top 22
            detections.sort(key=lambda x: x[4], reverse=True)
            detections = detections[:22]
        
        return np.array(detections) if detections else np.empty((0, 5))
    
    def _detect_with_cascade(self, frame):
        """Detect using Haar cascade"""
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect persons
        persons = self.person_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        for (x, y, w, h) in persons:
            confidence = 0.8  # Fixed confidence for cascade detection
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            if self._is_valid_detection(x1, y1, x2, y2, frame.shape):
                detections.append([x1, y1, x2, y2, confidence])
        
        return detections
    
    def _detect_with_motion(self, frame):
        """Detect moving objects using background subtraction"""
        detections = []
        
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area - focus on player-sized objects on field
            if 2000 < area < 25000:  # Stricter area filtering for field players
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio (soccer players standing/running)
                aspect_ratio = h / w if w > 0 else 0
                if 1.8 < aspect_ratio < 3.5:  # Tighter aspect ratio for players
                    x1, y1, x2, y2 = x, y, x + w, y + h
                    
                    # Additional filter: players should be in field area (not stands)
                    frame_height = frame.shape[0]
                    # Focus on center 80% of frame height (field area)
                    field_top = int(frame_height * 0.1)
                    field_bottom = int(frame_height * 0.9)
                    
                    if field_top < y1 < field_bottom and field_top < y2 < field_bottom:
                        confidence = min(0.8, area / 15000)  # Higher confidence for field detections
                        
                        if self._is_valid_detection(x1, y1, x2, y2, frame.shape):
                            detections.append([x1, y1, x2, y2, confidence])
        
        return detections
    
    def _is_valid_detection(self, x1, y1, x2, y2, frame_shape):
        """Validate detection bounding box"""
        height, width = frame_shape[:2]
        
        # Check if coordinates are within frame bounds
        if x1 < 0 or y1 < 0 or x2 >= width or y2 >= height:
            return False
            
        # Check if box has reasonable dimensions
        box_width = x2 - x1
        box_height = y2 - y1
        
        # Minimum and maximum size constraints
        min_size = 20
        max_width_ratio = 0.5
        max_height_ratio = 0.8
        
        if (box_width < min_size or box_height < min_size or
            box_width > width * max_width_ratio or 
            box_height > height * max_height_ratio):
            return False
            
        # Check aspect ratio
        aspect_ratio = box_height / box_width
        if aspect_ratio < 0.8 or aspect_ratio > 5.0:
            return False
            
        return True
    
    def _non_max_suppression(self, detections, overlap_threshold=0.3):
        """Remove overlapping detections"""
        if not detections:
            return []
        
        detections = np.array(detections)
        
        # Calculate areas
        x1 = detections[:, 0]
        y1 = detections[:, 1]
        x2 = detections[:, 2]
        y2 = detections[:, 3]
        scores = detections[:, 4]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        # Sort by score
        indices = np.argsort(scores)[::-1]
        
        keep = []
        while len(indices) > 0:
            # Pick the detection with highest score
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            # Calculate IoU with remaining detections
            other_indices = indices[1:]
            
            xx1 = np.maximum(x1[current], x1[other_indices])
            yy1 = np.maximum(y1[current], y1[other_indices])
            xx2 = np.minimum(x2[current], x2[other_indices])
            yy2 = np.minimum(y2[current], y2[other_indices])
            
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            intersection = w * h
            union = areas[current] + areas[other_indices] - intersection
            iou = intersection / union
            
            # Keep detections with IoU below threshold
            indices = other_indices[iou <= overlap_threshold]
        
        return detections[keep].tolist()
    
    def visualize_detections(self, frame, detections):
        """Draw detection bounding boxes on frame"""
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