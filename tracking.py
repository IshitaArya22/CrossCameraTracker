import cv2
import numpy as np
from scipy.spatial.distance import cdist
from collections import OrderedDict

class PlayerTracker:
    """
    Multi-object tracker based on Deep SORT principles
    Simplified implementation using Kalman filters and Hungarian algorithm
    """
    
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        Initialize the tracker
        
        Args:
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum detections to confirm track
            iou_threshold: IoU threshold for association
        """
        self.max_age = max_age
        self.min_hits = max(1, min_hits)  # At least 1 hit to confirm track
        self.iou_threshold = iou_threshold
        self.tracks = OrderedDict()
        self.next_id = 1
        self.max_players = 22  # Limit to realistic number of soccer players
        
    def update(self, detections, frame):
        """
        Update tracks with new detections
        
        Args:
            detections: Array of detections [x1, y1, x2, y2, conf]
            frame: Current frame for feature extraction
            
        Returns:
            List of active tracks
        """
        # Predict next positions for existing tracks
        self._predict()
        
        # Associate detections with existing tracks
        matches, unmatched_dets, unmatched_trks = self._associate(detections)
        
        # Update matched tracks
        for m in matches:
            self.tracks[m[1]].update(detections[m[0]], frame)
        
        # Create new tracks for unmatched detections (limit to max_players)
        for det_idx in unmatched_dets:
            if len(self.tracks) < self.max_players:
                self._create_track(detections[det_idx], frame)
        
        # Mark unmatched tracks for deletion
        for trk_idx in unmatched_trks:
            self.tracks[trk_idx].mark_missed()
        
        # Remove dead tracks
        self._remove_dead_tracks()
        
        # Return active tracks
        return self._get_active_tracks()
    
    def _predict(self):
        """Predict next positions for all tracks"""
        for track in self.tracks.values():
            track.predict()
    
    def _associate(self, detections):
        """Associate detections with tracks using IoU"""
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        
        # Compute IoU matrix
        track_boxes = np.array([track.get_state()[:4] for track in self.tracks.values()])
        det_boxes = detections[:, :4]
        
        iou_matrix = self._compute_iou_matrix(det_boxes, track_boxes)
        
        # Perform Hungarian assignment
        matches, unmatched_dets, unmatched_trks = self._hungarian_assignment(
            iou_matrix, self.iou_threshold
        )
        
        # Convert track indices to track IDs
        track_ids = list(self.tracks.keys())
        matches = [(m[0], track_ids[m[1]]) for m in matches]
        unmatched_trks = [track_ids[i] for i in unmatched_trks]
        
        return matches, unmatched_dets, unmatched_trks
    
    def _compute_iou_matrix(self, det_boxes, track_boxes):
        """Compute IoU matrix between detections and tracks"""
        if len(det_boxes) == 0 or len(track_boxes) == 0:
            return np.empty((0, 0))
        
        # Compute intersection areas
        inter_x1 = np.maximum(det_boxes[:, None, 0], track_boxes[None, :, 0])
        inter_y1 = np.maximum(det_boxes[:, None, 1], track_boxes[None, :, 1])
        inter_x2 = np.minimum(det_boxes[:, None, 2], track_boxes[None, :, 2])
        inter_y2 = np.minimum(det_boxes[:, None, 3], track_boxes[None, :, 3])
        
        inter_w = np.maximum(0, inter_x2 - inter_x1)
        inter_h = np.maximum(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        
        # Compute areas
        det_area = (det_boxes[:, 2] - det_boxes[:, 0]) * (det_boxes[:, 3] - det_boxes[:, 1])
        track_area = (track_boxes[:, 2] - track_boxes[:, 0]) * (track_boxes[:, 3] - track_boxes[:, 1])
        
        # Compute IoU
        union_area = det_area[:, None] + track_area[None, :] - inter_area
        iou_matrix = inter_area / (union_area + 1e-10)
        
        return iou_matrix
    
    def _hungarian_assignment(self, cost_matrix, threshold):
        """Simplified Hungarian assignment"""
        matches = []
        unmatched_dets = list(range(cost_matrix.shape[0]))
        unmatched_trks = list(range(cost_matrix.shape[1]))
        
        if cost_matrix.size == 0:
            return matches, unmatched_dets, unmatched_trks
        
        # Simple greedy assignment
        while True:
            # Find maximum IoU
            max_iou = np.max(cost_matrix)
            if max_iou < threshold:
                break
            
            # Find indices of maximum IoU
            det_idx, trk_idx = np.unravel_index(np.argmax(cost_matrix), cost_matrix.shape)
            
            # Add to matches
            matches.append((det_idx, trk_idx))
            
            # Remove from unmatched lists
            unmatched_dets.remove(det_idx)
            unmatched_trks.remove(trk_idx)
            
            # Set row and column to zero
            cost_matrix[det_idx, :] = 0
            cost_matrix[:, trk_idx] = 0
        
        return matches, unmatched_dets, unmatched_trks
    
    def _create_track(self, detection, frame):
        """Create new track from detection"""
        track_id = self.next_id
        self.next_id += 1
        
        self.tracks[track_id] = Track(track_id, detection, frame)
    
    def _remove_dead_tracks(self):
        """Remove tracks that have been inactive too long"""
        dead_tracks = []
        for track_id, track in self.tracks.items():
            if track.time_since_update > self.max_age:
                dead_tracks.append(track_id)
        
        for track_id in dead_tracks:
            del self.tracks[track_id]
    
    def _get_active_tracks(self):
        """Get list of active tracks"""
        active_tracks = []
        for track in self.tracks.values():
            if track.hit_streak >= self.min_hits or track.time_since_update == 0:
                state = track.get_state()
                active_tracks.append({
                    'id': track.track_id,
                    'bbox': state[:4],
                    'confidence': state[4] if len(state) > 4 else 1.0
                })
        
        return active_tracks

class Track:
    """Individual track for a player"""
    
    def __init__(self, track_id, detection, frame):
        """Initialize track with first detection"""
        self.track_id = track_id
        self.hit_streak = 1
        self.time_since_update = 0
        
        # Initialize Kalman filter state [x, y, w, h, vx, vy, vw, vh]
        x1, y1, x2, y2 = detection[:4]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        
        self.state = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)
        self.confidence = detection[4] if len(detection) > 4 else 1.0
        
        # History for features
        self.feature_history = []
        
    def predict(self):
        """Predict next state using simple motion model"""
        # Update position based on velocity
        self.state[0] += self.state[4]  # x += vx
        self.state[1] += self.state[5]  # y += vy
        self.state[2] += self.state[6]  # w += vw
        self.state[3] += self.state[7]  # h += vh
        
        # Increase time since update
        self.time_since_update += 1
    
    def update(self, detection, frame):
        """Update track with new detection"""
        self.time_since_update = 0
        self.hit_streak += 1
        
        # Extract new state
        x1, y1, x2, y2 = detection[:4]
        new_cx = (x1 + x2) / 2
        new_cy = (y1 + y2) / 2
        new_w = x2 - x1
        new_h = y2 - y1
        
        # Update velocity
        self.state[4] = new_cx - self.state[0]  # vx
        self.state[5] = new_cy - self.state[1]  # vy
        self.state[6] = new_w - self.state[2]   # vw
        self.state[7] = new_h - self.state[3]   # vh
        
        # Update position
        self.state[0] = new_cx
        self.state[1] = new_cy
        self.state[2] = new_w
        self.state[3] = new_h
        
        # Update confidence
        self.confidence = detection[4] if len(detection) > 4 else 1.0
    
    def mark_missed(self):
        """Mark track as missed"""
        self.hit_streak = 0
    
    def get_state(self):
        """Get current state as bounding box"""
        cx, cy, w, h = self.state[:4]
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        
        return np.array([x1, y1, x2, y2, self.confidence])
