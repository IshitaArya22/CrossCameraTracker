import cv2
import numpy as np
import tempfile
import os
from typing import Dict, List, Any

class VideoProcessor:
    """Utility class for video processing operations"""
    
    @staticmethod
    def get_video_info(video_path):
        """
        Get basic information about a video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info
    
    @staticmethod
    def extract_frames(video_path, max_frames=None, skip_frames=1):
        """
        Extract frames from video
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            skip_frames: Number of frames to skip between extractions
            
        Returns:
            List of frames
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % skip_frames == 0:
                frames.append(frame.copy())
                extracted_count += 1
                
                if max_frames and extracted_count >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        return frames

def save_video_with_annotations(video_path, video_data, mapping_results, view_type):
    """
    Save video with player ID annotations
    
    Args:
        video_path: Path to original video
        video_data: Processed video data with tracks
        mapping_results: Player mapping results
        view_type: 'broadcast' or 'tactical'
        
    Returns:
        Path to annotated video file
    """
    # Create temporary output file
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    
    # Open original video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Get consistent ID mapping
    from player_mapping import PlayerMapper
    mapper = PlayerMapper()
    id_mapping = mapper.assign_consistent_ids(mapping_results)
    
    # Process each frame
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Find tracks for current frame
        current_frame_data = None
        for frame_data in video_data['frame_data']:
            if frame_data['frame_number'] == frame_count:
                current_frame_data = frame_data
                break
        
        if current_frame_data:
            # Draw annotations for each track
            for track in current_frame_data['tracks']:
                original_id = track['id']
                bbox = track['bbox']
                
                # Get consistent ID
                if view_type == 'broadcast':
                    consistent_id = id_mapping['broadcast_to_consistent'].get(original_id)
                else:
                    consistent_id = id_mapping['tactical_to_consistent'].get(original_id)
                
                if consistent_id:
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw player ID
                    label = f"Player {consistent_id}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    
                    # Background for text
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                 (x1 + label_size[0], y1), (0, 255, 0), -1)
                    
                    # Text
                    cv2.putText(frame, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Write frame
        out.write(frame)
        frame_count += 1
    
    # Clean up
    cap.release()
    out.release()
    
    return output_path

def calculate_frame_similarity(frame1, frame2):
    """
    Calculate similarity between two frames
    
    Args:
        frame1, frame2: Input frames
        
    Returns:
        Similarity score between 0 and 1
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Resize to same size if different
    if gray1.shape != gray2.shape:
        h, w = min(gray1.shape[0], gray2.shape[0]), min(gray1.shape[1], gray2.shape[1])
        gray1 = cv2.resize(gray1, (w, h))
        gray2 = cv2.resize(gray2, (w, h))
    
    # Calculate structural similarity
    diff = cv2.absdiff(gray1, gray2)
    mean_diff = np.mean(diff)
    
    # Normalize to 0-1 range
    similarity = 1 - (mean_diff / 255)
    
    return max(0, min(1, similarity))

def validate_video_file(file_path):
    """
    Validate if file is a proper video file
    
    Args:
        file_path: Path to video file
        
    Returns:
        Boolean indicating if file is valid
    """
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return False
        
        # Try to read first frame
        ret, frame = cap.read()
        cap.release()
        
        return ret and frame is not None
    except:
        return False

def create_summary_report(mapping_results, broadcast_data, tactical_data):
    """
    Create a summary report of the processing results
    
    Args:
        mapping_results: Player mapping results
        broadcast_data: Processed broadcast data
        tactical_data: Processed tactical data
        
    Returns:
        Dictionary with summary information
    """
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'broadcast_stats': {
            'total_frames': broadcast_data['total_frames'],
            'detected_players': len(broadcast_data['player_features']),
            'player_ids': list(broadcast_data['player_features'].keys())
        },
        'tactical_stats': {
            'total_frames': tactical_data['total_frames'],
            'detected_players': len(tactical_data['player_features']),
            'player_ids': list(tactical_data['player_features'].keys())
        },
        'mapping_stats': {
            'successful_mappings': len(mapping_results['mappings']),
            'unmapped_broadcast': len(mapping_results['unmapped_broadcast']),
            'unmapped_tactical': len(mapping_results['unmapped_tactical']),
            'average_similarity': np.mean(list(mapping_results['mappings'].values())) if mapping_results['mappings'] else 0
        }
    }
    
    return report
