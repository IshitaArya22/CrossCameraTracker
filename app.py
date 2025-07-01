import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import os
import json

# Try to import the full computer vision modules, fallback to simple versions
try:
    import cv2
    # Try the OpenCV-based detector first, fallback to complex YOLO if needed
    try:
        from opencv_detection import OpenCVPlayerDetector as PlayerDetector
        print("Using OpenCV-based player detection")
    except ImportError:
        from detection import PlayerDetector
        print("Using YOLO-based player detection")
    CV_AVAILABLE = True
except ImportError:
    from simple_detection import SimplePlayerDetector as PlayerDetector
    CV_AVAILABLE = False
    print("Using simplified detection - OpenCV not available")

# Import other modules with fallbacks
try:
    from tracking import PlayerTracker
    from feature_extraction import FeatureExtractor
    from player_mapping import PlayerMapper
    from utils import VideoProcessor, save_video_with_annotations
    FULL_CV_AVAILABLE = True
except ImportError:
    from simple_modules import (SimplePlayerTracker as PlayerTracker, 
                               SimpleFeatureExtractor as FeatureExtractor,
                               SimplePlayerMapper as PlayerMapper,
                               SimpleVideoProcessor as VideoProcessor,
                               simple_save_video_with_annotations as save_video_with_annotations)
    FULL_CV_AVAILABLE = False

def main():
    st.title("⚽ Cross-Camera Player Mapping System")
    st.markdown("Track and map soccer players consistently across multiple camera views")
    
    # Display system status only if in demo mode
    if not FULL_CV_AVAILABLE:
        st.warning("⚠️ **Demo Mode**: Some computer vision libraries are not available. "
                  "The app is running in demonstration mode with simplified functionality. "
                  "For full functionality, additional setup may be required.")
        
        st.info("📋 **What works in demo mode:**\n"
               "- Interface demonstration\n"
               "- Algorithm structure\n"
               "- Results visualization\n"
               "- Basic player mapping logic")
    
    # Use optimized default settings for best performance
    confidence_threshold = 0.5
    max_tracking_age = 30
    similarity_threshold = 0.7
    
    # File upload section
    st.header("Upload Videos")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Broadcast Camera")
        broadcast_video = st.file_uploader(
            "Upload broadcast video", 
            type=['mp4', 'avi', 'mov'], 
            key="broadcast"
        )
        
    with col2:
        st.subheader("Tactical Camera")
        tactical_video = st.file_uploader(
            "Upload tactical video", 
            type=['mp4', 'avi', 'mov'], 
            key="tactical"
        )
    
    if broadcast_video and tactical_video:
        if st.button("Process Videos", type="primary"):
            process_videos(broadcast_video, tactical_video, confidence_threshold, 
                          max_tracking_age, similarity_threshold)

def process_videos(broadcast_video, tactical_video, confidence_threshold, 
                  max_tracking_age, similarity_threshold):
    """Process both videos and perform player mapping"""
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Save uploaded files temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_broadcast:
            tmp_broadcast.write(broadcast_video.read())
            broadcast_path = tmp_broadcast.name
            
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_tactical:
            tmp_tactical.write(tactical_video.read())
            tactical_path = tmp_tactical.name
        
        # Initialize components
        status_text.text("Initializing detection and tracking components...")
        detector = PlayerDetector(confidence_threshold=confidence_threshold)
        tracker_broadcast = PlayerTracker(max_age=max_tracking_age)
        tracker_tactical = PlayerTracker(max_age=max_tracking_age)
        feature_extractor = FeatureExtractor()
        mapper = PlayerMapper(similarity_threshold=similarity_threshold)
        
        progress_bar.progress(10)
        
        # Process broadcast video
        status_text.text("Processing broadcast video...")
        broadcast_data = process_single_video(
            broadcast_path, detector, tracker_broadcast, feature_extractor
        )
        progress_bar.progress(40)
        
        # Process tactical video
        status_text.text("Processing tactical video...")
        tactical_data = process_single_video(
            tactical_path, detector, tracker_tactical, feature_extractor
        )
        progress_bar.progress(70)
        
        # Perform player mapping
        status_text.text("Mapping players across views...")
        mapping_results = mapper.map_players(broadcast_data, tactical_data)
        progress_bar.progress(90)
        
        # Display results
        status_text.text("Generating visualizations...")
        display_results(mapping_results, broadcast_data, tactical_data)
        
        # Generate annotated videos
        generate_annotated_videos(
            broadcast_path, tactical_path, broadcast_data, tactical_data, 
            mapping_results, detector
        )
        
        progress_bar.progress(100)
        status_text.text("Processing complete!")
        
    except Exception as e:
        st.error(f"Error processing videos: {str(e)}")
    finally:
        # Clean up temporary files
        try:
            os.unlink(broadcast_path)
            os.unlink(tactical_path)
        except:
            pass

def process_single_video(video_path, detector, tracker, feature_extractor):
    """Process a single video for detection, tracking, and feature extraction"""
    
    frame_data = []
    player_features = {}
    
    if CV_AVAILABLE:
        # Full OpenCV processing
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect players
            detections = detector.detect(frame)
            
            # Update tracker
            tracks = tracker.update(detections, frame)
            
            # Extract features for each tracked player
            for track in tracks:
                player_id = track['id']
                bbox = track['bbox']
                
                # Crop player image
                x1, y1, x2, y2 = map(int, bbox)
                player_crop = frame[y1:y2, x1:x2]
                
                if player_crop.size > 0:
                    # Extract features
                    features = feature_extractor.extract_features(player_crop)
                    
                    if player_id not in player_features:
                        player_features[player_id] = []
                    player_features[player_id].append(features)
            
            # Store frame data
            frame_data.append({
                'frame_number': frame_count,
                'tracks': tracks,
                'detections': detections
            })
            
            frame_count += 1
        
        cap.release()
    else:
        # Simplified processing without OpenCV
        st.info("Using simplified processing mode - full computer vision features require additional setup")
        
        # Simulate processing frames
        frame_count = 100  # Assume 100 frames for demo
        
        for i in range(10):  # Process 10 sample frames
            # Create dummy frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Detect players (simplified)
            detections = detector.detect(frame)
            
            # Update tracker
            tracks = tracker.update(detections, frame)
            
            # Extract features for each tracked player
            for track in tracks:
                player_id = track['id']
                
                # Generate dummy crop
                player_crop = np.random.randint(0, 255, (64, 32, 3), dtype=np.uint8)
                
                # Extract features
                features = feature_extractor.extract_features(player_crop)
                
                if player_id not in player_features:
                    player_features[player_id] = []
                player_features[player_id].append(features)
            
            # Store frame data
            frame_data.append({
                'frame_number': i,
                'tracks': tracks,
                'detections': detections
            })
    
    # Average features for each player
    averaged_features = {}
    for player_id, features_list in player_features.items():
        if features_list:
            averaged_features[player_id] = np.mean(features_list, axis=0)
    
    return {
        'frame_data': frame_data,
        'player_features': averaged_features,
        'total_frames': frame_count
    }

def display_results(mapping_results, broadcast_data, tactical_data):
    """Display player mapping with consistent IDs across both feeds"""
    
    st.header("Cross-Camera Player Mapping")
    
    # Get unique players from each camera
    broadcast_players = list(broadcast_data['player_features'].keys())
    tactical_players = list(tactical_data['player_features'].keys())
    
    st.markdown(f"**Broadcast Camera**: {len(broadcast_players)} unique players detected")
    st.markdown(f"**Tactical Camera**: {len(tactical_players)} unique players detected")
    
    if mapping_results['mappings']:
        st.success(f"Successfully matched {len(mapping_results['mappings'])} players across both camera feeds!")
        
        # Generate consistent player IDs
        mapper = PlayerMapper()
        consistent_ids = mapper.assign_consistent_ids(mapping_results)
        
        st.subheader("Player Matching Results")
        st.markdown("Each tactical camera player matched to their corresponding identity in the broadcast camera:")
        
        # Create mapping table showing tactical to broadcast correspondence
        mapping_data = []
        consistent_player_num = 1
        
        for (broadcast_id, tactical_id), similarity in mapping_results['mappings'].items():
            mapping_data.append({
                'Consistent Player ID': f"Player {consistent_player_num}",
                'Tactical Camera ID': f"T{tactical_id}",
                'Broadcast Camera ID': f"B{broadcast_id}",
                'Match Confidence': f"{similarity:.1%}",
                'Status': "Successfully Matched"
            })
            consistent_player_num += 1
        
        # Add unmapped tactical players
        for tactical_id in mapping_results.get('unmapped_tactical', []):
            mapping_data.append({
                'Consistent Player ID': f"Player T{tactical_id}",
                'Tactical Camera ID': f"T{tactical_id}",
                'Broadcast Camera ID': "No Match Found",
                'Match Confidence': "N/A",
                'Status': "Only in Tactical"
            })
            
        # Add unmapped broadcast players
        for broadcast_id in mapping_results.get('unmapped_broadcast', []):
            mapping_data.append({
                'Consistent Player ID': f"Player B{broadcast_id}",
                'Tactical Camera ID': "No Match Found",
                'Broadcast Camera ID': f"B{broadcast_id}",
                'Match Confidence': "N/A",
                'Status': "Only in Broadcast"
            })
        
        mapping_df = pd.DataFrame(mapping_data)
        st.dataframe(mapping_df, use_container_width=True)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Successfully Matched", len(mapping_results['mappings']))
        with col2:
            tactical_only = len(mapping_results.get('unmapped_tactical', []))
            st.metric("Tactical Only", tactical_only)
        with col3:
            broadcast_only = len(mapping_results.get('unmapped_broadcast', []))
            st.metric("Broadcast Only", broadcast_only)
        
        # Match success rate
        total_tactical_players = len(tactical_players)
        matched_players = len(mapping_results['mappings'])
        match_rate = (matched_players / total_tactical_players * 100) if total_tactical_players > 0 else 0
        
        st.info(f"Match Success: {matched_players} out of {total_tactical_players} tactical camera players successfully matched to broadcast camera ({match_rate:.1f}%)")
        
        # Create consistent ID mapping for download
        consistent_mapping = {}
        player_num = 1
        for (broadcast_id, tactical_id), similarity in mapping_results['mappings'].items():
            consistent_mapping[f"Player_{player_num}"] = {
                "tactical_camera_id": tactical_id,
                "broadcast_camera_id": broadcast_id,
                "match_confidence": similarity,
                "status": "matched"
            }
            player_num += 1
        
        # Download results
        st.download_button(
            label="Download Player Mapping Results",
            data=json.dumps(consistent_mapping, indent=2),
            file_name="player_cross_camera_mapping.json",
            mime="application/json"
        )
        
    else:
        st.warning("No players could be matched between the two camera feeds.")
        st.markdown("Possible reasons:")
        st.markdown("- Different players visible in each camera")
        st.markdown("- Poor video quality affecting feature extraction")
        st.markdown("- Significant lighting or angle differences between cameras")

def generate_annotated_videos(broadcast_path, tactical_path, broadcast_data, 
                            tactical_data, mapping_results, detector):
    """Generate annotated videos with consistent player IDs"""
    
    st.subheader("Annotated Videos")
    
    with st.spinner("Generating annotated videos..."):
        # Generate broadcast annotated video
        broadcast_annotated = save_video_with_annotations(
            broadcast_path, broadcast_data, mapping_results, 'broadcast'
        )
        
        # Generate tactical annotated video  
        tactical_annotated = save_video_with_annotations(
            tactical_path, tactical_data, mapping_results, 'tactical'
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Broadcast Video (Annotated)**")
            if os.path.exists(broadcast_annotated):
                with open(broadcast_annotated, 'rb') as f:
                    st.download_button(
                        label="Download Broadcast Video",
                        data=f.read(),
                        file_name="broadcast_annotated.mp4",
                        mime="video/mp4"
                    )
        
        with col2:
            st.write("**Tactical Video (Annotated)**")
            if os.path.exists(tactical_annotated):
                with open(tactical_annotated, 'rb') as f:
                    st.download_button(
                        label="Download Tactical Video", 
                        data=f.read(),
                        file_name="tactical_annotated.mp4",
                        mime="video/mp4"
                    )

if __name__ == "__main__":
    main()
