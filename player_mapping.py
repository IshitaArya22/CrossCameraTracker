import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

class PlayerMapper:
    """Map players across different camera views using feature similarity"""
    
    def __init__(self, similarity_threshold=0.7, max_distance=0.5):
        """
        Initialize player mapper
        
        Args:
            similarity_threshold: Minimum similarity score for mapping
            max_distance: Maximum distance for valid mapping
        """
        self.similarity_threshold = similarity_threshold
        self.max_distance = max_distance
    
    def map_players(self, broadcast_data, tactical_data):
        """
        Map players between broadcast and tactical camera views
        
        Args:
            broadcast_data: Processed data from broadcast camera
            tactical_data: Processed data from tactical camera
            
        Returns:
            Dictionary containing mapping results
        """
        broadcast_features = broadcast_data['player_features']
        tactical_features = tactical_data['player_features']
        
        if not broadcast_features or not tactical_features:
            return {
                'mappings': {},
                'unmapped_broadcast': list(broadcast_features.keys()),
                'unmapped_tactical': list(tactical_features.keys()),
                'similarity_matrix': np.array([])
            }
        
        # Extract player IDs and feature vectors
        broadcast_ids = list(broadcast_features.keys())
        tactical_ids = list(tactical_features.keys())
        
        broadcast_vectors = np.array([broadcast_features[pid] for pid in broadcast_ids])
        tactical_vectors = np.array([tactical_features[pid] for pid in tactical_ids])
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(broadcast_vectors, tactical_vectors)
        
        # Find optimal assignments using Hungarian algorithm
        mappings = self._find_optimal_mappings(
            similarity_matrix, broadcast_ids, tactical_ids
        )
        
        # Filter mappings by threshold
        filtered_mappings = {}
        mapped_broadcast = set()
        mapped_tactical = set()
        
        for (broadcast_id, tactical_id), score in mappings.items():
            if score >= self.similarity_threshold:
                filtered_mappings[(broadcast_id, tactical_id)] = score
                mapped_broadcast.add(broadcast_id)
                mapped_tactical.add(tactical_id)
        
        # Find unmapped players
        unmapped_broadcast = [pid for pid in broadcast_ids if pid not in mapped_broadcast]
        unmapped_tactical = [pid for pid in tactical_ids if pid not in mapped_tactical]
        
        return {
            'mappings': filtered_mappings,
            'unmapped_broadcast': unmapped_broadcast,
            'unmapped_tactical': unmapped_tactical,
            'similarity_matrix': similarity_matrix,
            'broadcast_ids': broadcast_ids,
            'tactical_ids': tactical_ids
        }
    
    def _find_optimal_mappings(self, similarity_matrix, broadcast_ids, tactical_ids):
        """
        Find optimal player mappings using Hungarian algorithm
        
        Args:
            similarity_matrix: Cosine similarity matrix
            broadcast_ids: List of broadcast player IDs
            tactical_ids: List of tactical player IDs
            
        Returns:
            Dictionary of mappings with similarity scores
        """
        # Convert similarity to cost (Hungarian algorithm minimizes cost)
        cost_matrix = 1 - similarity_matrix
        
        # Apply Hungarian algorithm
        broadcast_indices, tactical_indices = linear_sum_assignment(cost_matrix)
        
        # Create mappings dictionary
        mappings = {}
        for b_idx, t_idx in zip(broadcast_indices, tactical_indices):
            broadcast_id = broadcast_ids[b_idx]
            tactical_id = tactical_ids[t_idx]
            similarity_score = similarity_matrix[b_idx, t_idx]
            
            mappings[(broadcast_id, tactical_id)] = similarity_score
        
        return mappings
    
    def assign_consistent_ids(self, mapping_results):
        """
        Assign consistent IDs across both views
        
        Args:
            mapping_results: Results from map_players method
            
        Returns:
            Dictionary mapping original IDs to consistent IDs
        """
        consistent_id = 1
        id_mapping = {
            'broadcast_to_consistent': {},
            'tactical_to_consistent': {},
            'consistent_to_original': {}
        }
        
        # Assign consistent IDs to mapped players
        for (broadcast_id, tactical_id), score in mapping_results['mappings'].items():
            id_mapping['broadcast_to_consistent'][broadcast_id] = consistent_id
            id_mapping['tactical_to_consistent'][tactical_id] = consistent_id
            id_mapping['consistent_to_original'][consistent_id] = {
                'broadcast': broadcast_id,
                'tactical': tactical_id,
                'similarity': score
            }
            consistent_id += 1
        
        # Assign unique IDs to unmapped players
        for broadcast_id in mapping_results['unmapped_broadcast']:
            id_mapping['broadcast_to_consistent'][broadcast_id] = consistent_id
            id_mapping['consistent_to_original'][consistent_id] = {
                'broadcast': broadcast_id,
                'tactical': None,
                'similarity': 0.0
            }
            consistent_id += 1
        
        for tactical_id in mapping_results['unmapped_tactical']:
            id_mapping['tactical_to_consistent'][tactical_id] = consistent_id
            id_mapping['consistent_to_original'][consistent_id] = {
                'broadcast': None,
                'tactical': tactical_id,
                'similarity': 0.0
            }
            consistent_id += 1
        
        return id_mapping
    
    def validate_mapping(self, mapping_results, min_confidence=0.8):
        """
        Validate mapping results and identify potential issues
        
        Args:
            mapping_results: Results from map_players method
            min_confidence: Minimum confidence for high-quality mappings
            
        Returns:
            Dictionary with validation results
        """
        mappings = mapping_results['mappings']
        
        # Calculate statistics
        if mappings:
            similarity_scores = list(mappings.values())
            avg_similarity = np.mean(similarity_scores)
            min_similarity = np.min(similarity_scores)
            max_similarity = np.max(similarity_scores)
            
            high_confidence_mappings = sum(1 for score in similarity_scores 
                                         if score >= min_confidence)
        else:
            avg_similarity = 0
            min_similarity = 0
            max_similarity = 0
            high_confidence_mappings = 0
        
        # Identify potential issues
        issues = []
        
        if len(mappings) == 0:
            issues.append("No player mappings found")
        
        if avg_similarity < self.similarity_threshold:
            issues.append(f"Low average similarity: {avg_similarity:.3f}")
        
        unmapped_ratio = (len(mapping_results['unmapped_broadcast']) + 
                         len(mapping_results['unmapped_tactical'])) / max(1, 
                         len(mapping_results['broadcast_ids']) + 
                         len(mapping_results['tactical_ids']))
        
        if unmapped_ratio > 0.5:
            issues.append(f"High unmapped ratio: {unmapped_ratio:.3f}")
        
        return {
            'total_mappings': len(mappings),
            'high_confidence_mappings': high_confidence_mappings,
            'average_similarity': avg_similarity,
            'min_similarity': min_similarity,
            'max_similarity': max_similarity,
            'unmapped_ratio': unmapped_ratio,
            'issues': issues,
            'is_valid': len(issues) == 0
        }
