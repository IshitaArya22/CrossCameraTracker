import cv2
import numpy as np
from sklearn.preprocessing import normalize

class FeatureExtractor:
    """Extract visual features from player crops for cross-view matching"""
    
    def __init__(self):
        """Initialize feature extractor"""
        self.target_size = (64, 128)  # Standard person re-identification size
        
    def extract_features(self, player_crop):
        """
        Extract comprehensive features from player crop
        
        Args:
            player_crop: Cropped image of player
            
        Returns:
            Feature vector combining multiple descriptors
        """
        if player_crop.size == 0:
            return np.zeros(512)  # Return zero vector for empty crops
        
        # Resize crop to standard size
        resized_crop = cv2.resize(player_crop, self.target_size)
        
        # Extract different types of features
        color_features = self._extract_color_features(resized_crop)
        texture_features = self._extract_texture_features(resized_crop)
        shape_features = self._extract_shape_features(resized_crop)
        
        # Combine all features
        combined_features = np.concatenate([
            color_features,
            texture_features,
            shape_features
        ])
        
        # Normalize features
        normalized_features = normalize(combined_features.reshape(1, -1))[0]
        
        return normalized_features
    
    def _extract_color_features(self, image):
        """Extract color-based features"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Extract color histograms
        features = []
        
        # HSV histograms
        h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        
        # LAB histograms
        l_hist = cv2.calcHist([lab], [0], None, [32], [0, 256])
        a_hist = cv2.calcHist([lab], [1], None, [32], [0, 256])
        b_hist = cv2.calcHist([lab], [2], None, [32], [0, 256])
        
        # Normalize and flatten histograms
        histograms = [h_hist, s_hist, v_hist, l_hist, a_hist, b_hist]
        for hist in histograms:
            hist = hist.flatten()
            hist = hist / (np.sum(hist) + 1e-10)  # Normalize
            features.extend(hist)
        
        # Dominant colors (mean colors in different regions)
        h, w = image.shape[:2]
        regions = [
            image[:h//2, :],          # Upper half
            image[h//2:, :],          # Lower half
            image[:, :w//2],          # Left half
            image[:, w//2:],          # Right half
            image[h//4:3*h//4, w//4:3*w//4]  # Center region
        ]
        
        for region in regions:
            if region.size > 0:
                mean_color = np.mean(region.reshape(-1, 3), axis=0)
                features.extend(mean_color)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_texture_features(self, image):
        """Extract texture-based features using Local Binary Patterns"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Local Binary Pattern
        lbp = self._compute_lbp(gray)
        
        # LBP histogram
        lbp_hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
        lbp_hist = lbp_hist.flatten()
        lbp_hist = lbp_hist / (np.sum(lbp_hist) + 1e-10)
        
        # Gradient features
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_hist = cv2.calcHist([gradient_magnitude.astype(np.uint8)], 
                                   [0], None, [32], [0, 256])
        gradient_hist = gradient_hist.flatten()
        gradient_hist = gradient_hist / (np.sum(gradient_hist) + 1e-10)
        
        # Combine texture features
        texture_features = np.concatenate([lbp_hist, gradient_hist])
        
        return texture_features.astype(np.float32)
    
    def _compute_lbp(self, image, radius=1, n_points=8):
        """Compute Local Binary Pattern"""
        h, w = image.shape
        lbp = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = image[i, j]
                pattern = 0
                
                for p in range(n_points):
                    angle = 2.0 * np.pi * p / n_points
                    x = int(round(i + radius * np.cos(angle)))
                    y = int(round(j - radius * np.sin(angle)))
                    
                    if 0 <= x < h and 0 <= y < w:
                        if image[x, y] >= center:
                            pattern |= (1 << p)
                
                lbp[i, j] = pattern
        
        return lbp
    
    def _extract_shape_features(self, image):
        """Extract shape-based features"""
        # Convert to grayscale and apply edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shape_features = []
        
        if contours:
            # Find largest contour (assuming it's the person)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Compute shape descriptors
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            if perimeter > 0:
                compactness = 4 * np.pi * area / (perimeter ** 2)
            else:
                compactness = 0
            
            # Bounding rectangle features
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = w / h if h > 0 else 0
            extent = area / (w * h) if w * h > 0 else 0
            
            shape_features = [area/1000, perimeter/100, compactness, aspect_ratio, extent]
        else:
            shape_features = [0, 0, 0, 0, 0]
        
        # Hu moments
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        
        # Combine shape features
        combined_shape = np.concatenate([shape_features, hu_moments])
        
        return combined_shape.astype(np.float32)
    
    def extract_batch_features(self, player_crops):
        """
        Extract features for multiple player crops
        
        Args:
            player_crops: List of player crop images
            
        Returns:
            Array of feature vectors
        """
        features = []
        for crop in player_crops:
            feature = self.extract_features(crop)
            features.append(feature)
        
        return np.array(features) if features else np.empty((0, 512))
