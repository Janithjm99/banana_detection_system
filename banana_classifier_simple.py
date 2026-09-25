import cv2
import numpy as np
import json
from datetime import datetime
from PIL import Image
import io

class BananaClassifierSimple:
    """Lightweight banana classifier using color analysis only"""
    
    def __init__(self):
        # Define ripeness stages with color codes
        self.stages = {
            'raw': {
                'stage': 1,
                'name': 'Raw (Green)',
                'description': 'Unripe, starchy, not sweet',
                'color': (0, 255, 0),
                'usage': 'Cooking, frying'
            },
            'turning': {
                'stage': 2,
                'name': 'Turning',
                'description': 'Beginning to ripen, some starch',
                'color': (0, 200, 100),
                'usage': 'Frying, cooking'
            },
            'more_green': {
                'stage': 3,
                'name': 'More Green than Yellow',
                'description': 'Mostly starchy, firm',
                'color': (50, 200, 50),
                'usage': 'Frying'
            },
            'more_yellow': {
                'stage': 4,
                'name': 'More Yellow than Green',
                'description': 'Sweet, firm, creamy',
                'color': (0, 255, 200),
                'usage': 'Fresh eating'
            },
            'ripe': {
                'stage': 5,
                'name': 'Ripe (Yellow)',
                'description': 'Very sweet, soft texture',
                'color': (0, 255, 255),
                'usage': 'Snacking, smoothies'
            },
            'fully_ripe': {
                'stage': 6,
                'name': 'Fully Ripe',
                'description': 'Maximum sweetness, soft flesh',
                'color': (50, 200, 255),
                'usage': 'Cereal, smoothies'
            },
            'speckled': {
                'stage': 7,
                'name': 'Speckled (Sugar Spots)',
                'description': 'Very sweet, aromatic, soft',
                'color': (100, 150, 255),
                'usage': 'Banana bread, baking'
            },
            'overripe': {
                'stage': 8,
                'name': 'Overripe (Brown)',
                'description': 'Mushy, extremely sweet',
                'color': (255, 200, 100),
                'usage': 'Pureeing, freezing'
            },
            'spoiled': {
                'stage': 9,
                'name': 'Spoiled',
                'description': 'Moldy, rotten, not edible',
                'color': (0, 0, 255),
                'usage': 'Compost only'
            }
        }
        
        # HSV color ranges for detection
        self.green_lower = np.array([35, 50, 50])
        self.green_upper = np.array([85, 255, 255])
        self.yellow_lower = np.array([20, 50, 50])
        self.yellow_upper = np.array([35, 255, 255])
        self.brown_lower = np.array([0, 50, 50])
        self.brown_upper = np.array([20, 255, 255])
        
        print("✅ Banana Classifier Initialized (Simple Mode)")
    
    def detect_banana_simple(self, image):
        """Simple banana detection using color segmentation"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for banana colors (yellow-green spectrum)
        lower_yellow = np.array([15, 30, 30])
        upper_yellow = np.array([45, 255, 255])
        
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area to avoid noise
                x, y, w, h = cv2.boundingRect(contour)
                # Filter by aspect ratio (bananas are elongated)
                aspect_ratio = w / h if h > 0 else 0
                if 0.2 < aspect_ratio < 5.0:  # Reasonable banana shape
                    detections.append({
                        'bbox': [x, y, x + w, y + h],
                        'confidence': 0.8  # Default confidence for simple detection
                    })
        
        return detections
    
    def extract_banana_region(self, image, bbox):
        """Extract the banana region from the image"""
        x1, y1, x2, y2 = bbox
        return image[y1:y2, x1:x2]
    
    def analyze_color_distribution(self, banana_region):
        """Analyze color distribution in HSV space"""
        if banana_region.size == 0:
            return None
        
        hsv = cv2.cvtColor(banana_region, cv2.COLOR_BGR2HSV)
        
        # Calculate color distribution
        green_pixels = cv2.inRange(hsv, self.green_lower, self.green_upper)
        yellow_pixels = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        brown_pixels = cv2.inRange(hsv, self.brown_lower, self.brown_upper)
        
        total_pixels = banana_region.shape[0] * banana_region.shape[1]
        
        green_ratio = np.sum(green_pixels > 0) / total_pixels
        yellow_ratio = np.sum(yellow_pixels > 0) / total_pixels
        brown_ratio = np.sum(brown_pixels > 0) / total_pixels
        
        return {
            'green': green_ratio,
            'yellow': yellow_ratio,
            'brown': brown_ratio
        }
    
    def detect_defects(self, banana_region):
        """Detect spoilage, bruises, and mold"""
        if banana_region.size == 0:
            return {'has_defects': False, 'defect_type': None}
        
        gray = cv2.cvtColor(banana_region, cv2.COLOR_BGR2GRAY)
        
        # Detect dark spots (bruises)
        _, dark_spots = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        dark_ratio = np.sum(dark_spots > 0) / (gray.shape[0] * gray.shape[1])
        
        # Detect texture changes (mold/rot)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = np.var(laplacian)
        
        defects = []
        
        if dark_ratio > 0.1:  # More than 10% dark spots
            defects.append('bruise')
        
        if texture_variance > 1000:  # High texture variance suggests mold
            defects.append('mold')
        
        if len(defects) > 0:
            return {'has_defects': True, 'defect_type': defects}
        else:
            return {'has_defects': False, 'defect_type': None}
    
    def classify_ripeness(self, color_distribution, defects):
        """Determine ripeness based on color distribution and defects"""
        if color_distribution is None:
            return self.stages['spoiled']
        
        green = color_distribution['green']
        yellow = color_distribution['yellow']
        brown = color_distribution['brown']
        
        # Check for spoilage first
        if defects and defects['has_defects']:
            if 'mold' in defects['defect_type']:
                return self.stages['spoiled']
        
        # If mostly brown/black
        if brown > 0.4:
            return self.stages['overripe']
        
        # If mostly green
        if green > 0.6:
            if yellow < 0.1:
                return self.stages['raw']
            elif yellow < 0.2:
                return self.stages['turning']
            else:
                return self.stages['more_green']
        
        # If mostly yellow
        elif yellow > 0.5:
            if brown > 0.1 and brown < 0.3:
                return self.stages['speckled']
            elif brown < 0.05:
                return self.stages['ripe']
            elif brown < 0.15:
                return self.stages['fully_ripe']
            else:
                return self.stages['fully_ripe']
        
        # Mixed colors
        else:
            if green > yellow and green > brown:
                return self.stages['more_green']
            elif yellow > green and yellow > brown:
                return self.stages['more_yellow']
            else:
                return self.stages['ripe']
    
    def process_image(self, image_path):
        """Main function to process image"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return {'error': 'Could not read image'}
        
        # Detect bananas
        detections = self.detect_banana_simple(image)
        
        if len(detections) == 0:
            return {
                'detected': False,
                'message': 'No banana detected in image'
            }
        
        results = []
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            # Extract banana region
            banana_region = self.extract_banana_region(image, bbox)
            
            # Analyze color
            color_dist = self.analyze_color_distribution(banana_region)
            
            # Detect defects
            defects = self.detect_defects(banana_region)
            
            # Classify ripeness
            stage_info = self.classify_ripeness(color_dist, defects)
            
            results.append({
                'bbox': bbox,
                'confidence': confidence,
                'ripeness': stage_info,
                'color_distribution': color_dist,
                'defects': defects,
                'timestamp': datetime.now().isoformat()
            })
        
        return {
            'detected': True,
            'number_of_bananas': len(results),
            'results': results
        }
    
    def process_video_frame(self, frame):
        """Process a single video frame for real-time detection"""
        # Detect bananas
        detections = self.detect_banana_simple(frame)
        
        if len(detections) == 0:
            return frame, None
        
        results = []
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            x1, y1, x2, y2 = bbox
            
            # Extract and analyze
            banana_region = self.extract_banana_region(frame, bbox)
            color_dist = self.analyze_color_distribution(banana_region)
            defects = self.detect_defects(banana_region)
            stage_info = self.classify_ripeness(color_dist, defects)
            
            # Draw bounding box
            color = stage_info['color']
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw label
            label = f"{stage_info['name']} ({confidence:.2f})"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw info
            if defects and defects['has_defects']:
                defect_text = f"⚠️ {', '.join(defects['defect_type'])}"
                cv2.putText(frame, defect_text, (x1, y2+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            results.append({
                'bbox': bbox,
                'ripeness': stage_info,
                'confidence': confidence,
                'defects': defects
            })
        
        return frame, results

if __name__ == "__main__":
    print("✅ Banana Classifier Simple Version Loaded")