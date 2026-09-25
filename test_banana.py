import cv2
import sys
from banana_classifier_simple import BananaClassifierSimple

def test_with_webcam():
    """Test banana detection with webcam"""
    classifier = BananaClassifierSimple()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
    
    print("✅ Webcam opened. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        processed_frame, results = classifier.process_video_frame(frame)
        
        # Show frame
        cv2.imshow('Banana Detection', processed_frame)
        
        # Print results
        if results:
            for i, result in enumerate(results):
                print(f"Banana {i+1}: {result['ripeness']['name']} - {result['ripeness']['description']}")
                if result['defects'] and result['defects']['has_defects']:
                    print(f"  ⚠️ Defects: {', '.join(result['defects']['defect_type'])}")
        
        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def test_with_image(image_path):
    """Test banana detection with image file"""
    classifier = BananaClassifierSimple()
    
    result = classifier.process_image(image_path)
    
    if result.get('detected'):
        print(f"✅ Detected {result['number_of_bananas']} banana(s)")
        for i, detection in enumerate(result['results']):
            print(f"\nBanana {i+1}:")
            print(f"  Stage: {detection['ripeness']['name']} (Stage {detection['ripeness']['stage']})")
            print(f"  Description: {detection['ripeness']['description']}")
            print(f"  Usage: {detection['ripeness']['usage']}")
            if detection['defects'] and detection['defects']['has_defects']:
                print(f"  ⚠️ Defects: {', '.join(detection['defects']['defect_type'])}")
    else:
        print(f"❌ {result.get('message', 'No banana detected')}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test with image file
        test_with_image(sys.argv[1])
    else:
        # Test with webcam
        test_with_webcam()