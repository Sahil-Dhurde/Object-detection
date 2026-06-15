import cv2
import argparse
import sys
from src.detector import YOLODetector 
from src.visualizer import Visualizer

def run_detection(source=0, model='yolo11n.pt', confidence=0.25):
    """
    Main loop for object detection.
    """
    # Initialize detector
    detector = YOLODetector(model_version=model, conf_threshold=confidence)
    
    # Initialize visualizer
    visualizer = Visualizer(detector.get_class_names())
    
    # Open video source
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    print(f"Starting detection on source {source}...")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Finished processing or source disconnected.")
            break
            
        # Detect objects
        detections = detector.detect(frame)
        
        # Calculate FPS
        fps = visualizer.calculate_fps()
        
        # Draw detections and UI
        frame = visualizer.draw_detections(frame, detections)
        frame = visualizer.draw_ui(frame, fps, len(detections))
        
        # Display the frame
        cv2.imshow("Object Detection Project - YOLO", frame)
        
        # Handle key events
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modern YOLO Object Detection Project")
    parser.add_argument("--source", type=str, default="0", help="Source: 0 for webcam, or path to image/video file")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="YOLO model version (e.g., yolo11n.pt, yolo11s.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # Handle webcam index if it's a digit
    source = int(args.source) if args.source.isdigit() else args.source
    
    run_detection(source=source, model=args.model, confidence=args.conf)
