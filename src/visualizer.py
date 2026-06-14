import cv2
import numpy as np
import time

class Visualizer:
    """
    Handles high-quality drawing of detections and UI elements.
    """
    def __init__(self, class_names):
        self.class_names = class_names
        # Generate random unique colors for each class
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype=np.uint8)
        self.prev_time = 0

    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels on the frame.
        """
        for det in detections:
            x1, y1, x2, y2 = det['box']
            cls_id = det['class_id']
            cls_name = det['class_name']
            conf = det['conf']
            
            color = [int(c) for c in self.colors[cls_id]]
            
            # Draw semi-transparent background for label
            label = f"{cls_name} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            
            # Modern bounding box (rounded-ish appearance using thickness and lines)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Label background
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 7), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def draw_ui(self, frame, fps, detection_count):
        """
        Draw a premium status overlay.
        """
        h, w, _ = frame.shape
        
        # Draw a semi-transparent overlay at the top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), (45, 45, 45), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add Text
        cv2.putText(frame, f"YOLO REAL-TIME DETECTION | FPS: {fps:.1f}", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.putText(frame, f"OBJECTS: {detection_count}", (w - 150, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        
        return frame

    def calculate_fps(self):
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time) if self.prev_time != 0 else 0
        self.prev_time = curr_time
        return fps
