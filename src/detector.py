import cv2
import numpy as np
from ultralytics import YOLO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLODetector:
    """
    A class to handle YOLO model loading and inference.
    """
    def __init__(self, model_version='yolo11n.pt', conf_threshold=0.25):
        """
        Initialize the YOLO detector.
        
        Args:
            model_version (str): The YOLO model version to load (e.g., 'yolo11n.pt', 'yolov8n.pt').
            conf_threshold (float): Confidence threshold for detections.
        """
        try:
            logger.info(f"Loading YOLO model: {model_version}")
            self.model = YOLO(model_version)
            self.conf_threshold = conf_threshold
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def detect(self, frame):
        """
        Perform inference on a single frame.
        
        Args:
            frame (numpy.ndarray): The input frame (BGR image).
            
        Returns:
            list: A list of detections, where each detection is a dictionary
                  containing 'box', 'conf', 'class_id', and 'class_name'.
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        detections = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for box in boxes:
                # Get box coordinates (xyxy)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Get confidence and class
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                
                detections.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'conf': conf,
                    'class_id': cls_id,
                    'class_name': cls_name
                })
                
        return detections

    def get_class_names(self):
        """Returns the class names dict {id: name}."""
        return self.model.names
