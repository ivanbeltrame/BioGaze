import cv2
import os
from ultralytics import YOLO
import numpy as np
import config
from pathlib import Path

class FaceDetector:
    def __init__(self):
        """
        Initializes the face detector with the specified model path.
        """
        MODEL_NAME = "yolo26n.pt"
        
        YOLO(MODEL_NAME) # Downloads the pre trained model to your current directory if not already there
        
        model_path = Path(MODEL_NAME).resolve() # Get the exact absolute path on your disk
        
        self.model = YOLO(model_path)

    def detector_analysis(self, image_path):
        """
        Detects faces in the given image path.
        If there is only one face, it also computes the exposure

        Args:
            image_path (str): Path to the image file.

        Returns:
            num_faces: number of faces detected.
            correct_exposure: Boolean that indicates whether the face has correct exposure.
        """
        results = self.model(image_path)  # Perform object detection
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Extract coordinates [x1, y1, x2, y2]

        if len(boxes) != 1:
            return len(boxes), False
        
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        box = boxes[0]
        top_left_x = int(box[0])
        top_left_y = int(box[1])
        bottom_right_x = int(box[2])
        bottom_right_y = int(box[3])

        roi = gray[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        histogram = cv2.calcHist([roi], [0], None, [256], [0, 256])
        histo_sum = histogram.sum()
        if histo_sum > 0:
            histogram = histogram / histo_sum
        bad_exposure = self.analyze_exposure(histogram)

        return len(boxes), not bad_exposure
    
    def detect_and_draw_faces(self, image_path, output_path=None):
        """
        Detects faces in the given image path.

        Args:
            image_path (str): Path to the image file.
            output_path (str, optional): Path to save the processed image. If provided,
                                         the image will be saved in that directory with "_detect"
                                         suffix appended to the filename. Otherwise, the image
                                         will be saved in a folder called "faces_detected"
                                         in the directory of the script (where your Python file is).

        Returns:
            list: List of bounding boxes for detected faces in the format [x_min, y_min, x_max, y_max].
        """
        img = cv2.imread(image_path)
        results = self.model(image_path)  # Perform object detection
        print('detected')

        boxes = results[0].boxes.xyxy.cpu().numpy()  # Extract coordinates

        # Extract and convert bounding boxes for detected faces
        for box in boxes:
            top_left_x = int(box[0])
            top_left_y = int(box[1])
            bottom_right_x = int(box[2])
            bottom_right_y = int(box[3])
            cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (50, 200, 129), 2)
        
        filename, ext = os.path.splitext(os.path.basename(image_path))  # Extract filename and extension
        output_filename = filename + "_detect" + ext
        
        if output_path:
            try:
                # Create the output directory if it doesn't exist
                os.makedirs(output_path, exist_ok=True)
                # Save with rectangle in the provided output_path (directory)
                faces_dir = output_path
            except OSError as e:
                print(f"Error creating output directory: {e}")
                # Handle the error gracefully, potentially use default saving behavior
        else:
            # Save with rectangle in "faces_detected" folder in script's directory
            script_dir = os.path.dirname(os.path.realpath(__file__))  # Get script's directory
            faces_dir = os.path.join(script_dir, "faces_detected")
            os.makedirs(faces_dir, exist_ok=True)  # Create "faces_detected" if it doesn't exist
        
        output_image_path = os.path.join(faces_dir, output_filename)
        cv2.imwrite(output_image_path, img)
    
    def get_bounding_box(self, image_path):
        """
        Detects faces in the given image path.

        Args:
            image_path (str): Path to the image file.

        Returns:
            tuple: Bounding box coordinates (top_left_x, top_left_y, bottom_right_x, bottom_right_y).
        """
        img = cv2.imread(image_path)
        results = self.model(image_path)  # Perform object detection

        image_height, image_width = img.shape[:2]
        boxes = results[0].boxes.xyxy.cpu().numpy()

        if len(boxes) < 1:
            return 0, 0, image_width, image_height

        # Extract and convert bounding boxes for detected faces
        box = boxes[0]
        top_left_x = int(box[0])
        top_left_y = int(box[1])
        bottom_right_x = int(box[2])
        bottom_right_y = int(box[3])

        return top_left_x, top_left_y, bottom_right_x, bottom_right_y

    def analyze_exposure(self, histogram):
        avg_light = np.mean(histogram[220:256])
        max_light = np.max(histogram[220:256])

        # Check for overexposure

        avg_dark = np.mean(histogram[0:170])
        max_dark = np.max(histogram[0:170])
        return (avg_dark > config.AVG_DARK_THRESHOLD and max_dark > config.MAX_DARK_THRESHOLD) or (avg_light > config.AVG_LIGHT_THRESHOLD and max_light > config.MAX_LIGHT_THRESHOLD)