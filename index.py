from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import base64
import json
import cv2
import datetime
import numpy as np
import mediapipe as mp
from scipy.spatial.distance import euclidean
import math

app = Flask(__name__)
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Other route definitions remain the same


# Initialize mediapipe face mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# Define thresholds
MIN_FACE_AREA = 70000  # Minimum face area in pixels
MAX_TILT_ANGLE = 15  # Maximum tilt angle in degrees

# Outer face contour indices
FACE_OUTLINE_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]

# Eye indices for tilt calculation
LEFT_EYE_INDEX = 33  # Left eye outer corner
RIGHT_EYE_INDEX = 263  # Right eye outer corner

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, 
    max_num_faces=1, 
    min_detection_confidence=0.5
)

@app.route('/')
def default_route():
    return jsonify({"message": "Welcome to the Face API"}), 200

@app.route('/Face_API/receive', methods=['POST'])
@cross_origin()  # This decorator is optional if you've set CORS globally
def receive_image():
    try:
        # Parse the incoming JSON data
        data = request.get_json()
        base64_string = data.get('image')
        
        # Check if the image data is present
        if not base64_string:
            return jsonify({"error": "No image data provided"}), 400

        # Decode the base64 string to get the image bytes
        image_data = base64.b64decode(base64_string)
        
        # Process the image data as needed
        # For demonstration, we'll just print the first 100 bytes
        print(image_data[:100])

        # Respond back to the client
        return jsonify({"message": "Image received and processed"}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/Face_API/register', methods=['POST', 'OPTIONS'])
def register_user():

    try:
        # Main handling for POST request
        data = request.get_json()
        base64_string = data.get('image')
        name = data.get('name')
        role = data.get('role')
        department = data.get('department')

        if not (name and role and department and base64_string):
            return jsonify({"success": False, "message": "All fields are required"}), 400

        image_data = base64.b64decode(base64_string)
        print("Image data received")
        nparr = np.frombuffer(image_data, np.uint8)
        print("Image data converted to numpy array")
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print("Image decoded and converted to RGB")
        results = face_mesh.process(image_rgb)
        print("Face mesh processed")
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get the image dimensions
                h, w, _ = image.shape

                # Calculate the face tilt angle using eye landmarks
                left_eye = face_landmarks.landmark[LEFT_EYE_INDEX]
                right_eye = face_landmarks.landmark[RIGHT_EYE_INDEX]
                left_eye_coords = (int(left_eye.x * w), int(left_eye.y * h))
                right_eye_coords = (int(right_eye.x * w), int(right_eye.y * h))

                dx = right_eye_coords[0] - left_eye_coords[0]
                dy = right_eye_coords[1] - left_eye_coords[1]
                tilt_angle = abs(math.degrees(math.atan2(dy, dx)))

                # Get outer contour landmarks for cropping
                outline_points = [
                    (int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h))
                    for i in FACE_OUTLINE_INDICES
                ]

                # Calculate bounding box for the face outline points
                x_coords = [p[0] for p in outline_points]
                y_coords = [p[1] for p in outline_points]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                # Calculate face area to ensure it's large enough
                face_area = (x_max - x_min) * (y_max - y_min)

                # Draw landmarks on the image
                for idx, landmark in enumerate(face_landmarks.landmark):
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
                
                # Check if face meets area and tilt angle thresholds
                if face_area >= MIN_FACE_AREA and tilt_angle <= MAX_TILT_ANGLE:
                    # Crop the face region
                    cropped_face = image[y_min:y_max, x_min:x_max]

                    # Get the landmark coordinates and convert them to a list
                    landmarks = [[landmark.x, landmark.y, landmark.z] for landmark in face_landmarks.landmark]

                    # Convert the landmarks list to a JSON string
                    landmarks_json = json.dumps(landmarks)

                return jsonify({"success": True, "message": "Registered Successfully", "landmarks": landmarks_json}), 200
        else:
            if face_area < MIN_FACE_AREA:
                print("Face too far away; skipping detection.")
            if tilt_angle > MAX_TILT_ANGLE:
                print("Face is tilted; skipping detection.")
            return jsonify({"success": False, "message": "No face detected, please try again"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

