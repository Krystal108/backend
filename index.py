from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import base64
import json
import cv2
import datetime
import numpy as np
import mediapipe as mp
from scipy.spatial.distance import euclidean

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
            return jsonify({"success": True, "message": "Registered Successfully"}), 200
        else:
            return jsonify({"success": False, "message": "No face detected, please try again"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

