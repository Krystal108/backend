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
CORS(app, resources={r"/Face_API/*": {"origins": "https://superpack-fe.vercel.app"}})
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

@app.route('/Face_API/register', methods=['POST'])
@cross_origin()  # This decorator is optional if you've set CORS globally
def register_user():
    if request.method == 'OPTIONS':
        # Preflight response with the allowed headers and methods
        response = jsonify({'status': 'Preflight check successful'})
        response.headers.add("Access-Control-Allow-Origin", "https://superpack-fe.vercel.app")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        return response
    
    # Regular POST handling
    try:
        data = request.get_json()
        # Process data and return a response
        response = jsonify({"success": True, "message": "User registered successfully"})
        response.headers.add("Access-Control-Allow-Origin", "https://superpack-fe.vercel.app")
        return response
    except Exception as e:
        response = jsonify({"error": str(e)})
        response.headers.add("Access-Control-Allow-Origin", "https://superpack-fe.vercel.app")
        return response, 400

if __name__ == '__main__':
    app.run(debug=True)
