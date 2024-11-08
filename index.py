from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import base64
import json
import cv2
import numpy as np
import mediapipe as mp

app = Flask(__name__)
# Enable CORS with the correct origin and options
CORS(app, resources={r"/*": {"origins": "https://superpack-fe.vercel.app", "supports_credentials": True}})

# Initialize mediapipe face mesh
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True, 
    max_num_faces=1, 
    min_detection_confidence=0.5
)

@app.route('/Face_API/receive', methods=['POST'])
@cross_origin()
def receive_image():
    try:
        data = request.get_json()
        base64_string = data.get('image')
        if not base64_string:
            return jsonify({"error": "No image data provided"}), 400
        image_data = base64.b64decode(base64_string)
        print(image_data[:100])
        return jsonify({"message": "Image received and processed"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/Face_API/register', methods=['POST'])
@cross_origin()
def register_user():
    try:
        data = request.get_json()
        base64_string = data.get('image')
        name = data.get('name')
        role = data.get('role')
        department = data.get('department')

        if not name or not role or not department:
            return jsonify({"success": False, "message": "Name, role, and department are required"}), 400
        if not base64_string:
            return jsonify({"success": False, "message": "Image data is required"}), 400

        image_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = [[landmark.x, landmark.y, landmark.z] for landmark in face_landmarks.landmark]
            landmarks_json = json.dumps(landmarks)
            print(f"User {name} registered with landmarks: {landmarks_json}")
            return jsonify({"success": True, "message": f"{name} Registered Successfully"}), 200
        else:
            return jsonify({"success": False, "message": "No face detected, please try again"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
