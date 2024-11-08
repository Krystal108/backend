import json
import base64
import cv2
import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)  # Enable CORS globally

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
    pass

if __name__ == '__main__':
    app.run(debug=True)
