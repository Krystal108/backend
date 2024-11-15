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
import mysql.connector
import time

app = Flask(__name__)
from flask import Flask, request, jsonify
from flask_cors import CORS

# Define the mysql connection
connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="face_id",
    port="3307"
)

# Define the cursor
cursor = connection.cursor()


app = Flask(__name__)
CORS(app)

# Define the threshold times
ON_TIME_THRESHOLD = "07:30:00"
LATE_THRESHOLD = "07:31:00"

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

@app.route('/widgets/count-total-employees', methods=['GET'])
def count_total_employees():
    try:
        query = "SELECT COUNT(*) FROM register"
        with connection.cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchone()
            total_employees = result[0]
        return jsonify({"total_employees": total_employees}), 200
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 400


@app.route('/widgets/on-time-check', methods=['POST'])
def on_time_check():
    try:
        data = request.get_json()
        name = data.get('name')
        print(name)
        threshold_hour = 7  # Threshold hour for being on time
        if not name:
            return jsonify({"error": "Name is required"}), 400
        query = "SELECT time_in FROM attendance WHERE name = %s ORDER BY time_in DESC LIMIT 1"
        with connection.cursor() as cursor:
            cursor.execute(query, (name,))
            result = cursor.fetchone()
            if result:
                time_in = result[0]
                print(time_in)
                if time_in and isinstance(time_in, datetime.datetime):
                    time_in_hour = time_in.hour
                    if time_in_hour <= threshold_hour:
                        return jsonify({"status": "On Time"}), 200
                    else:
                        return jsonify({"status": "Is Late"}), 200
                else:
                    return jsonify({"error": "Invalid time_in value"}), 400
            else:
                return jsonify({"error": "No attendance record found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 400



@app.route('/Face_API/register', methods=['POST'])
def register_user():
    try:
        # Main handling for POST request
        data = request.get_json()
        base64_string = data.get('image')
        name = data.get('name')
        role = data.get('role')
        department = data.get('department')
        pin = data.get('pin')
        securityQuestion = data.get('securityQuestion')
        securityAnswer = data.get('securityAnswer')

        if not (name and role and department and pin and securityQuestion and securityAnswer and base64_string):
            return jsonify({"success": False, "message": "All fields are required"}), 400

        # Check if user already exists
        with connection.cursor() as cursor:
            query = "SELECT * FROM register WHERE name = %s"
            cursor.execute(query, (name,))
            result = cursor.fetchone()

        if result:
            return jsonify({"success": False, "message": "User already exists"}), 400

        # Decode and process the image
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
                    
                if face_area < MIN_FACE_AREA:
                    return jsonify({"success": False, "message": "Face too far away!"}), 400
                if tilt_angle > MAX_TILT_ANGLE:
                    return jsonify({"success": False, "message": "Face is tilted"}), 400
                
                # Check if face meets area and tilt angle thresholds
                if face_area >= MIN_FACE_AREA and tilt_angle <= MAX_TILT_ANGLE:
                    # Get the landmark coordinates and convert them to a list
                    landmarks = [[landmark.x, landmark.y, landmark.z] for landmark in face_landmarks.landmark]

                    # Convert the landmarks list to a JSON string
                    landmarks_json = json.dumps(landmarks)
                    
                    # Insert the user data into the database
                    with connection.cursor() as cursor:
                        query = """
                            INSERT INTO register (name, role, department, pin, landmarks_hash, sec_question, sec_answer)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """
                        cursor.execute(query, (name, role, department, pin, landmarks_json, securityQuestion, securityAnswer))
                        connection.commit()
                    
                    return jsonify({"success": True, "message": f"{name} is registered successfully"}), 200

        return jsonify({"success": False, "message": "No face detected, please try again"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# mark-attendance endpoint
@app.route('/Face_API/mark-attendance', methods=['POST'])
def mark_attendance():
    try:
        data = request.get_json()
        base64_string = data.get('image')
        name = data.get('name')

        if not (name and base64_string):
            return jsonify({"success": False, "message": "All fields are required"}), 400

        # Decode and process the image
        image_data = base64.b64decode(base64_string)
        print("Image data received")
        nparr = np.frombuffer(image_data, np.uint8)
        print("Image data converted to numpy array")
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        print("Image decoded")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process face landmarks
        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return jsonify({"success": False, "message": "No face detected, please try again"}), 400

        for face_landmarks in results.multi_face_landmarks:
            print("Face detected")
            h, w, _ = image.shape
            left_eye = face_landmarks.landmark[LEFT_EYE_INDEX]
            right_eye = face_landmarks.landmark[RIGHT_EYE_INDEX]
            left_eye_coords = (int(left_eye.x * w), int(left_eye.y * h))
            right_eye_coords = (int(right_eye.x * w), int(right_eye.y * h))
            dx, dy = right_eye_coords[0] - left_eye_coords[0], right_eye_coords[1] - left_eye_coords[1]
            tilt_angle = abs(math.degrees(math.atan2(dy, dx)))

            # Check face area and tilt angle
            outline_points = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in FACE_OUTLINE_INDICES]
            x_min, x_max = min(p[0] for p in outline_points), max(p[0] for p in outline_points)
            y_min, y_max = min(p[1] for p in outline_points), max(p[1] for p in outline_points)
            face_area = (x_max - x_min) * (y_max - y_min)

            if face_area < MIN_FACE_AREA:
                return jsonify({"success": False, "message": "Face too far away!"}), 400
            if tilt_angle > MAX_TILT_ANGLE:
                return jsonify({"success": False, "message": "Face is tilted"}), 400

            # Retrieve registered landmarks from the database
            with connection.cursor() as cursor:
                cursor.execute("SELECT landmarks_hash, role, department FROM register WHERE name = %s", (name,))
                user_record = cursor.fetchone()
            
            if not user_record:
                return jsonify({"success": False, "message": "User not registered"}), 400

            stored_landmarks, role, department = json.loads(user_record[0]), user_record[1], user_record[2]
            landmarks = [[landmark.x, landmark.y, landmark.z] for landmark in face_landmarks.landmark]
            distances = [euclidean(stored_landmarks[i], landmarks[i]) for i in range(len(landmarks))]
            avg_distance = sum(distances) / len(distances)

            if avg_distance >= 0.1:
                return jsonify({"success": False, "message": "Face does not match registered data"}), 400

            # Check if the user is already marked present
            
            check_in = datetime.datetime.now().strftime("%Y-%m-%d")
            today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Check if a record exists for today's date
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM attendance WHERE name = %s AND DATE(time_in) = %s", (name, check_in))
                if cursor.fetchone():
                    return jsonify({
                        "success": True,
                        "message": f"{name} is already marked present",
                        "name": name,
                        "role": role,
                        "department": department
                    }), 200
                    
            print("Inserting new attendance record")
            
            # Insert new attendance record
            with connection.cursor() as cursor:
                cursor.execute("INSERT INTO attendance (name, role, department, time_in) VALUES (%s, %s, %s, %s)", (name, role, department, today))
                connection.commit()

            print("Attendance marked successfully")

            return jsonify({
                "success": True,
                "message": f"{name} is marked present",
                "name": name,
                "role": role,
                "department": department
            }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# bar-chart endpoint /charts/bar-chart
@app.route('/charts/bar-chart', methods=['POST'])
def bar_chart():
    try:
        data = request.get_json()
        name = data.get('name')
        print(f"Name: {name}")
        
        if not name:
            return jsonify({"error": "Name is required"}), 400
        
        # Query to fetch time_in per employee by quarter
        query = "SELECT name, time_in, QUARTER(time_in) AS quarter FROM attendance WHERE name = %s"
        with connection.cursor() as cursor:
            cursor.execute(query, (name,))
            results = cursor.fetchall()
        
        # Initialize structures to store attendance status counts
        employees = {}
        on_time_data = []
        late_data = []
        absent_data = []
        
        # Process each record from the results
        for row in results:
            employee_name = row[0]  # Access name
            time_in = row[1]        # Access time_in
            quarter = row[2]        # Access quarter
            
            # Initialize data for the employee and quarter if not already set
            if employee_name not in employees:
                employees[employee_name] = {}
            if quarter not in employees[employee_name]:
                employees[employee_name][quarter] = {'on_time': 0, 'late': 0, 'absent': 0}

            # Check attendance status
            if time_in is None:
                # Absent if no time_in value
                employees[employee_name][quarter]['absent'] += 1
            else:
                # Convert time_in to time-only format
                time_only = time_in.strftime('%H:%M:%S')
                if time_only <= ON_TIME_THRESHOLD:
                    employees[employee_name][quarter]['on_time'] += 1
                elif time_only <= LATE_THRESHOLD:
                    employees[employee_name][quarter]['late'] += 1
                else:
                    employees[employee_name][quarter]['absent'] += 1
                    
        # Organize data for chart response
        categories = []
        for employee_name, quarters in employees.items():
            for quarter, counts in quarters.items():
                categories.append(f"{employee_name} (Q{quarter})")
                on_time_data.append(counts['on_time'])
                late_data.append(counts['late'])
                absent_data.append(counts['absent'])
                
        # Prepare JSON response
        return jsonify({
            'categories': categories,
            'on_time_data': on_time_data,
            'late_data': late_data,
            'absent_data': absent_data
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# pie-chart endpoint /charts/pie-chart
@app.route('/charts/pie-chart', methods=['GET'])
def pie_chart():
    
    connection_pie = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="task_management",
    port="3307"
    )

    cursor_pie = connection_pie.cursor()
    try:
        query = "SELECT table_name, table_rows FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'task_management'"
        with connection_pie.cursor() as cursor_pie:
            cursor_pie.execute(query)
            results = cursor_pie.fetchall()
        
        print(results)
        
        # Initialize list to store table names and row counts
        table_names = []
        row_counts = []
        
        # Process each record from the results
        for row in results:
            table_names.append(row[0])
            row_counts.append(row[1])
            
        # Prepare JSON response
        return jsonify({
            'table_names': table_names,
            'row_counts': row_counts
        }), 200
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 400
    

# leave-request endpoint /tablbes/leave-table
@app.route('/tables/leave-table', methods=['POST'])
def leave_request():
# Define the mysql connection
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="face_id",
        port="3307"
    )
   
    # Define the cursor
    cursor = connection.cursor()

    try:
        data = request.get_json()
        name = data.get('username')
        print(f"Leave Request: {name}")
        
        if not name:
            return jsonify({"error": "Name is required"}), 400
        
        query = "SELECT * FROM leave_request WHERE name = %s"
        with connection.cursor() as cursor:
            cursor.execute(query, (name,))
            results = cursor.fetchall()
        
        # Initialize list to store leave requests
        leave_types = []
        start_dates = []
        end_dates = []
        statuses = []
        
        # Process each record from the results
        for row in results:
            leave_types.append(row[2])
            start_dates.append(row[3].strftime('%Y-%m-%d'))
            end_dates.append(row[4].strftime('%Y-%m-%d'))
            statuses.append(row[5])
        
        # Prepare JSON response
        return jsonify({
            'leave_types': leave_types,
            'start_dates': start_dates,
            'end_dates': end_dates,
            'statuses': statuses
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# employee table endpoint /tables/employee-table
@app.route('/tables/employee-table', methods=['GET'])
def employee_table():
    connection_worker = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="factory_workers",
        port="3307"
    )
    
    cursor_worker = connection_worker.cursor()

    try:
        query = "SELECT * FROM employee_records"
        with connection_worker.cursor() as cursor_worker:
            cursor_worker.execute(query)
            results = cursor_worker.fetchall()
            
        # Initialize list to store employee records
        employee_names = []
        employee_positions = []
        employee_start_dates = []
        
        # Process each record from the results
        for row in results:
            employee_names.append(row[1])
            employee_positions.append(row[2])
            employee_start_dates.append(row[11].strftime('%Y-%m-%d'))
        
        # Prepare JSON response
        return jsonify({
            'employee_names': employee_names,
            'employee_positions': employee_positions,
            'employee_start_dates': employee_start_dates
        }), 200
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 400
    
if __name__ == '__main__':
    app.run(debug=True)

