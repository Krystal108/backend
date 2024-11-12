import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

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

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and find face landmarks
    results = face_mesh.process(image_rgb)

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

            # Check if face meets area and tilt angle thresholds
            if face_area >= MIN_FACE_AREA and tilt_angle <= MAX_TILT_ANGLE:
                # Crop the face region
                cropped_face = image[y_min:y_max, x_min:x_max]

                # Display the cropped face
                cv2.imshow('Cropped Face Outline', cropped_face)
            else:
                if face_area < MIN_FACE_AREA:
                    print("Face too far away; skipping detection.")
                if tilt_angle > MAX_TILT_ANGLE:
                    print("Face is tilted; skipping detection.")

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
