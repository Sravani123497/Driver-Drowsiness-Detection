import cv2
import dlib
from scipy.spatial import distance as dist
import numpy as np
# Function to calculate eye aspect ratio
def eye_aspect_ratio(eye):
A = dist.euclidean(eye[1], eye[5])
B = dist.euclidean(eye[2], eye[4])
C = dist.euclidean(eye[0], eye[3])
ear = (A + B) / (2.0 * C)
return ear
# Load the pre-trained models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# Get the landmarks for the eyes
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)
DRIVING DROWSINESS DETECTION 35 ACE ENGINEERING COLLEGE CSE(DS)
def test_eye_blink_detection(image_path, expected_blinks):
frame = cv2.imread(image_path)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 0)
actual_blinks = 0
for rect in rects:
shape = predictor(gray, rect)
shape = np.array([[p.x, p.y] for p in shape.parts()])
leftEye = shape[lStart:lEnd]
rightEye = shape[rStart:rEnd]
leftEAR = eye_aspect_ratio(leftEye)
rightEAR = eye_aspect_ratio(rightEye)
ear = (leftEAR + rightEAR) / 2.0
if ear < EYE_AR_THRESH:
actual_blinks += 1
print(f"Expected Blinks: {expected_blinks}, Actual Blinks: {actual_blinks}")
assert actual_blinks == expected_blinks, "Blink detection test failed"
# Example test
EYE_AR_THRESH = 0.3
test_eye_blink_detection("test_image.jpg", expected_blinks=2)
