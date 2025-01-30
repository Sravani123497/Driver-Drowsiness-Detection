import cv2
import dlib
from scipy.spatial import distance as dist
# Define constants
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3
# Initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0
# Load the pre-trained models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# Eye aspect ratio calculation
def eye_aspect_ratio(eye):
A = dist.euclidean(eye[1], eye[5])
B = dist.euclidean(eye[2], eye[4])
C = dist.euclidean(eye[0], eye[3])
ear = (A + B) / (2.0 * C)
return ear
# Get the landmarks for the eyes
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)
# Start the video stream
cap = cv2.VideoCapture(0)
while True:
ret, frame = cap.read()
if not ret:
break
DRIVING DROWSINESS DETECTION 31 ACE ENGINEERING COLLEGE CSE(DS)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 0)
for rect in rects:
shape = predictor(gray, rect)
shape = face_utils.shape_to_np(shape)
leftEye = shape[lStart:lEnd]
rightEye = shape[rStart:rEnd]
leftEAR = eye_aspect_ratio(leftEye)
rightEAR = eye_aspect_ratio(rightEye)
ear = (leftEAR + rightEAR) / 2.0
# Compute the convex hull for the eyes and visualize it
leftEyeHull = cv2.convexHull(leftEye)
rightEyeHull = cv2.convexHull(rightEye)
cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
if ear < EYE_AR_THRESH:
COUNTER += 1
else:
if COUNTER >= EYE_AR_CONSEC_FRAMES:
TOTAL += 1
COUNTER = 0
cv2.putText(frame, f"Blinks: {TOTAL}", (10, 30),
cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.imshow("Frame", frame)
if cv2.waitKey(1) & 0xFF == ord('q'):
break
cap.release()
cv2.destroyAllWindows()
