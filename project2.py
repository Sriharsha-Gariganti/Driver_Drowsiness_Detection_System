import numpy as np
import imutils
import time
import timeit
import dlib
import cv2
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from pygame import mixer
from imutils import face_utils
from threading import Thread




def eye_aspect_ratio(eye):
    """Calculate the Eye Aspect Ratio (EAR) for a given eye shape."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def init_open_ear():
    """Initialize the EAR for open eyes."""
    time.sleep(5)
    print("Open init time sleep")
    ear_list = [both_ear] * 7
    global OPEN_EAR
    OPEN_EAR = sum(ear_list) / len(ear_list)
    print("OPEN_EAR =", OPEN_EAR, "\n")

def init_close_ear():
    """Initialize the EAR for closed eyes and calculate threshold."""
    time.sleep(2)
    th_open.join()
    time.sleep(5)
    print("Close init time sleep")
    ear_list = [both_ear] * 7
    global EAR_THRESH
    CLOSE_EAR = sum(ear_list) / len(ear_list)
    EAR_THRESH = (((OPEN_EAR - CLOSE_EAR) / 2) + CLOSE_EAR)
    print("CLOSE_EAR =", CLOSE_EAR, "\n")
    print("EAR_THRESH =", EAR_THRESH, "\n")

def init_message():
    """Play an initialization sound."""
    print("init_message")
    # Placeholder for actual alarm functionality
    print("Playing init sound")

def check_fps(prev_time):
    """Calculate the FPS and return the current time and FPS."""
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
    return current_time, fps

# Global variables
OPEN_EAR = 0
EAR_THRESH = 0
EAR_CONSEC_FRAMES = 25
COUNTER = 0
TIMER_FLAG = False
ALARM_FLAG = False
ALARM_COUNT = 0
RUNNING_TIME = 0
PREV_TERM = 0

# Placeholder for `make_train_data`
def start(num_samples):
    return [0], [0], [0]

def run(features, power, normal, short):
    return 0

# Initialize training data
np.random.seed(30)
power, nomal, short = start(25)

test_data = []
result_data = []
prev_time = time.time()

# Load facial landmark predictor
print("Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Start video stream
print("Starting video stream thread...")
vs = VideoStream(src=0).start()
time.sleep(1.0)

# Initialize EAR thresholds
th_open = Thread(target=init_open_ear)
th_open.daemon = True
th_open.start()
th_close = Thread(target=init_close_ear)
th_close.daemon = True
th_close.start()

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # Ensure frame is valid
    if frame is None:
        print("Error: Frame is None")
        continue

    # Check the frame dtype and shape
    print(f"Frame dtype: {frame.dtype}, shape: {frame.shape}")

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Ensure grayscale image is valid
    if gray is None or gray.dtype != np.uint8:
        print("Error: Grayscale image is invalid or not 8-bit.")
        continue

    rects = detector(gray, 0)
    
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        both_ear = (leftEAR + rightEAR) * 500  # Scale EAR for comparison

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if both_ear < EAR_THRESH:
            if not TIMER_FLAG:
                start_closing = timeit.default_timer()
                TIMER_FLAG = True
                COUNTER += 1
                if COUNTER >= EAR_CONSEC_FRAMES:
                    mid_closing = timeit.default_timer()
                    closing_time = round((mid_closing - start_closing), 3)
                    if closing_time >= RUNNING_TIME:
                        if RUNNING_TIME == 0:
                            CUR_TERM = timeit.default_timer()
                            OPENED_EYES_TIME = round((CUR_TERM - PREV_TERM), 3)
                            PREV_TERM = CUR_TERM
                            RUNNING_TIME = 1.75
                            RUNNING_TIME += 2
                            ALARM_FLAG = True
                            ALARM_COUNT += 1
                            print(f"{ALARM_COUNT}st ALARM")
                            print(f"The time eyes are open before the alarm went off: {OPENED_EYES_TIME}")
                            print(f"Closing time: {closing_time}")
                            test_data.append([OPENED_EYES_TIME, round(closing_time * 10, 3)])
                            result = run([OPENED_EYES_TIME, closing_time * 10], power, nomal, short)
                            result_data.append(result)
                            # Placeholder for actual alarm functionality
                            print(f"Triggering alarm with result {result}")

                        else:
                            COUNTER = 0
                            TIMER_FLAG = False
    
    # Check and display FPS
    prev_time, fps = check_fps(prev_time)
    cv2.putText(frame, "fps : {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Debugging: Check image type and shape before displaying
    print(f"Frame dtype: {frame.dtype}, shape: {frame.shape}")

    # Ensure frame is in the correct format
    if frame is not None and frame.ndim == 3 and frame.shape[2] == 3:
        cv2.imshow("Frame", frame)
    else:
        print("Warning: Frame is not in the correct format for display.")

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
vs.stop()
cv2.destroyAllWindows()
