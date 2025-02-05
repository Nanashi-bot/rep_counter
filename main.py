import mediapipe as mp
import cv2
import math

model_path = '/home/aditya/codes/rep_counter/pose_landmarker_heavy.task'

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE  # Change to IMAGE for synchronous processing
)


def angles(x1,x2,x3,y1,y2,y3):
    dist1 = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    dist2 = math.sqrt((x3-x2)**2 + (y3-y2)**2)
    num = (x2-x1)*(x2-x3) + (y2-y1)*(y2-y3)
    angle = math.acos(num/(dist1*dist2))
    return math.degrees(angle)


with PoseLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    framenum = 0
    rep_counter = 0
    arr = []
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        cv2.imshow('Pose Detection', frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        framenum += 1
        # Get the pose detection result synchronously
        result = landmarker.detect(mp_image)
        if result and result.pose_landmarks:
            # 12: right shoulder, 14: right elbow, 16: right wrist
            x_rs = int(result.pose_landmarks[0][12].x * frame.shape[1])
            y_rs = int(result.pose_landmarks[0][12].y * frame.shape[0])
            cv2.circle(frame, (x_rs, y_rs), radius=5, color=(0, 0, 255), thickness=-1)

            x_re = int(result.pose_landmarks[0][14].x * frame.shape[1])
            y_re = int(result.pose_landmarks[0][14].y * frame.shape[0])
            cv2.circle(frame, (x_re, y_re), radius=5, color=(0, 255, 0), thickness=-1)

            # 12: right shoulder, 14: right elbow, 16: right wrist
            x_rw = int(result.pose_landmarks[0][16].x * frame.shape[1])
            y_rw = int(result.pose_landmarks[0][16].y * frame.shape[0])
            cv2.circle(frame, (x_rw, y_rw), radius=5, color=(255, 0, 0), thickness=-1)

            if framenum % 50 == 0:
                #print(f"Right Shoulder: {x_rs}, {y_rs}")
                #print(f"Right Elbow: {x_re}, {y_re}")
                #print(f"Right Wrist: {x_rw}, {y_rw}")
                print(arr)

        angle = angles(x_rs, x_re, x_rw, y_rs, y_re, y_rw)
        angletext = f"Angle: {angle}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (10, frame.shape[0] - 10)
        font_scale = 1
        font_color = (255, 255, 255)
        thickness = 2
        line_type = cv2.LINE_AA

        cv2.putText(frame, angletext, (10, frame.shape[0] - 35), font, font_scale, font_color, thickness, line_type)

        # 0 means arm angle < 90 and 1 means arm angle > 90
        if not arr:
            if angle < 60:
                arr.append(0)
            elif angle > 130:
                arr.append(1)
        else:
            if angle < 60 and arr[-1] == 1:
                arr.append(0)
            if angle > 130 and arr[-1] == 0:
                arr.append(1)
        if len(arr) > 2:
            if arr[-1] == 1 and arr[-2] == 0 and arr[-3] == 1:
                rep_counter += 1
                arr.pop()
                arr.pop()

        text = f"Rep counter: {rep_counter}"
        cv2.putText(frame, text, position, font, font_scale, font_color, thickness, line_type)


        cv2.imshow('Pose Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
