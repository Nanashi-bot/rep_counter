import mediapipe as mp
import cv2

model_path = '/home/aditya/codes/pose/pose_landmarker_lite.task'

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE  # Change to IMAGE for synchronous processing
)

with PoseLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        cv2.imshow('Pose Detection', frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

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


        cv2.imshow('Pose Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
