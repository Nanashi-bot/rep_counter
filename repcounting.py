from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import mediapipe as mp
import cv2
import time
import math

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def angles(x1,x2,x3,y1,y2,y3):
    return 2
    distance = math.sqrt(((x2-x1)**2 + (y2-y1)**2) * ((x3-x2)**2 + (y3-y2)**2))
    num = (x2-x1)*(x2-x3) + (y2-y1)*(y2-y3)
    angle = math.acos(num/distance)
    return angle



stop_loop = False
rep_counter = 0
# Create a pose landmarker instance with the live stream mode:
def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    dot_x = result.pose_landmarks[0][0].x
    dot_y = result.pose_landmarks[0][0].y
    # 12 14 16
    # # Draw a red dot on the image at the 
    print(dot_x, dot_y)
    dot_x = dot_x * output_image.width
    dot_y = dot_y * output_image.height
    global rep_counter
    output_image_np = output_image.numpy_view()
    annotated_image = draw_landmarks_on_image(output_image_np, result)
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    # angle = angles()
    # text2 = f"Angle: {angle}"
    text1 = f"Rep counter: {rep_counter}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (10, annotated_image.shape[0] - 10)  # Bottom-left corner of the frame
    font_scale = 1
    font_color = (255, 255, 255)  # White color
    thickness = 2
    line_type = cv2.LINE_AA
    cv2.putText(annotated_image, text1, position, font, font_scale, font_color, thickness, line_type)
    cv2.circle(annotated_image, (int(dot_x), int(dot_y)), radius=5, color=(0, 0, 255), thickness=-1)
    # print('pose landmarker result: {}'.format(result))
    cv2.imshow("annotated",annotated_image)
    rep_counter += 1
    cv2.waitKey(1)
    print(timestamp_ms,format(result.pose_landmarks[0][12]))

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_heavy.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

# Bottom function is useless
def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image


with PoseLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    # start_time = time.time()
    frame_timestamp_ms = 0
    while cap.isOpened():
        success, frame = cap.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        # frame_timestamp_ms = int(time.time() - start_time)
        frame_timestamp_ms += 1
        landmarker.detect_async(mp_image, frame_timestamp_ms)