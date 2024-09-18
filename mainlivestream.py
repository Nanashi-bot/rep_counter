from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import mediapipe as mp
import cv2
import time

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the live stream mode:
def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # print('pose landmarker result: {}'.format(result))
    output_image_np = np.array(output_image.numpy_view(), dtype=np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=output_image_np)
    annotated_image = draw_landmarks_on_image(output_image_np, result)
    cv2.imshow("annotated",cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)

    # output_image_np = np.array(output_image.numpy_view(), dtype=np.uint8)
    # dot_x = result.pose_landmarks[0][0].x * output_image_np.shape[1]
    # dot_y = result.pose_landmarks[0][0].y * output_image_np.shape[0]
        
    # # Draw a red dot on the image at the specified coordinate
    # # Note: OpenCV uses (B, G, R) format for colors, so red is (0, 0, 255)
    # cv2.circle(output_image_np, (int(dot_x), int(dot_y)), radius=5, color=(0, 0, 255), thickness=-1)
        
    # # Display the image with the red dot
    # cv2.imshow('PoseLandmarker Output', output_image_np)
    # cv2.waitKey(0) 
    # print(timestamp_ms,format(result.pose_landmarks[0][0]))
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
        # frame_timestamp_ms = frame_index * frame_interval
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_timestamp_ms+=1
        # frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        # # Create a MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        # frame_timestamp_ms = int(time.time() - start_time)
        landmarker.detect_async(mp_image, frame_timestamp_ms)
        # height, width, _ = frame.shape
        # Normalized coordinates (0.0 to 1.0)
        # normalized_x = 0.600645124912262
        # normalized_y = 0.6174551844596863
        # print(poseLandmarkerResult)
        # Convert normalized coordinates to pixel coordinates
        # x = int(normalized_x * width)
        # y = int(normalized_y * height)
        # cv2.circle(frame, (x,y), radius=5, color=(0, 0, 255), thickness=-1) 
        # Display the frame with landmarks
        # cv2.imshow('Pose Detection', frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
