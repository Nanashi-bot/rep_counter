# Bicep Curl Counter

This project tracks and counts the number of bicep curl repetitions performed with the right arm using computer vision.

## Features
- Detects arm movements using a webcam.
- Tracks right-arm bicep curls.
- Counts the number of repetitions in real time.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Nanashi-bot/rep_counter.git
   cd https://github.com/Nanashi-bot/rep_counter.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the main script to start tracking bicep curls:
```bash
python main.py
```

Make sure your webcam is connected and positioned correctly to capture the right arm movements.

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Mediapipe (if used for pose detection)

## How It Works
1. The script captures live video from the webcam.
2. It detects the position of the right shoulder, elbow and wrist.
3. It counts the number of bicep curl reps based on arm movement patterns.

## Contributing
Feel free to submit issues or pull requests to improve the project.

## License
This project is licensed under the MIT License.


