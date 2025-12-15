# Real-Time Mood Detection using Facial Feature Analysis
This project is a computer vision application that detects a user's emotional state (Smiling, Frowning, or Neutral) in real-time using a standard webcam. Unlike deep learning approaches, this system uses fundamental geometric modeling techniques. It isolates the mouth region, extracts edges using the Canny Edge Detector, and fits a parabolic curve to the lips using the RANSAC algorithm. The concavity of the fitted curve determines the user's mood.
## Prerequisites
Ensure you have Python installed (Python 3.12 or 3.11 is recommended for library compatibility).

You will need the following Python libraries:
- opencv-python
- numpy

## Installation
Clone the Repository (or extract the project files):
Navigate to the project directory in your terminal or command prompt.

Install Dependencies:
Run the following command to install the required libraries:
```
pip install opencv-python numpy
```

## Usage
To run the application, execute the main Python script:
```
python mood_detector.py
```
### Selecting the Proper Webcam
If the program opens the wrong camera (e.g., a built-in laptop camera instead of your USB webcam), you can specify the camera index as a command-line argument.

Default Camera (Index 0):
```
python mood_detector.py
```
External Camera (Index 1):
```
python mood_detector.py 1
```
Alternative Camera (Index 2):
```
python mood_detector.py 2
```

## Controls
Quit: Press q on your keyboard while the video window is focused to stop the program.

## Troubleshooting
-"Error: Could not open webcam": This means the camera index is incorrect or another application (like Zoom/Teams) is using the camera. Try changing the index number or closing other apps.
-"ModuleNotFoundError": Ensure you have installed the requirements in the correct Python environment. If using PyCharm, verify the interpreter settings.
