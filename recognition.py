import sys
import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

# Remove TensorFlow warnings
tf.get_logger().setLevel("ERROR")

output_directory = "output_samples"
model = tf.keras.models.load_model(r"models\emosense_finalmodel_82.h5")

# Emotion classes
emotions = ["Neutral", "Happy", "Surprise", "Sadness", "Anger", "Disgust", "Fear"]

# Initialize YOLOv11 face detection model
yolo_face_model = YOLO("yolov8n.pt")  # generic YOLOv8 nano
  # Tiny face model for speed; download automatically

# Detect faces using YOLOv11
def face_detect(frame):
    results = yolo_face_model.predict(frame, conf=0.5, verbose=False)
    faces = []
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                faces.append((x1, y1, x2 - x1, y2 - y1))
    return faces

# Emotion detection on frame
def detect_emotions_in_frame(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_detect(frame)
    for x, y, w, h in faces:
        # Preprocess face ROI
        face_roi = gray[y:y+h, x:x+w]
        resized_face = cv.resize(face_roi, (48, 48))
        normalized_face = resized_face / 255.0
        reshaped_face = np.reshape(normalized_face, (1, 48, 48, 1))

        predictions = model.predict(reshaped_face)
        probabilities = [round(pred * 100, 2) for pred in predictions.tolist()[0]]

        max_index = np.argmax(predictions)
        emotion = f"{emotions[max_index]} {max(probabilities)}%"

        # Draw bounding box and label
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv.putText(frame, emotion, (x, y - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 4, lineType=cv.LINE_AA)
        cv.putText(frame, emotion, (x, y - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, lineType=cv.LINE_AA)
    return frame

# Resize image/video frames
def resize_frame(image):
    if image.shape[1] > 800 or image.shape[1] < 400:
        new_width = 800
        new_height = int((image.shape[0] / image.shape[1]) * new_width)
        image = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)
    return image

# Process single image
def process_image(image_path):
    image = cv.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return

    print("Processing image...")
    result_image = detect_emotions_in_frame(resize_frame(image))

    filename = os.path.basename(image_path)
    file_root, file_ext = os.path.splitext(filename)
    output_path = os.path.join(output_directory, f"{file_root}_result{file_ext}")
    os.makedirs(output_directory, exist_ok=True)
    cv.imwrite(output_path, result_image)
    print(f"Processed image saved as {output_path}")

    cv.imshow("Processed Image", result_image)
    while cv.getWindowProperty("Processed Image", cv.WND_PROP_VISIBLE) >= 1:
        cv.waitKey(100)
    cv.destroyAllWindows()

# Process video
def process_video(video_path):
    video = cv.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return

    filename = os.path.basename(video_path)
    file_root, file_ext = os.path.splitext(filename)
    output_path = os.path.join(output_directory, f"{file_root}_result{file_ext}")
    os.makedirs(output_directory, exist_ok=True)

    fourcc = cv.VideoWriter_fourcc(*"XVID")
    out = cv.VideoWriter(output_path, fourcc, video.get(cv.CAP_PROP_FPS),
                         (int(video.get(3)), int(video.get(4))))

    print("Processing video...")
    while True:
        ret, frame = video.read()
        if not ret:
            break
        result_frame = detect_emotions_in_frame(frame)
        out.write(result_frame)
        cv.imshow("Video Emotion Detection", result_frame)
        if cv.waitKey(1) == ord("q") or cv.getWindowProperty("Video Emotion Detection", cv.WND_PROP_VISIBLE) < 1:
            break

    video.release()
    out.release()
    print(f"Processed video saved as {output_path}")
    cv.destroyAllWindows()

# Real-time webcam
def realtime_emotion_detection():
    video = cv.VideoCapture(0)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        result_frame = detect_emotions_in_frame(frame)
        cv.imshow("Real-time Emotion Detection", result_frame)
        if cv.waitKey(1) == ord("q") or cv.getWindowProperty("Real-time Emotion Detection", cv.WND_PROP_VISIBLE) < 1:
            break
    video.release()
    cv.destroyAllWindows()

# Main
if __name__ == "__main__":
    if len(sys.argv) == 1:
        realtime_emotion_detection()
    elif len(sys.argv) == 2:
        input_path = sys.argv[1]
        if input_path.lower().endswith(("jpg", "jpeg", "png")):
            process_image(input_path)
        elif input_path.lower().endswith(("mp4", "mov", "avi", "mkv")):
            process_video(input_path)
        else:
            print("Unsupported file format. Please provide an image or video file.")
    else:
        print("Usage:")
        print("  python recognition.py                        # Real-time webcam")
        print("  python recognition.py image.jpg/png/jpeg     # Process single image")
        print("  python recognition.py video.mp4/mov/mkv/avi  # Process video")
