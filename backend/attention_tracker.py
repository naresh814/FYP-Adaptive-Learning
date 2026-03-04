import cv2
import csv
import os
import time
from datetime import datetime
from deepface import DeepFace

# CSV file name
csv_file = "engagement_data.csv"

# Create CSV file if it doesn't exist
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "child_id",
            "timestamp",
            "engagement",
            "face_detected",
            "emotion"
        ])

# Function to save engagement
def save_engagement(child_id, engagement, face_detected, emotion):

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            child_id,
            timestamp,
            engagement,
            face_detected,
            emotion
        ])


# ✅ MAIN TRACKING FUNCTION (IMPORTANT)
def start_tracking():

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    child_id = 1
    last_saved_time = 0

    save_interval = 3  # seconds

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Detect emotion
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        # Determine face_detected
        if len(faces) > 0:
            face_detected = "yes"
        else:
            face_detected = "no"

        # Determine engagement
        if face_detected == "yes" and emotion in ["happy", "neutral"]:
            engagement = "engaged"
        else:
            engagement = "disengaged"

        # Draw face rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(frame,
                          (x, y),
                          (x+w, y+h),
                          (0, 255, 0),
                          2)

        # Save data every few seconds
        current_time = time.time()

        if current_time - last_saved_time > save_interval:

            save_engagement(
                child_id,
                engagement,
                face_detected,
                emotion
            )

            print("Saved:", engagement, "| Face:", face_detected, "| Emotion:", emotion)

            last_saved_time = current_time

        # Show emotion text
        cv2.putText(frame,
                    f"Emotion: {emotion}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2)

        # Show engagement text
        cv2.putText(frame,
                    f"Engagement: {engagement}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,0,0),
                    2)

        # Show webcam
        cv2.imshow("Engagement Tracker", frame)

        # Exit on ESC key
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
