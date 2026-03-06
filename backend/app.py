from flask import Flask, render_template, jsonify
import cv2
import numpy as np
import threading
from tensorflow.keras.models import load_model

app = Flask(__name__,
            template_folder="../templates",
            static_folder="../static")

# Load emotion model
model = load_model("../model/emotion_model.h5")

# Emotion labels
emotions = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

camera_running = False

# Load face detector
face_cascade = cv2.CascadeClassifier(
cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def calculate_attention(emotion):

    attention_map = {
        "Happy": 90,
        "Neutral": 80,
        "Surprise": 75,
        "Sad": 40,
        "Fear": 30,
        "Angry": 25,
        "Disgust": 20
    }

    return attention_map.get(emotion,50)


def start_camera():

    global camera_running
    camera_running = True

    cap = cv2.VideoCapture(1)

    while camera_running:

        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x,y,w,h) in faces:

            face = gray[y:y+h, x:x+w]

            face = cv2.resize(face,(48,48))
            face = face/255.0
            face = np.reshape(face,(1,48,48,1))

            prediction = model.predict(face)
            emotion_index = np.argmax(prediction)

            emotion = emotions[emotion_index]

            attention = calculate_attention(emotion)

            label = f"{emotion} | Attention: {attention}%"

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            cv2.putText(frame,label,(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,(0,255,0),2)

        cv2.imshow("AI Attention Monitoring", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    camera_running = False


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start_camera")
def start_camera_route():

    global camera_running

    if not camera_running:
        thread = threading.Thread(target=start_camera)
        thread.start()

    return jsonify({"status":"camera started"})


@app.route("/stop_camera")
def stop_camera():

    global camera_running
    camera_running = False

    return jsonify({"status":"camera stopped"})


if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)
