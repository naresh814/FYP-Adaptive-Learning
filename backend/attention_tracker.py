import cv2
import mediapipe as mp

from tensorflow.keras.models import load_model

model =  load_model("emotion_model.h5")

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Start webcam
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Camera not detected")
    exit()

attention_score = 100

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    status = "Not Detected"

    if results.multi_face_landmarks:

        status = "Focused"

        for face_landmarks in results.multi_face_landmarks:

            for lm in face_landmarks.landmark:

                h, w, c = frame.shape
                x = int(lm.x * w)
                y = int(lm.y * h)

                cv2.circle(frame, (x, y), 1, (0,255,0), -1)

    else:
        attention_score -= 1
        status = "Distracted"

    cv2.putText(frame, f"Attention Score: {attention_score}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, f"Status: {status}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Attention Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
