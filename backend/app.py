from flask import Flask, jsonify
import threading
import attention_tracker  # your existing tracker file

app = Flask(__name__)

tracking_status = {
    "running": False
}

# Home route
@app.route("/")
def home():
    return "Adaptive Learning Backend Running"

# System status route
@app.route("/status")
def status():
    return jsonify({
        "system": "active",
        "tracking_running": tracking_status["running"],
        "dataset": "FER2013",
        "emotion_model": "DeepFace"
    })

# Start tracking route
# @app.route("/start-tracking")
# def start_tracking():
#     if not tracking_status["running"]:
#         tracking_status["running"] = True
#         thread = threading.Thread(target=attention_tracker.start_tracking)
#         thread.start()
#         return jsonify({"message": "Tracking started"})
#     else:
#         return jsonify({"message": "Already running"})

@app.route("/start-tracking")
def start_tracking():
    attention_tracker.start_tracking()
    return jsonify({"message": "Tracking started"})




if __name__ == "__main__":
    app.run(debug=True)
