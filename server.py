# server.py
from flask import Flask, request, jsonify
import cv2, mediapipe as mp
import numpy as np
import threading

app = Flask(__name__)

# Default settings
settings = {
    "face_confidence": 0.5,
    "hand_confidence": 0.5
}

mp_face = mp.solutions.face_detection.FaceDetection(settings["face_confidence"])
mp_hand = mp.solutions.hands.Hands(min_detection_confidence=settings["hand_confidence"])

@app.route("/process", methods=["POST"])
def process():
    import time
    start = time.time()
    img_data = request.files["image"].read()
    recv_t = time.time()
    npimg = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Detection
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = mp_face.process(img_rgb)
    hands = mp_hand.process(img_rgb)
    det_t = time.time()

    # draw
    h, w, _ = img.shape
    if faces.detections:
        for d in faces.detections:
            bbox = d.location_data.relative_bounding_box
            x, y, bw, bh = int(bbox.xmin*w), int(bbox.ymin*h), int(bbox.width*w), int(bbox.height*h)
            cv2.rectangle(img, (x,y), (x+bw, y+bh), (0,255,0), 2)
    if hands.multi_hand_landmarks:
        for handLm in hands.multi_hand_landmarks:
            pts = [(int(lm.x*w), int(lm.y*h)) for lm in handLm.landmark]
            for pt in pts:
                cv2.circle(img, pt, 5, (255,0,0), -1)

    # encode back
    _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    out_blob = buf.tobytes()
    send_t = time.time()

    return (out_blob, 200, {
        "Content-Type": "image/jpeg",
        "X-t_recv": str(recv_t - start),
        "X-t_det": str(det_t - recv_t),
        "X-t_send": str(send_t - det_t),
        "X-total": str(send_t - start)
    })

@app.route("/settings", methods=["POST"])
def update_settings():
    data = request.json
    if "face_confidence" in data:
        settings["face_confidence"] = float(data["face_confidence"])
    if "hand_confidence" in data:
        settings["hand_confidence"] = float(data["hand_confidence"])
    global mp_face, mp_hand
    mp_face = mp.solutions.face_detection.FaceDetection(settings["face_confidence"])
    mp_hand = mp.solutions.hands.Hands(min_detection_confidence=settings["hand_confidence"])
    return jsonify(settings)

if __name__=="__main__":
    app.run(host='0.0.0.0', port=5000)
