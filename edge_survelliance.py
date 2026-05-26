import streamlit as st
import cv2
import os
import sqlite3
import threading
import asyncio
from datetime import datetime
from ultralytics import YOLO
from dotenv import load_dotenv
import face_recognition
from telegram import Bot

# ---------------- ENV ----------------
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SAVE_DIR = "unauthorized_captures"
KNOWN_DIR = "known_faces"
DB_NAME = "surveillance_logs.db"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(KNOWN_DIR, exist_ok=True)

# ---------------- DATABASE ----------------
def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            track_id INTEGER,
            identity TEXT,
            image_path TEXT
        )
        ''')

def log_event(track_id, identity, path):
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute(
            "INSERT INTO detections (timestamp, track_id, identity, image_path) VALUES (?, ?, ?, ?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), track_id, identity, path)
        )
        conn.commit()

# ---------------- TELEGRAM (ASYNC) ----------------
async def send_telegram_async(photo_path, identity, track_id):
    if not TOKEN or not CHAT_ID:
        return
    
    try:
        bot = Bot(token=TOKEN)
        caption = f"⚠️ ALERT\nID: {track_id}\n{identity}"

        with open(photo_path, "rb") as photo:
            await bot.send_photo(chat_id=CHAT_ID, photo=photo, caption=caption)
    
    except Exception as e:
        print("Telegram error:", e)

# Run async in thread (IMPORTANT)
def send_telegram(photo_path, identity, track_id):
    asyncio.run(send_telegram_async(photo_path, identity, track_id))

# ---------------- FACE DATA ----------------
def load_known_faces():
    encodings, names = [], []
    for file in os.listdir(KNOWN_DIR):
        if file.lower().endswith((".jpg", ".png")):
            img = face_recognition.load_image_file(os.path.join(KNOWN_DIR, file))
            enc = face_recognition.face_encodings(img)
            if enc:
                encodings.append(enc[0])
                names.append(os.path.splitext(file)[0])
    return encodings, names

# ---------------- STREAMLIT UI ----------------
st.set_page_config(layout="wide")
st.title("🛡️ AI Surveillance Dashboard")

run = st.sidebar.toggle("Start System")

col1, col2 = st.columns([2,1])
frame_window = col1.image([])
log_box = col2.empty()

# ---------------- MAIN ----------------
if run:
    init_db()

    model = YOLO("yolov8n.pt")
    known_encodings, known_names = load_known_faces()

    cap = cv2.VideoCapture(0)

    last_alert = {}
    COOLDOWN = 10
    frame_count = 0

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not accessible")
            break

        frame_count += 1

        results = model.track(frame, persist=True)

        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                if cls != 0:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0]) if box.id is not None else -1

                h, w, _ = frame.shape
                x1, y1 = max(0,x1), max(0,y1)
                x2, y2 = min(w,x2), min(h,y2)

                person = frame[y1:y2, x1:x2]

                identity = "Unknown"

                # 🔥 Run face recognition every 5 frames
                if frame_count % 5 == 0:
                    face_locs = face_recognition.face_locations(person)
                    face_encs = face_recognition.face_encodings(person, face_locs)

                    for enc in face_encs:
                        matches = face_recognition.compare_faces(known_encodings, enc)
                        if True in matches:
                            identity = known_names[matches.index(True)]
                            break

                color = (0,255,0) if identity != "Unknown" else (0,0,255)

                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, f"{identity} ({track_id})",
                            (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            color, 2)

                # 🔥 Cooldown logic
                now = datetime.now().timestamp()
                if identity == "Unknown":
                    if track_id not in last_alert or now - last_alert[track_id] > COOLDOWN:

                        filename = f"intruder_{track_id}_{int(now)}.jpg"
                        path = os.path.join(SAVE_DIR, filename)

                        cv2.imwrite(path, person)
                        log_event(track_id, identity, path)

                        # 🔥 Async Telegram via thread
                        threading.Thread(
                            target=send_telegram,
                            args=(path, identity, track_id),
                            daemon=True
                        ).start()

                        last_alert[track_id] = now

        frame_window.image(frame, channels="BGR")

        # ---------------- DASHBOARD ----------------
        with sqlite3.connect(DB_NAME) as conn:
            rows = conn.execute(
                "SELECT * FROM detections ORDER BY id DESC LIMIT 5"
            ).fetchall()

        log_box.write("### Recent Alerts")

        for row in rows:
            log_box.write(f"{row[1]} | ID:{row[2]} | {row[3]}")
            if os.path.exists(row[4]):
                log_box.image(row[4], width=150)

    cap.release()