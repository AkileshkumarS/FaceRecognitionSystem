from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import cv2
import face_recognition
import csv
import os
import time

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)

class RecognitionEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.String(20), nullable=False)
    category = db.Column(db.String(20), nullable=False)

@app.route('/')
def home():
    entries = RecognitionEntry.query.all()
    return render_template('index.html', entries=entries)

def take_photo(filename='captured.jpg'):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cv2.imwrite(filename, frame)
    cap.release()

worker_images_folder = r"C:\Users\Sarvesh\Downloads\python projects\photos"
worker_face_encodings = []

for filename in os.listdir(worker_images_folder):
    image_path = os.path.join(worker_images_folder, filename)
    if os.path.isdir(image_path):
        continue
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)
    if len(face_encodings) > 0:
        encoding = face_encodings[0]
        worker_face_encodings.append((filename, encoding))

face_recognition_threshold = 0.6
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
csv_file = f"{current_date}.csv"
time_window = 5
last_recognition_time = 0
unique_patient_encodings = []
duplicate_patient_encodings = []
worker_count = 0
patient_count = 0
duplicate_patient_count = 0

with open(csv_file, 'w+', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Category"])

    while True:
        take_photo()
        captured_image = face_recognition.load_image_file('captured.jpg')
        captured_encodings = face_recognition.face_encodings(captured_image)

        if len(captured_encodings) > 0:
            captured_encoding = captured_encodings[0]
            found_worker = False
            found_patient = False
            category = "Unknown"

            for filename, encoding in worker_face_encodings:
                face_distance = face_recognition.face_distance([encoding], captured_encoding)
                if face_distance[0] < face_recognition_threshold:
                    found_worker = True
                    current_time = time.time()
                    if current_time - last_recognition_time > time_window:
                        category = "Worker"
                        last_recognition_time = current_time
                        worker_count += 1
                    else:
                        category = "Duplicate Patient"
                        duplicate_patient_count += 1
                    break

            if not found_worker:
                for patient_encoding in unique_patient_encodings:
                    face_distance = face_recognition.face_distance([patient_encoding], captured_encoding)
                    if face_distance[0] < face_recognition_threshold:
                        found_patient = True
                        category = "Duplicate Patient"
                        duplicate_patient_encodings.append(captured_encoding)
                        duplicate_patient_count += 1
                        break

                if not found_patient:
                    category = "Patient"
                    unique_patient_encodings.append(captured_encoding)
                    patient_count += 1

            current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([current_time_str, category])

        time.sleep(5)

if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)
