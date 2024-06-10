import cv2
import face_recognition
import csv
import os
import time
from datetime import datetime

# Function to capture a photo from the webcam
def take_photo(filename='captured.jpg'):
    cap = cv2.VideoCapture(0)

    # Capture a single frame
    ret, frame = cap.read()

    # Save the captured frame as an image
    cv2.imwrite(filename, frame)

    # Release the capture object
    cap.release()

# Set the path to the directory where the worker images are stored
worker_images_folder = r"C:\Users\Sarvesh\Downloads\python projects\photos"

# Load face encodings for workers (your photos) from a folder
worker_face_encodings = []

# Load face encodings from each image in the folder
for filename in os.listdir(worker_images_folder):
    image_path = os.path.join(worker_images_folder, filename)

    # Skip directories
    if os.path.isdir(image_path):
        print(f"Skipping directory: {image_path}")
        continue

    print("Processing image:", image_path)  # Debugging print statement
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)

    if len(face_encodings) > 0:
        encoding = face_encodings[0]
        worker_face_encodings.append((filename, encoding))
    else:
        print(f"No face found in {image_path}")

# Set a threshold for face recognition
face_recognition_threshold = 0.6

# Initialize variables and open the CSV file for writing
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
csv_file = f"{current_date}.csv"

worker_count = 0
patient_count = 0

# Time window for deduplication in seconds
time_window = 5

last_recognition_time = 0

with open(csv_file, 'w+', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Timestamp", "Filename"])

    while True:
        # Capture a photo automatically
        take_photo()

        # Load the captured photo
        captured_image = face_recognition.load_image_file('captured.jpg')
        captured_encodings = face_recognition.face_encodings(captured_image)

        if len(captured_encodings) > 0:
            captured_encoding = captured_encodings[0]

            # Perform face recognition
            found_worker = False
            for filename, encoding in worker_face_encodings:
                face_distance = face_recognition.face_distance([encoding], captured_encoding)
                if face_distance[0] < face_recognition_threshold:
                    found_worker = True
                    current_time = time.time()
                    # Check if enough time has passed since the last recognition
                    if current_time - last_recognition_time > time_window:
                        name = "You"
                        worker_count += 1
                        last_recognition_time = current_time
                    else:
                        name = "Duplicate Worker"
                    break

            if not found_worker:
                name = "Patient"
                patient_count += 1

            current_time_str = now.strftime("%H-%M-%S")
            writer.writerow([name, current_time_str, "captured.jpg"])

            print("Total Worker Count:", worker_count)
            print("Total Patient Count:", patient_count)

        # Wait for a few seconds before capturing the next photo
        time.sleep(5)
