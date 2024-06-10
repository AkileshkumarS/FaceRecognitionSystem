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
worker_images_folder = "D:/python projects/photos"  # Use forward slashes for paths

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

# Time window for deduplication in seconds
time_window = 5

last_recognition_time = 0

# Lists to store unique patient encodings
unique_patient_encodings = []
duplicate_patient_encodings = []
duplicate_worker_patient_encodings = []

# Initialize counters
worker_count = 0
patient_count = 0
duplicate_patient_count = 0

with open(csv_file, 'w+', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Category"])

    while True:
        # Capture a photo automatically
        take_photo()

        # Load the captured photo
        image = face_recognition.load_image_file("captured.jpg")
        face_encodings = face_recognition.face_encodings(image)

        if len(face_encodings) > 0:
            encoding = face_encodings[0]

            # Check if the face matches any of the workers
            matches = face_recognition.compare_faces([x[1] for x in worker_face_encodings], encoding, tolerance=face_recognition_threshold)

            if True in matches:
                # Face matches a worker
                worker_count += 1
                writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Worker"])
            else:
                # Face does not match a worker
                patient_count += 1

                # Check if the face is a duplicate
                is_duplicate = False
                for unique_encoding in unique_patient_encodings:
                    if face_recognition.compare_faces([unique_encoding], encoding, tolerance=face_recognition_threshold)[0]:
                        is_duplicate = True
                        break

                if is_duplicate:
                    duplicate_patient_count += 1
                    writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Duplicate Patient"])
                    duplicate_worker_patient_encodings.append(encoding)
                else:
                    unique_patient_encodings.append(encoding)
                    writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Patient"])
        else:
            print("No face found in the captured photo")

        # Deduplicate patient encodings
        current_time = time.time()
        unique_patient_encodings = [x for x in unique_patient_encodings if (current_time - last_recognition_time) < time_window]
        duplicate_worker_patient_encodings = [x for x in duplicate_worker_patient_encodings if (current_time - last_recognition_time) < time_window]

        # Update last recognition time
        last_recognition_time = current_time



