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

with open(csv_file, 'w+', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Category"])

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
            found_patient = False
            category = "Unknown"

            for filename, encoding in worker_face_encodings:
                face_distance = face_recognition.face_distance([encoding], captured_encoding)
                if face_distance[0] < face_recognition_threshold:
                    found_worker = True
                    current_time = time.time()
                    # Check if enough time has passed since the last recognition
                    if current_time - last_recognition_time > time_window:
                        category = "Worker"
                        last_recognition_time = current_time
                    else:
                        category = "Worker Duplicate Patient"
                    break

            if not found_worker:
                for patient_encoding in unique_patient_encodings:
                    face_distance = face_recognition.face_distance([patient_encoding], captured_encoding)
                    if face_distance[0] < face_recognition_threshold:
                        found_patient = True
                        category = "Duplicate Patient"
                        duplicate_patient_encodings.append(captured_encoding)
                        break

                if not found_patient:
                    category = "Patient"
                    unique_patient_encodings.append(captured_encoding)

            current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([current_time_str, category])

            print("Timestamp:", current_time_str)
            print("Category:", category)

            # Draw a rectangle around the face
            face_locations = face_recognition.face_locations(captured_image)
            for face_location in face_locations:
                top, right, bottom, left = face_location
                cv2.rectangle(captured_image, (left, top), (right, bottom), (0, 255, 0), 2)

            # Display the live video feed
            cv2.imshow("Live Video", captured_image)
            cv2.waitKey(1)  # Adjust waitKey to control display speed

        # Wait for a few seconds before capturing the next photo
        time.sleep(5)
