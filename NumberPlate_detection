import cv2
import numpy as np
import pytesseract
import datetime
import sqlite3
import os
import csv

frameWidth = 640  # Frame Width
frameHeight = 480  # Frame Height
plateCascade = cv2.CascadeClassifier(
    r"C:\Users\Anabhra\OneDrive\Desktop\number-plate-detection\number_plate.xml")
minArea = 500

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

# Flag to track if an entry has already been made for a number plate
number_plate_entry = False

# Define the path to save the CSV file and images
base_directory = r"C:\Users\Anabhra\OneDrive\Desktop\number-plate-detection"

# Define paths for the CSV file and images
csv_file_path = os.path.join(base_directory, "number_plate_log.csv")
# Images saved in IMAGES folder
image_directory = os.path.join(base_directory, "IMAGES")

# Create CSV file and add headers if it doesn't exist
if not os.path.exists(csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Number Plate', 'Entry Time'])  # Write header

# Ensure the directory exists for saving images
if not os.path.exists(image_directory):
    os.makedirs(image_directory)

while True:
    success, img = cap.read()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    numberPlates = plateCascade.detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in numberPlates:
        area = w * h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "NumberPlate", (x, y - 5),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            imgRoi = img[y:y + h, x:x + w]
            cv2.imshow("ROI", imgRoi)

            if number_plate_entry:
                continue

            # Use Tesseract to extract the text from the image region
            plate_number = pytesseract.image_to_string(imgRoi)

            # Clean the plate_number by stripping any leading or trailing whitespace or newlines
            plate_number = plate_number.strip()

            # Print the extracted number plate text
            print("Number Plate:", plate_number)

            if plate_number != "":
                # Save the image with a unique name based on the timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                image_filename = os.path.join(
                    image_directory, timestamp + ".jpg")
                cv2.imwrite(image_filename, imgRoi)

                # Set the entry flag to True and record the entry time
                number_plate_entry = True
                entry_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Print the data before saving to CSV
                print(f"Saving to CSV: {plate_number}, {entry_time}")

                # Save the extracted number plate and entry time to the CSV file
                with open(csv_file_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([plate_number, entry_time])

                # Display the "Scan Saved" message
                cv2.rectangle(img, (0, 200), (640, 300),
                              (0, 255, 0), cv2.FILLED)
                cv2.putText(img, "Scan Saved", (15, 265),
                            cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
                cv2.imshow("Result", img)
                cv2.waitKey(0)

    # Show the live webcam feed
    cv2.imshow("Webcam", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
