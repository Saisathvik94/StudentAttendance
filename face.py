import os
import face_recognition
import cv2
import numpy as np

def register_students(image_folder):
    student_images = []
    student_names = []
    
    # Load all the images and their corresponding names
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Find face encodings
            face_encodings = face_recognition.face_encodings(image_rgb)
            if face_encodings:  # Ensure at least one face is found
                student_images.append(face_encodings[0])
                student_names.append(os.path.splitext(filename)[0])  # Use filename as student name

    return student_images, student_names
import pandas as pd
from datetime import datetime

def mark_attendance(student_name):
    # Load or create an attendance DataFrame
    filename = 'attendance.xlsx'
    
    # If file exists, load it; else create a new one
    if os.path.exists(filename):
        df = pd.read_excel(filename)
    else:
        df = pd.DataFrame(columns=["Name", "Date", "Status"])

    # Get today's date
    today_date = datetime.today().strftime('%Y-%m-%d')
    
    # Check if today's attendance for the student is already recorded
    if student_name in df['Name'].values and today_date in df['Date'].values:
        df.loc[(df['Name'] == student_name) & (df['Date'] == today_date), 'Status'] = 'Present'
    else:
        df = df.append({"Name": student_name, "Date": today_date, "Status": "Present"}, ignore_index=True)

    # Save the DataFrame to the Excel file
    df.to_excel(filename, index=False)

def recognize_faces(student_encodings, student_names):
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find all faces in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare the captured face with the known student faces
            matches = face_recognition.compare_faces(student_encodings, face_encoding)
            name = "Unknown"
            
            # If there's a match, get the student's name
            if True in matches:
                first_match_index = matches.index(True)
                name = student_names[first_match_index]
                print(f"Recognized: {name}")
                mark_attendance(name)  # Mark attendance as 'Present'
            
            # Draw a box around the face and label it
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, bottom + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Display the resulting image
        cv2.imshow('Attendance System', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
def main():
    # Path to folder where student images are stored
    image_folder = "student_faces/"
    
    # Register students and get their face encodings
    student_encodings, student_names = register_students(image_folder)
    
    # Start the face recognition and attendance marking process
    recognize_faces(student_encodings, student_names)

if __name__ == "__main__":
    main()
