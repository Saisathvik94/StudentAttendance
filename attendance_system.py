import cv2
import os
import numpy as np
from datetime import datetime

# Step 1: Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Step 2: Create the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Step 3: Register Students (Training Data)
def train_face_recognizer(image_folder):
    faces = []
    labels = []
    student_names = {}
    label_counter = 0
    
    # Loop through the folder containing the student images
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces in the image
            faces_detected = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
            
            for (x, y, w, h) in faces_detected:
                # Get the face region
                face_region = gray_image[y:y+h, x:x+w]
                
                # Assign a unique label to each student
                faces.append(face_region)
                labels.append(label_counter)
                student_names[label_counter] = os.path.splitext(filename)[0]  # Save student name
                
            label_counter += 1

    # Train the recognizer with the collected faces and labels
    recognizer.train(faces, np.array(labels))
    recognizer.save('face_trainer.yml')
    
    return student_names

# Step 4: Mark Attendance in a Text File
def mark_attendance(student_name):
    today_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Open the attendance file and append the attendance record
    with open("attendance.txt", "a") as file:
        file.write(f"{student_name} - Present - {today_date}\n")
    print(f"Attendance marked for {student_name} at {today_date}")

# Step 5: Real-time Face Recognition for Attendance
def recognize_faces(student_names):
    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)

    # Keep track of the students who have been marked present
    recognized_students = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        for (x, y, w, h) in faces_detected:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Recognize the face in the face region
            face_region = gray[y:y + h, x:x + w]
            label, confidence = recognizer.predict(face_region)
            
            # If the recognition confidence is below a threshold, it's a valid recognition
            if confidence < 100:  # You can adjust this threshold as needed
                name = student_names[label]
                confidence_text = f"Confidence: {round(100 - confidence)}%"
                cv2.putText(frame, f"{name} - {confidence_text}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                
                # Mark attendance only once per detected person in the session
                if name not in recognized_students:
                    recognized_students.add(name)
                    mark_attendance(name)
            else:
                name = "Unknown"
                cv2.putText(frame, f"{name} - Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Show the video feed
        cv2.imshow("Attendance System", frame)
        
        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main Function
def main():
    # Path to the folder containing student images
    image_folder = "student_faces"
    
    # Step 1: Train the recognizer with student images
    student_names = train_face_recognizer(image_folder)
    
    # Step 2: Start recognizing faces and mark attendance
    recognize_faces(student_names)

if __name__ == "__main__":
    main()

