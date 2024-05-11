import cv2  # Import OpenCV (Open Source Computer Vision Library)
from picamera2 import Picamera2 # Import for controlling the Raspberry Pi camera module
import time  # Import for timing functions

picam2 = Picamera2()  # Create a Picamera2 object
picam2.preview_configuration.main.size = (640, 720)  # Set the camera resolution
picam2.preview_configuration.main.format = "RGB888" # Set the image format
picam2.preview_configuration.align() # Align camera configuration settings for optimal capture and preview.
picam2.configure("preview") # start camera preview
picam2.start()  # Start capturing video frames
time.sleep(2)  # Wait for two seconds to let the camera initialize

recognizer = cv2.face.LBPHFaceRecognizer_create()  # Create a face recognizer object
recognizer.read('/home/dean/Desktop/face_github_ealon/trainer.yml')  # Load the trained face recognition model
cascadePath = "/home/dean/Desktop/face_github_ealon/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);  # Load the face detection model
font = cv2.FONT_HERSHEY_SIMPLEX  # Define a font for text display

id = 0  # Initialize a counter (likely for assigning new IDs)
names = ['None', 'Ealon', 'Paula', 'Ilza', 'Z', 'W']  # List of names corresponding to IDs

while True:  # Start the main loop for real-time processing
    img = picam2.capture_array()  # Capture an image frame from the camera
    img = cv2.flip(img, -1)  # Flip the image for a mirror-like view 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    faces = faceCascade.detectMultiScale(gray, 1.2, 5, minSize=(200, 200))  # Detect faces!

    for (x, y, w, h) in faces:  # Loop through detected faces
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a green rectangle

        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])  # Try to recognize the face

        if (confidence < 35):  # If the recognition confidence is high enough:
            face_name = names[id]     # Get the name from the 'names' list
            confidence = "  {0}%".format(round(100 - confidence)) 
        else:
            face_name = "unknown" 
            confidence = "  {0}%".format(round(100 - confidence))

        # Add labels to the image
        cv2.putText(img, face_name, (x + 5, y - 5), font, 1, (255, 255, 255), 2) 
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)  #  Display the image

    k = cv2.waitKey(10) & 0xff  
    if k == 27:  # Press 'ESC' to quit
        break

# Cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cv2.destroyAllWindows()  # Close all OpenCV windows
