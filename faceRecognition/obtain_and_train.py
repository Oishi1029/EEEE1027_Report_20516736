
from picamera2 import Picamera2 # Import for controlling the Raspberry Pi camera
import time  # Import for timing functions
import cv2  # Import  OpenCV for image processing
import numpy as np  # Import NumPy for numerical operations
from PIL import Image  # Import for image manipulation
import os  # Import for file path operations


picam2 = Picamera2()  # Create a Picamera2 object to represent the camera
picam2.preview_configuration.main.size = (640,720)  # Set the image resolution
picam2.preview_configuration.main.format = "RGB888" # Set image format
picam2.preview_configuration.align()  # Align camera settings
picam2.configure("preview")  # Configure camera with the provided settings
picam2.start()  # Start capturing frames from the camera

time.sleep(2)  # Wait for 2 seconds to allow the camera to initialize

face_detector = cv2.CascadeClassifier('/home/dean/Desktop/face_github_ealon/haarcascade_frontalface_default.xml')  # Load pre-trained face detection model

# Create face recognizer and face detector objects
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("/home/dean/Desktop/face_github_ealon/haarcascade_frontalface_default.xml");

face_id = input('\n enter user id end press <return> ==>  ')  # Get user ID from user input

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
count = 0  # Initialize a counter to track the number of images captured for a user


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]  # Get paths of all images in the dataset directory
    faceSamples = []  # Empty list to store face images
    ids = []  # Empty list to store corresponding ID labels

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')  # Open image as grayscale using PIL
        img_numpy = np.array(PIL_img, 'uint8')  # Convert PIL image to NumPy array

        id = int(os.path.split(imagePath)[-1].split(".")[1])  # Extract ID from filename
        faces = detector.detectMultiScale(img_numpy)  # Detect faces within the image

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])  # Add detected face region to faceSamples
            ids.append(id)  # Add the corresponding ID

    return faceSamples, ids  # Return the collected images and labels

def trainFaceImage():
    # Path for face image database
    path = '/home/dean/Desktop/face_github_ealon/dataset'
    # Start the training process
    print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces, ids = getImagesAndLabels(path)  # Get images and labels
    recognizer.train(faces, np.array(ids))  # Train the face recognizer
    # Save the trained model
    recognizer.write('/home/dean/Desktop/face_github_ealon/trainer.yml')  # Save the model to a file



while(True):  # Main loop for face capture

    img= picam2.capture_array()  # Capture an image frame from the camera
    img = cv2.flip(img, -1)  # Flip the image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale for face detection
    faces = face_detector.detectMultiScale(gray, 1.3, 5)  # Detect faces in the grayscale image

    for (x,y,w,h) in faces:  # Iterate over each detected face
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)  # Draw a blue rectangle around the face
        count += 1  # Increment the image capture counter

        # Save the face image to the dataset folder
        cv2.imwrite("/home/dean/Desktop/face_github_ealon/dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)  # Display the image with the rectangle

    if count == 400:  # Train image if 400 images are captured
        trainFaceImage()
        break

    if cv2.waitKey(100) & 0xff:  # Exit if the 'ESC' key is pressed
        break

picam2.release()
cv2.destroyAllWindows()  # Close any OpenCV windows
