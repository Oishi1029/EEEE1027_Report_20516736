# Import necessary libraries
from keras.models import load_model
import cv2
import numpy as np
from picamera2 import Picamera2
import time


# Initialize the camera using the Picamera2 library
picam2 = Picamera2()
picam2.preview_configuration.main.size = (224, 224)  # Set the camera resolution
picam2.preview_configuration.main.format = "RGB888"  # Set the camera color format
picam2.preview_configuration.align()  # Align the camera configuration
picam2.configure("preview")  # Set the camera to preview mode
picam2.start()  # Start the camera
time.sleep(2)  # Wait for 2 seconds to let the camera warm up

# Configure numpy to suppress scientific notation for easier readability
np.set_printoptions(suppress=True)

# Load the pre-trained neural network model from a specified path
model = load_model("/home/dean/Documents/224_pixel_images/Can work using 224/keras_model.h5", compile=False)

# Load the class labels from a text file
class_names = open("/home/dean/Documents/224_pixel_images/Can work using 224/labels.txt", "r").readlines()

def detectSymbol(image):
    """Function to detect symbols using the pre-trained model."""
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)  # Resize the image

    # Prepare the image for prediction
    image_for_prediction = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    image_for_prediction = (image_for_prediction / 127.5) - 1  # Normalize the image

    # Predict the symbol using the model
    prediction = model.predict(image_for_prediction)
    index = np.argmax(prediction)  # Get the index of the highest prediction score

    class_name = class_names[index].strip()  # Get the class name removing any extra spaces or newlines

    return class_name  # Return the detected symbol name


# Main loop to continuously detect symbols and colors
while True:
    imageFrame = picam2.capture_array()
    
    name = detectSymbol(imageFrame)  # Detect symbols in the image

    print(name, "detected !")
    
    cv2.imshow('camera', imageFrame)  # Display the image

    k = cv2.waitKey(10) & 0xff
    if k == 27:  # Press 'ESC' to quit
        break

# Release the camera and close all OpenCV windows
picam2.release()
cv2.destroyAllWindows()
