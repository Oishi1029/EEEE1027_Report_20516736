# Import necessary libraries
import cv2
from picamera2 import Picamera2
import time
from gpiozero import LED
import numpy as np

# Initialize the camera using the Picamera2 library
picam2 = Picamera2()
picam2.preview_configuration.main.size = (224, 224)  # Set the camera resolution
picam2.preview_configuration.main.format = "RGB888"  # Set the camera color format
picam2.preview_configuration.align()  # Align the camera configuration
picam2.configure("preview")  # Set the camera to preview mode
picam2.start()  # Start the camera
time.sleep(2)  # Wait for 2 seconds to let the camera warm up

# Initialize LED pins for different colors
red = LED(11)
green = LED(15)
blue = LED(16)

detected_colors = set()

def detectColour(imageFrame):
    """Function to detect colors in the captured video frame."""
    global detected_colors
    rgbFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2RGB)  # Convert frame from BGR to RGB color space

    # Define the color range for red, green, and blue
    red_lower = np.array([120, 4, 10], np.uint8)
    red_upper = np.array([195, 10, 22], np.uint8)

    green_lower = np.array([0, 80, 43], np.uint8)
    green_upper = np.array([9, 160, 90], np.uint8)

    blue_lower = np.array([6, 13, 44], np.uint8)
    blue_upper = np.array([11, 25, 90], np.uint8)

    # Create masks for each color
    red_mask = cv2.inRange(rgbFrame, red_lower, red_upper)
    green_mask = cv2.inRange(rgbFrame, green_lower, green_upper)
    blue_mask = cv2.inRange(rgbFrame, blue_lower, blue_upper)

    # Dilate the masks to increase the size of the regions
    kernel = np.ones((5, 5), "uint8")
    red_mask = cv2.dilate(red_mask, kernel)
    green_mask = cv2.dilate(green_mask, kernel)
    blue_mask = cv2.dilate(blue_mask, kernel)

    # Detect contours for red color
    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            cv2.putText(imageFrame, "Red Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
            detected_colors.add("red")

    # Repeat contour detection for green and blue colors
    contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            cv2.putText(imageFrame, "Green Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
            detected_colors.add("green")

    contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            cv2.putText(imageFrame, "Blue Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))
            detected_colors.add("blue")
def showLED():
    if "red" in detected_colors:
        red.on()
    if "green" in detected_colors:
        green.on()
    if "blue" in detected_colors:
        blue.on()


def turnOffLED():
    red.off()
    green.off()
    blue.off()

while True:
    imageFrame = picam2.capture_array()
    detectColour(imageFrame)  # Detect colors in the image
    showLED()
    time.sleep(2)
    cv2.imshow('frame', imageFrame)
    turnOffLED()
    if cv2.waitKey(1) == ord('q'):
        break


# Release the camera and close all OpenCV windows
picam2.release()
cv2.destroyAllWindows()
