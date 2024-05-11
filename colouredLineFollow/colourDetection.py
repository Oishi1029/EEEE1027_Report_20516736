# import necessary libraries
import numpy as np
import cv2
from picamera2 import Picamera2
from time import sleep

picam2 = Picamera2()
picam2.resolution = (192, 108)
picam2.framerate = 35
picam2.preview_configuration.main.size = (192, 108)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
sleep(2)

detected_colors = set()

while True:
    global detected_colors
    
    imageFrame = picam2.capture_array()

    rgbFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2RGB)

    red_lower = np.array([120, 4, 10], np.uint8)
    red_upper = np.array([195, 10, 22], np.uint8)

    green_lower = np.array([0, 80, 43], np.uint8)
    green_upper = np.array([9, 160, 90], np.uint8)

    blue_lower = np.array([6, 13, 44], np.uint8)
    blue_upper = np.array([11, 25, 90], np.uint8)

    red_mask = cv2.inRange(rgbFrame, red_lower, red_upper)
    green_mask = cv2.inRange(rgbFrame, green_lower, green_upper)
    blue_mask = cv2.inRange(rgbFrame, blue_lower, blue_upper)

    kernel = np.ones((5, 5), "uint8")

    red_mask = cv2.dilate(red_mask, kernel)
    green_mask = cv2.dilate(green_mask, kernel)
    blue_mask = cv2.dilate(blue_mask, kernel)

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
            
    print(detected_colors)

    cv2.imshow("Red Mask", red_mask)
    cv2.imshow("Green Mask", green_mask)
    cv2.imshow("Blue Mask", blue_mask)

    detected_colors.clear()
    
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close all OpenCV windows
picam2.release()
cv2.destroyAllWindows()


