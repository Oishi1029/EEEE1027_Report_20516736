# import necessary libraries
import numpy as np
import cv2
from picamera2 import Picamera2
from gpiozero import DigitalOutputDevice
from gpiozero import PWMOutputDevice
from time import sleep

red_lower = np.array([120, 4, 10], np.uint8)
red_upper = np.array([195, 10, 22], np.uint8)

green_lower = np.array([0, 80, 43], np.uint8)
green_upper = np.array([9, 160, 90], np.uint8)

blue_lower = np.array([6, 13, 44], np.uint8)
blue_upper = np.array([11, 25, 90], np.uint8)
kernel = np.ones((5, 5), "uint8")

detected_colors = set()
preferred_color1 = input("Enter the preferred color (yellow, red, green, blue): ").lower()  # Get input from the user
preferred_color2 = input("Enter the preferred color (yellow, red, green, blue): ").lower()  # Get input from the user

picam2 = Picamera2()
picam2.resolution = (192, 108)
picam2.framerate = 35
picam2.preview_configuration.main.size = (192, 108)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
sleep(2)

in1_pin = 24
in2_pin = 23
in3_pin = 27
in4_pin = 22
en1_pin = 12
en2_pin = 13

in1 = DigitalOutputDevice(in1_pin)  # Set up a digital output device on a specific GPIO pin
in2 = DigitalOutputDevice(in2_pin)  # Set up a digital output device on a specific GPIO pin
in3 = DigitalOutputDevice(in3_pin)  # Set up a digital output device on a specific GPIO pin
in4 = DigitalOutputDevice(in4_pin)  # Set up a digital output device on a specific GPIO pin
en1 = PWMOutputDevice(en1_pin)  # Set up a PWM output device for controlling power output
en2 = PWMOutputDevice(en2_pin)  # Set up a PWM output device for controlling power output

Kp = 0.0080
setpoint = 102.5  # Desired distance from the line  # Import specific modules from a library

def forward():
    in1.on()
    in2.off()
    in3.on()
    in4.off()


def stop():
    in1.on()
    in2.on()
    in3.on()
    in4.on()

def right():
    in1.on()
    in2.off()
    in3.off()
    in4.on()


def left():
    in1.off()
    in2.on()
    in3.on()
    in4.off()

def blackLineFollow():
    global prev_error, integral, derivative

    img = picam2.capture_array()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    black_lower = np.array([10, 20, 20], np.uint8)
    black_upper = np.array([50, 60, 50], np.uint8)
    mask = cv2.inRange(rgb, black_lower, black_upper)
    contours, hierarchy = cv2.findContours(mask.copy(), 1, cv2.CHAIN_APPROX_NONE)

    # Find the biggest contour (if detected)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])

            error = cx - setpoint
            output = Kp * error

            if cx <= 40:
                print("Turn Left!")
                left()
                en1.value = max(0, min(0.38 - output, 0.75))
                en2.value = max(0, min(0.38 + output, 0.75))
            elif cx >= 150:
                print("Turn Right")
                right()
                en1.value = max(0, min(0.38 - output, 0.75))
                en2.value = max(0, min(0.38 + output, 0.75))

            else:
                print("On Track!")
                forward()
                en1.value = max(0, min(0.35 - output, 0.4))
                en2.value = max(0, min(0.35 + output, 0.4))

    else:
        stop()

def colorLineFollow():
    global detected_colors, preferred_color1, preferred_color2

    img = picam2.capture_array()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if "red" in detected_colors and (preferred_color1 == "red" or preferred_color2 == "red"):
        mask = cv2.inRange(rgb, red_lower, red_upper)

    elif "green" in detected_colors and (preferred_color1 == "green" or preferred_color2 == "green"):
        mask = cv2.inRange(rgb, green_lower, green_upper)

    elif "blue" in detected_colors and (preferred_color1 == "blue" or preferred_color2 == "blue"):
        mask = cv2.inRange(rgb, blue_lower, blue_upper)

    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
        else:
            cx = 0

        if cx <= 40:
            print("Colour Turn Left!")
            left()
            en1.value = 0.65
            en2.value = 0.65

        elif 40 < cx < 150:
            print("Colour On Track!")
            forward()
            en1.value = 0.4
            en2.value = 0.4
        else:
            print("Colour Turn Right")
            right()
            en1.value = 0.65
            en2.value = 0.65
    else:
        blackLineFollow()
        
def detectColour(imageFrame):
    global detected_colors

    rgbFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2RGB)

    red_mask = cv2.inRange(rgbFrame, red_lower, red_upper)
    green_mask = cv2.inRange(rgbFrame, green_lower, green_upper)
    blue_mask = cv2.inRange(rgbFrame, blue_lower, blue_upper)


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



while True:
    global detected_colors
    
    imageFrame = picam2.capture_array()
    
    detectColour(imageFrame)
    
    colorLineFollow()
    
    cv2.imshow('frame', imageFrame)
    
    detected_colors.clear()
    
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close all OpenCV windows
picam2.release()
cv2.destroyAllWindows()


