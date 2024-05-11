# import necessary libraries
import numpy as np
import cv2
from picamera2 import Picamera2
from gpiozero import DigitalOutputDevice
from gpiozero import PWMOutputDevice
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




while True:

    imageFrame = picam2.capture_array()
    
    blackLineFollow()
    
    cv2.imshow('frame', imageFrame)

    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close all OpenCV windows
picam2.release()
cv2.destroyAllWindows()
