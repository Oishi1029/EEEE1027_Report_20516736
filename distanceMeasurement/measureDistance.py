# import necessary libraries
from keras.models import load_model
from gpiozero import DigitalInputDevice
import numpy as np
import cv2
from picamera2 import Picamera2
from gpiozero import DigitalOutputDevice
from gpiozero import PWMOutputDevice
from time import sleep

ENCODER_PIN = 18  # Replace with the actual GPIO pin connected to the encoder
WHEEL_CIRCUMFERENCE_CM = 16.65  # wheel circumference constant in centimeters calculated by 2*pi*2.65  where 2.65 is the radius of the wheels in centimeters
encoder = DigitalInputDevice(ENCODER_PIN) # Initialize the encoder

# Initialize variables
rotations = 0
total_distance_cm = 0

# Configure numpy to suppress scientific notation for easier readability
np.set_printoptions(suppress=True)

# Load the pre-trained neural network model from a specified path
model = load_model("/home/dean/Documents/224_pixel_images/Can work using 224/keras_model.h5", compile=False)

# Load the class labels from a text file
class_names = open("/home/dean/Documents/224_pixel_images/Can work using 224/labels.txt", "r").readlines()


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

def handle_rotation():
    """Handles a single rotation event detected by the encoder."""
    global rotations, total_distance_cm  # Access global variables for updates

    rotations += 1 # Increment the rotation count

    if rotations == 20:
        rotations = 0 # Reset the rotation count after 20 rotations
    else:
        total_distance_cm += WHEEL_CIRCUMFERENCE_CM / 20  # Calculate partial distance

encoder.when_activated = handle_rotation  # Assign the handle_rotation function to the encoder

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

def detectSymbol():
    """Function to detect symbols using the pre-trained model."""
    image = picam2.capture_array()  # Capture an image
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)  # Resize the image

    # Prepare the image for prediction
    image_for_prediction = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    image_for_prediction = (image_for_prediction / 127.5) - 1  # Normalize the image

    # Predict the symbol using the model
    prediction = model.predict(image_for_prediction)
    index = np.argmax(prediction)  # Get the index of the highest prediction score

    class_name = class_names[index].strip()  # Get the class name removing any extra spaces or newlines

    return class_name  # Return the detected symbol name

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

def displayDistance():


while True:
    global detected_colors

    imageFrame = picam2.capture_array()

    detectColour(imageFrame)

    if "red" in detected_colors:
        name = detectSymbol()
        if name == "Measure Distance":
            stop()
            sleep(2)

            # display distance travelled
            print(f"Rotations: {rotations}, Total Distance (cm): {total_distance_cm / 15.65:.2f}")

            # overcome symbol
            for i in range(150):
                blackLineFollow()

    else:
        blackLineFollow()


    cv2.imshow('frame', imageFrame)

    detected_colors.clear()

    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close all OpenCV windows
picam2.release()
cv2.destroyAllWindows()
