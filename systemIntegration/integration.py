# import necessary libraries
import numpy as np
import cv2
from picamera2 import Picamera2
from gpiozero import DigitalOutputDevice
from gpiozero import PWMOutputDevice
from time import sleep
from keras.models import load_model
from gpiozero import LED
from gpiozero import DigitalInputDevice
from gtts import gTTS
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('/home/dean/Desktop/face_github_ealon/trainer.yml')  # Load a trained face recognizer model
cascadePath = "/home/dean/Desktop/face_github_ealon/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0
names = ['None', 'Ealon', 'Paula', 'Ilza', 'Z', 'W']

preferred_color1 = input("Enter the preferred color (yellow, red, green, blue): ").lower()  # Get input from the user
preferred_color2 = input("Enter the preferred color (yellow, red, green, blue): ").lower()  # Get input from the user

red = LED(11)  # Set up an LED connection to a GPIO pin 11
green = LED(15)  # Set up an LED connection to a GPIO pin 15
blue = LED(16)  # Set up an LED connection to a GPIO pin 16

ENCODER_PIN = 18  # Replace with the actual GPIO pin connected to the encoder
WHEEL_CIRCUMFERENCE_CM = 16.65  # 2*pi*2.65  where 2.65 is the radius of the wheels
encoder = DigitalInputDevice(ENCODER_PIN)
rotations = 0
total_distance_cm = 0

red_lower = np.array([120, 4, 10], np.uint8)
red_upper = np.array([195, 10, 22], np.uint8)

green_lower = np.array([0, 80, 43], np.uint8)
green_upper = np.array([9, 160, 90], np.uint8)

blue_lower = np.array([6, 13, 44], np.uint8)
blue_upper = np.array([11, 25, 90], np.uint8)

# Add black color detection
black_lower = np.array([15, 17, 15], np.uint8)
black_upper = np.array([39, 40, 37], np.uint8)

yellow_lower = np.array([150, 110, 1], np.uint8)
yellow_upper = np.array([230, 210, 35], np.uint8)
kernel = np.ones((5, 5), "uint8")

redSymbol_lower = np.array([109, 4, 10], np.uint8)
redSymbol_upper = np.array([195, 65, 60], np.uint8)

greenSymbol_lower = np.array([0, 50, 15], np.uint8)
greenSymbol_upper = np.array([46, 160, 90], np.uint8)

blueSymbol_lower = np.array([6, 13, 44], np.uint8)
blueSymbol_upper = np.array([31, 60, 162], np.uint8)

# Add black color detection
blackSymbol_lower = np.array([15, 17, 15], np.uint8)
blackSymbol_upper = np.array([39, 40, 37], np.uint8)

yellowSymbol_lower = np.array([180, 136, 1], np.uint8)
yellowSymbol_upper = np.array([230, 210, 21], np.uint8)

purpleSymbol_lower = np.array([70, 30, 80], np.uint8)
purpleSymbol_upper = np.array([130, 70, 160], np.uint8)

modelRed = load_model(  # Load a previously trained model
    "/home/dean/Documents/Week3 Training Image for 700/SYMBOLS WITH LIGHT/newRED/specialSymMODEL/converted_keras (40)/keras_model.h5",
    compile=False)
class_namesRed = open(
    "/home/dean/Documents/Week3 Training Image for 700/SYMBOLS WITH LIGHT/newRED/specialSymMODEL/converted_keras (40)/labels.txt",
    "r").readlines()

modelGreen = load_model(  # Load a previously trained model
    "/home/dean/Documents/Week3 Training Image for 700/SYMBOLS WITH LIGHT/newGREEN/KERASnewerWITHup_down_triangle/converted_keras (36)/keras_model.h5",
    compile=False)
class_namesGreen = open(
    "/home/dean/Documents/Week3 Training Image for 700/SYMBOLS WITH LIGHT/newGREEN/KERASnewerWITHup_down_triangle/converted_keras (36)/labels.txt",
    "r").readlines()
print("general model is used")

modelBlue = load_model(  # Load a previously trained model
    "/home/dean/Documents/Week3 Training Image for 700/SYMBOLS WITH LIGHT/newBLUE/KERASNEW/converted_keras (38)/keras_model.h5",
    compile=False)
class_namesBlue = open(
    "/home/dean/Documents/Week3 Training Image for 700/SYMBOLS WITH LIGHT/newBLUE/KERASNEW/converted_keras (38)/labels.txt",
    "r").readlines()

modelPurple = load_model(  # Load a previously trained model
    "/home/dean/Documents/Week3 Training Image for 700/SYMBOLS WITH LIGHT/PURPLE/converted_keras (39)/keras_model.h5",
    compile=False)
class_namesPurple = open(
    "/home/dean/Documents/Week3 Training Image for 700/SYMBOLS WITH LIGHT/PURPLE/converted_keras (39)/labels.txt",
    "r").readlines()

backwardCounter = 0
# Define GPIO pins
in1_pin = 24
in2_pin = 23
in3_pin = 27
in4_pin = 22
en1_pin = 12
en2_pin = 13

Kp = 0.0080
setpoint = 102.5  # Desired distance from the line  # Import specific modules from a library

# Initialize GPIO devices
in1 = DigitalOutputDevice(in1_pin)  # Set up a digital output device on a specific GPIO pin
in2 = DigitalOutputDevice(in2_pin)  # Set up a digital output device on a specific GPIO pin
in3 = DigitalOutputDevice(in3_pin)  # Set up a digital output device on a specific GPIO pin
in4 = DigitalOutputDevice(in4_pin)  # Set up a digital output device on a specific GPIO pin
en1 = PWMOutputDevice(en1_pin)  # Set up a PWM output device for controlling power output
en2 = PWMOutputDevice(en2_pin)  # Set up a PWM output device for controlling power output

picam2A = Picamera2(0)
picam2A.resolution = (192, 108)
picam2A.framerate = 35
picam2A.preview_configuration.main.size = (192, 108)
picam2A.preview_configuration.main.format = "RGB888"
picam2A.preview_configuration.align()
picam2A.configure("preview")
picam2A.start()
sleep(2)

picam2B = Picamera2(1)
picam2B.resolution = (640, 640)
picam2B.framerate = 35
picam2B.preview_configuration.main.size = (640, 640)
picam2B.preview_configuration.main.format = "RGB888"
picam2B.preview_configuration.align()
picam2B.configure("preview")
picam2B.start()
sleep(2)

# Initialize color sets
detected_colors = set()


def playSound(name):
    print("stopping...")
    mytext = name
    language = 'en'
    myobj = gTTS(text=mytext, lang=language, slow=False)
    myobj.save("welcome.mp3")
    os.system("mpg321 welcome.mp3")
    sleep(5)


def handle_rotation():
    global rotations, total_distance_cm
    rotations += 1
    total_distance_cm += WHEEL_CIRCUMFERENCE_CM


encoder.when_activated = handle_rotation


def backward():
    in1.off()
    in2.on()
    in3.off()
    in4.on()


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
    img = picam2A.capture_array()
    image_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the model's input shape.
    image_for_prediction = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image_for_prediction = (image_for_prediction / 127.5) - 1
    if "green" in detectedColors:
        # Predict the model
        prediction = modelGreen.predict(image_for_prediction)
        index = np.argmax(prediction)
        class_name = class_namesGreen[index]
        num_str = str(index)  # change index type to str
        length_str = len(num_str)  # Get the length of the string
        initial = length_str + 1  # get initial
        class_name = class_name[initial:].strip()
    elif "red" in detectedColors:
        prediction = modelRed.predict(image_for_prediction)
        index = np.argmax(prediction)
        class_name = class_namesRed[index]
        num_str = str(index)  # change index type to str
        length_str = len(num_str)  # Get the length of the string
        initial = length_str + 1  # get initial
        class_name = class_name[initial:].strip()
    elif "blue" in detectedColors:
        prediction = modelBlue.predict(image_for_prediction)
        index = np.argmax(prediction)
        class_name = class_namesBlue[index]
        num_str = str(index)  # change index type to str
        length_str = len(num_str)  # Get the length of the string
        initial = length_str + 1  # get initial
        class_name = class_name[initial:].strip()

    elif "purple" in detectedColors:
        prediction = modelPurple.predict(image_for_prediction)
        index = np.argmax(prediction)
        class_name = class_namesPurple[index]
        num_str = str(index)  # change index type to str
        length_str = len(num_str)  # Get the length of the string
        initial = length_str + 1  # get initial
        class_name = class_name[initial:].strip()
    elif "yellow" in detectedColors:
        class_name = "Yellow Line"

    return class_name


def blackLineFollow():
    global prev_error, integral, derivative

    img = picam2A.capture_array()
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

    img = picam2A.capture_array()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if "yellow" in detected_colors and (preferred_color1 == "yellow" or preferred_color2 == "yellow"):
        mask = cv2.inRange(rgb, yellow_lower, yellow_upper)

    elif "red" in detected_colors and (preferred_color1 == "red" or preferred_color2 == "red"):
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



def detectFace():
    img = picam2B.capture_array()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(200, 200),
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        if (confidence < 35):
            face_name = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            face_name = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(face_name), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    return face_name


def detectColour(imageFrame):
    global detected_colors
    
    rgbFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2RGB)
    red_mask = cv2.inRange(rgbFrame, redSymbol_lower, redSymbol_upper)
    green_mask = cv2.inRange(rgbFrame, greenSymbol_lower, greenSymbol_upper)
    blue_mask = cv2.inRange(rgbFrame, blueSymbol_lower, blueSymbol_upper)
    yellow_mask = cv2.inRange(rgbFrame, yellowSymbol_lower, yellowSymbol_upper)
    black_mask = cv2.inRange(rgbFrame, blackSymbol_lower, blackSymbol_upper)
    purple_mask = cv2.inRange(rgbFrame, purpleSymbol_lower, purpleSymbol_upper)

    kernel = np.ones((5, 5), "uint8")

    red_mask = cv2.dilate(red_mask, kernel)
    green_mask = cv2.dilate(green_mask, kernel)
    blue_mask = cv2.dilate(blue_mask, kernel)
    black_mask = cv2.dilate(black_mask, kernel)
    yellow_mask = cv2.dilate(yellow_mask, kernel)
    purple_mask = cv2.dilate(purple_mask, kernel)

    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            detected_colors.add("red")

    contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            detected_colors.add("green")

    contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            detected_colors.add("blue")

    contours, hierarchy = cv2.findContours(black_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            detected_colors.add("black")
    contours, hierarchy = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            detected_colors.add("yellow")

    contours, hierarchy = cv2.findContours(purple_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            detected_colors.add("purple")


def perform_FaceRecognition():
    blue.on()
    print("Face Recognition Symbol Detected !")
    print("detecting face......")
    sleep(2)
    blue.off()
    while 1:
        people = detectFace()
        print(people, "detected")
        if people == "Ealon":
            print("User detected !")
            red.on()
            green.on()
            blue.on()
            sleep(2.5)
            break
        else:
            red.on()

        red.off()
        green.off()
        blue.off()
        print("Overcome Face Recognition Symbol !")
        for i in range(100):
            blackLineFollow()


def perform_Handstop():
    perform_Handstop()
    print("handstopping......")
    red.on()
    green.on()
    sleep(2)
    red.off()
    green.off()
    print(symbolName + " detected !" + ", stop for awhile.....")
    sleep(4)
    print("Overcome symbol !")
    forward()
    sleep(0.3)
    for i in range(100):
        blackLineFollow()


def displayLED():  # Set up an LED connection to a GPIO pin
    global detected_colors

    if "red" in detected_colors:
        red.on()
    elif "green" in detected_colors:
        green.on()
    elif "blue" in detected_colors or "purple" in detected_colors:
        blue.on()


try:
    while (1):
        global detected_colors, preferred_color1, preferred_color2
        imageFrame = picam2A.capture_array()
        detectColour(imageFrame)

        displayLED() 
        print(detected_colors)
        print(len(detected_colors))

        if len(detected_colors) > 1:
            stop()
            sleep(1.5)
            symbolName = detectSymbol()
            playSound(symbolName)
            lineList = ["Line", "Black Line", "Yellow Line", "Green Line", "Red Line", "Blue Line", "Blank"]
            if symbolName not in lineList:
                if symbolName == "Face Recognition":
                    perform_FaceRecognition()

                elif symbolName == "Handstop":  # detected Handstop
                    perform_Handstop()

                elif symbolName == "Stop":
                    stop()
                    sleep(2)

                elif symbolName == "Measure Distance":
                    print(f"Rotations: {rotations}, Total Distance (cm): {total_distance_cm / 15.65:.2f}")
                    sleep(2)
            else:
                colorLineFollow()

        else:
            if preferred_color1 in detected_colors or preferred_color2 in detected_colors:
                colorLineFollow()
            else:
                blackLineFollow()

        red.off()
        green.off()
        blue.off()
        detected_colors.clear()

        cv2.imshow('frame', imageFrame)  # Display the image
        if cv2.waitKey(1) == ord('q'):
            break

except KeyboardInterrupt:
    print("Keyboard interrupt detected. Stopping motors.")
    stop()
finally:
    # Cleanup GPIO pins
    in1.close()
    in2.close()
    in3.close()
    in4.close()
    en1.close()
    en2.close()
    print("Motors stopped. Program terminated.")

