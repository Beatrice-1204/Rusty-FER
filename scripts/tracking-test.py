import cv2
import time
from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

i2c = busio.I2C(SCL, SDA)

pca = PCA9685(i2c)
pca.frequency = 50

pan = servo.Servo(pca.channels[0])
tilt = servo.Servo(pca.channels[1])

pan_angle = 90
tilt_angle = 90

pan.angle = pan_angle
tilt.angle = tilt_angle

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    h, w = frame.shape[:2]
    center_x = w // 2
    center_y = h // 2

    for (x, y, fw, fh) in faces:
        face_x = x + fw // 2
        face_y = y + fh // 2

        if face_x < center_x - 30:
            pan_angle += 2
        elif face_x > center_x + 30:
            pan_angle -= 2

        if face_y < center_y - 30:
            tilt_angle -= 2
        elif face_y > center_y + 30:
            tilt_angle += 2

        pan_angle = max(0, min(180, pan_angle))
        tilt_angle = max(0, min(180, tilt_angle))

        pan.angle = pan_angle
        tilt.angle = tilt_angle

        cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0,255,0), 2)

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()