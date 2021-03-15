from main2 import frame

import cv2
import time
import numpy as np

frameSize = (1280,720)
out = cv2.VideoWriter('output_video_2.avi',cv2.VideoWriter_fourcc(*'DIVX'), 40, frameSize)
make_video = True
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 10) # At least five neighbor zones in order to classify an image
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h,x:x+w ]
        roi_color = frame[y:y+h,x:x+w ]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            roi_gray_be = gray[y+eh: y + h, x:x+w]  # region of interest below eyes
            roi_color_be = frame[y+eh: y + h, x:x+w]  # region of interest below eyes
            smile = smile_cascade.detectMultiScale(roi_gray_be,1.7, 22)
            for (sx, sy, sw, sh) in smile:
                cv2.rectangle( roi_color_be, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
    return frame


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade= cv2.CascadeClassifier("haarcascade_eye.xml")
smile_cascade= cv2.CascadeClassifier("haarcascade_smile_detection.xml")
fps = 30
#video_name = "video_project2.mp4"
video_name = "inv3.mp4"
cap = cv2.VideoCapture(video_name)
frame_cnt = 0

# Color space of object to grab
yellow_lower_limit = (1, 120, 100)
yellow_upper_limit = (30, 255, 255)


if cap.isOpened() == False:
    print("ERROR FILE NOT FOUND OR WRONG CODEC USED !")

first_frame = True

while cap.isOpened():

    second_passed = frame_cnt/fps
    ret,img = cap.read()
    img2=img
    if first_frame:

        background = img
        background = np.flip(background, axis=1)
        first_frame = False

    if ret == True:

        #Play at normal speed
        time.sleep (1/fps)

        # Create frame object
        img = np.flip(img, axis=1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        value = (35, 35)
        blurred = cv2.GaussianBlur(hsv, value, 0)
        lower_red = np.array([1, 120, 100])
        upper_red = np.array([30, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        mask = mask1
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        img[np.where(mask == 255)] = background[np.where(mask == 255)]

        # Face,eye smile detection
        gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        canvas = detect(gray, img2)

        # Write to video writer
        if make_video:
            out.write(canvas)

        cv2.imshow('Display', canvas)
        k = cv2.waitKey(10)
        if k == ord('q'):
            break

out.release()
cap.release()
cv2.destroyAllWindows()