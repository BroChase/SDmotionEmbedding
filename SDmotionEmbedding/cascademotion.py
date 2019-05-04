import cv2
import time
import imutils
import dlib
from imutils import face_utils


cascadePath = 'smile.xml'
smileCascade = cv2.CascadeClassifier(cascadePath)

detector = dlib.get_frontal_face_detector()

# Looping video capture function.
# Captures video stream
# Detects face crops and detects smile.
# todo Facial Alignment. Before smile detection layer!!


def cascademoiton():
    # Start the video capturing
    cap = cv2.VideoCapture(0)
    # Camera warm up time
    time.sleep(2.0)
    while True:
        ret, img = cap.read()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(img_gray)
        # If face is detected
        w = 0
        h = 0
        if len(faces) >= 1:
            face = faces[0]
            for i in range(len(faces)):
                (_x, _y, _w, _h) = face_utils.rect_to_bb(faces[i])
                if _w > w or _h > h:
                    face = faces[i]
            (x, y, w, h) = face_utils.rect_to_bb(face)
            # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            # todo play with minNeighbors factor. Greater = less sensitive
            smile = smileCascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=16, minSize=(25, 25),
                                                   flags=cv2.CASCADE_SCALE_IMAGE)

            for (sx, sy, sw, sh) in smile:
                print(len(smile))
                cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)

        key = cv2.waitKey(1) & 0xFF
        cv2.imshow('Frame', img)
        # break from loop on q
        if key == ord('q'):
            break


cascademoiton()