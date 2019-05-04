import cv2
import imutils
import dlib
from imutils import face_utils


cascadePath = 'smile.xml'
smileCascade = cv2.CascadeClassifier(cascadePath)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def motion():
    cap = cv2.VideoCapture(0)
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS['mouth']
    (jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS['jaw']
    (lbStart, lbEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eyebrow']
    (rbStart, rbEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eyebrow']
    (nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS['nose']

    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        subjects = detector(gray, 0)
        for subject in subjects:
            shape = predictor(gray, subject)
            shape = face_utils.shape_to_np(shape)

            print(nStart, nEnd)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]
            jaw = shape[jStart:jEnd]
            leftbrow = shape[lbStart:lbEnd]
            rightbrow = shape[rbStart:rbEnd]
            nose = shape[nStart:nEnd]

            leftEyeHull = cv2.convexHull(leftEye)
            rightEeyeHull = cv2.convexHull(rightEye)
            mouthHull = cv2.convexHull(mouth)
            jawHull = cv2.convexHull(jaw)
            lbrowHull = cv2.convexHull(leftbrow)
            rbrowHull = cv2.convexHull(rightbrow)
            noseHull = cv2.convexHull(nose)

            cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(img, [rightEeyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(img, [mouthHull], -1, (0, 255, 0), 1)
            # cv2.drawContours(img, [jawHull], -1, (0, 255, 0), 1)
            cv2.drawContours(img, [lbrowHull], -1, (0, 255, 0), 1)
            cv2.drawContours(img, [rbrowHull], -1, (0, 255, 0), 1)
            cv2.drawContours(img, [noseHull], -1, (0, 255, 0), 1)

        cv2.imshow('Frame', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

motion()
